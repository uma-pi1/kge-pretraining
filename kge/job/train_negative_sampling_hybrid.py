import copy
import time
import torch
import torch.utils.data

from kge.job import Job
from kge.job.train import TrainingJob, _generate_worker_init_fn
from kge.util import KgeSampler
from kge.model.transe import TransEScorer
from kge.job.util import QueryType

SLOTS = [0, 1, 2]
S, P, O = SLOTS
SLOT_STR = ["s", "p", "o"]


class TrainingJobNegativeSamplingHybrid(TrainingJob):
    def __init__(
        self, config, dataset, parent_job=None, model=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only
        )
        self._sampler = KgeSampler.create(
            config, "negative_sampling_hybrid", dataset
        )
        self.type_str = "negative_sampling_hybrid"

        if self.__class__ == TrainingJobNegativeSamplingHybrid:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        super()._prepare()
        # select negative sampling implementation
        self._implementation = self.config.check(
            "negative_sampling_hybrid.implementation",
            ["triple", "all", "batch", "auto"],
        )
        if self._implementation == "auto":
            max_nr_of_negs = max(self._sampler.num_samples)
            if self._sampler.shared:
                self._implementation = "batch"
            elif max_nr_of_negs <= 30:
                self._implementation = "triple"
            else:
                self._implementation = "batch"
            self.config.set(
                "negative_sampling.implementation",
                self._implementation,
                log=True
            )

        self.config.log(
            "Preparing negative sampling hybrid training job with "
            "'{}' scoring function ...".format(self._implementation)
        )

        # determine enabled query types
        self._query_types = self.config.get(
            "negative_sampling_hybrid.query_types"
        )
        self._query_types = [
            QueryType(query_type, value) for query_type, value in self._query_types.items() if value > 0.0
        ]
        if not len(self._query_types):
            raise ValueError(
                "Negative Sampling Hybrid has no enabled query types."
            )

        # construct dataloader
        self.num_examples = self.dataset.split(self.train_split).size(0)
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=self._get_collate_fun(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )

    def _get_collate_fun(self):
        # create the collate function
        def collate(batch):
            """For a batch of size n, returns a tuple of:

            - triples (tensor of shape [n,3], ),
            - negative_samples (list of tensors of shape [n,num_samples]; 3
                elements in order S,P,O)
            """

            triples = self.dataset.split(self.train_split)[batch, :].long()

            negative_samples = list()
            for slot in [S, P, O]:
                negative_samples.append(self._sampler.sample(triples, slot))
            return {"triples": triples, "negative_samples": negative_samples}

        return collate

    def _prepare_batch(
        self, batch_index, batch, result: TrainingJob._ProcessBatchResult
    ):
        # move triples and negatives to GPU. With some implementaiton effort,
        # this may be avoided.
        result.prepare_time -= time.time()
        batch["triples"] = batch["triples"].to(self.device)
        for ns in batch["negative_samples"]:
            ns.positive_triples = batch["triples"]
        batch["negative_samples"] = [
            ns.to(self.device) for ns in batch["negative_samples"]
        ]

        batch["labels"] = [None] * 3  # reuse label tensors b/w subbatches
        result.size = len(batch["triples"])
        result.prepare_time += time.time()

    def _process_subbatch(
        self,
        batch_index,
        batch,
        subbatch_slice,
        result: TrainingJob._ProcessBatchResult,
    ):
        batch_size = result.size

        # prepare
        result.prepare_time -= time.time()
        triples = batch["triples"][subbatch_slice]
        batch_negative_samples = batch["negative_samples"]
        subbatch_size = len(triples)
        result.prepare_time += time.time()
        labels = batch["labels"]  # reuse b/w subbatches

        # process the subbatch for each query type separately
        for query_type in self._query_types:
            target_slot = query_type.target_slot
            num_samples = self._sampler.num_samples[target_slot]
            if num_samples <= 0:
                continue

            # construct gold labels: first column corresponds to positives,
            # remaining columns to negatives
            if labels[target_slot] is None or labels[target_slot].shape != (
                    subbatch_size,
                    1 + num_samples,
            ):
                result.prepare_time -= time.time()
                labels[target_slot] = torch.zeros(
                    (subbatch_size, 1 + num_samples), device=self.device
                )
                labels[target_slot][:, 0] = 1
                result.prepare_time += time.time()

            result.prepare_time -= time.time()
            # get input indices from batch
            input_indices = [
                triples[:, query_type.input_slots[0]],
                triples[:, query_type.input_slots[1]],
            ]

            # use placeholder embeddings if query type asks for it
            placeholder_slot = query_type.name.find("^")
            if placeholder_slot != -1:
                # get position of slot to replace
                replaced_input_slot = query_type.input_slots.index(
                    placeholder_slot
                )
                # replace slot with placeholder embedding
                input_indices[replaced_input_slot] = torch.tensor(
                    self.model.pseudo_indices[placeholder_slot]
                ).expand(input_indices[replaced_input_slot].size(0)).to(
                    self.device
                )
            result.prepare_time += time.time()

            # compute the scores
            result.forward_time -= time.time()
            scores = torch.empty((subbatch_size, num_samples + 1),
                                 device=self.device)
            triples_to_score = copy.deepcopy(triples)
            # apply placeholder embedding to positives if needed
            if placeholder_slot != -1:
                triples_to_score[:, placeholder_slot] = input_indices[
                    replaced_input_slot
                ]
            scores[:, 0] = self.model.score_spo(
                triples_to_score[:, S],
                triples_to_score[:, P],
                triples_to_score[:, O],
                direction=SLOT_STR[target_slot],
            )
            result.forward_time += time.time()
            scores[:, 1:] = batch_negative_samples[target_slot].score(
                self.model, indexes=subbatch_slice, query_type=query_type
            )
            result.forward_time += \
                batch_negative_samples[target_slot].forward_time
            result.prepare_time += \
                batch_negative_samples[target_slot].prepare_time

            # compute loss for slot in subbatch (concluding the forward pass)
            result.forward_time -= time.time()
            loss_value_torch = (
                    self.loss(
                        scores, labels[target_slot], num_negatives=num_samples
                    ) / batch_size
            )
            result.avg_loss += loss_value_torch.item()
            result.forward_time += time.time()

            # backward pass for this slot in the subbatch
            result.backward_time -= time.time()
            if not self.is_forward_only:
                loss_value_torch.backward()
            result.backward_time += time.time()
