import time

import torch
import torch.utils.data

from kge.job import Job
from kge.job.train import TrainingJob, _generate_worker_init_fn

SLOTS = [0, 1, 2]
S, P, O = SLOTS


class TrainingJob1vsAllHybrid(TrainingJob):
    """
    Samples SPO pairs and queries sp_ and _po, treating all other entities as
    negative. Supports different query types, e.g. sp?, s*?, etc.
    """

    def __init__(
        self, config, dataset, parent_job=None, model=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only
        )
        config.log("Initializing 1vsAll_hybrid training job...")
        self.type_str = "1vsAll_hybrid"

        if self.__class__ == TrainingJob1vsAllHybrid:
            for f in Job.job_created_hooks:
                f(self)

        # Slot constants for different query types
        # Semantics: {target_slot: ((input_slot_1, input_slot_2), target_slot)}
        self._query_type_slots = {
            S: ((1, 2), 0),
            P: ((0, 2), 1),
            O: ((0, 1), 2),
        }

        # Score functions for different query types
        # Semantics: {target_slot: score_function}
        self._query_type_score_func = {
            S: self.model.score_po,
            P: self.model.score_so,
            O: self.model.score_sp,
        }

    def _prepare(self):
        """Construct dataloader"""
        super()._prepare()

        # get query weights
        self._query_weights = self.config.get("1vsAll_hybrid.query_weights")

        # determine enabled query types
        self._query_types = self.config.get("1vsAll_hybrid.query_types")
        self._query_types = [
            query for query, enabled in self._query_types.items() if enabled
        ]

        self.num_examples = self.dataset.split(self.train_split).size(0)
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=lambda batch: {
                "triples": self.dataset.split(self.train_split)[batch, :].long()
            },
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )

    def _prepare_batch(
        self, batch_index, batch, result: TrainingJob._ProcessBatchResult
    ):
        result.size = len(batch["triples"])

    def _process_subbatch(
        self,
        batch_index,
        batch,
        subbatch_slice,
        result: TrainingJob._ProcessBatchResult,
    ):

        # prepare
        result.prepare_time -= time.time()
        triples = batch["triples"][subbatch_slice].to(self.device)
        batch_size = result.size
        result.prepare_time += time.time()

        for query_type in self._query_types:
            result.prepare_time -= time.time()
            # get score function for current query type
            target_slot = query_type.find("_")
            score_func = self._query_type_score_func[target_slot]
            # get weight for current query type or use 1.0 as default
            query_weight = self._query_weights.get(query_type, 1.0)

            # get input and target indices from batch
            input_indices = [
                triples[:, self._query_type_slots[target_slot][0][0]],
                triples[:, self._query_type_slots[target_slot][0][1]],
            ]
            target_indices = triples[:, self._query_type_slots[target_slot][1]]

            # use placeholder embeddings if query type asks for it
            placeholder_slot = query_type.find("^")
            if placeholder_slot != -1:
                # get position of slot to replace
                slot_position = self._query_type_slots[target_slot][0].index(
                    placeholder_slot
                )
                # replace slot with placeholder embedding
                input_indices[slot_position] = torch.tensor(
                    self.model.pseudo_indices[placeholder_slot]
                ).expand(input_indices[slot_position].size(0)).to(self.device)
            result.prepare_time += time.time()

            # forward/backward pass (sp)
            result.forward_time -= time.time()
            scores = score_func(input_indices[0], input_indices[1])
            loss_value_query = self.loss(scores, target_indices)
            loss_value_query = query_weight * (loss_value_query / batch_size)
            result.avg_loss += loss_value_query.item()
            result.forward_time += time.time()
            result.backward_time = -time.time()
            if not self.is_forward_only:
                loss_value_query.backward()
            result.backward_time += time.time()
