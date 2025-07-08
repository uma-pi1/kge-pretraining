import time
import torch
import torch.utils.data
from typing import List

from kge.job import Job
import kge.job.util
from kge.job.util import QueryType
from kge.job.train import TrainingJob, _generate_worker_init_fn
from kge.indexing import IndexWrapper, index_KvsAll

SLOTS = [0, 1, 2]
S, P, O = SLOTS
SLOT_STR = ["s", "p", "o"]


class TrainingJobKvsAllHybrid(TrainingJob):
    """Train with examples consisting of a query and its answers.

    Terminology:
    - Query type: which queries to ask (sp_, s_o, and/or _po), can be configured via
      configuration key `KvsAll.query_type` (which see)
    - Query: a particular query, e.g., (John,marriedTo) of type sp_
    - Labels: list of true answers of a query (e.g., [Jane])
    - Example: a query + its labels, e.g., (John,marriedTo), [Jane]
    """

    from kge.indexing import KvsAllIndex

    def __init__(
        self, config, dataset, parent_job=None, model=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only
        )
        self.label_smoothing = config.check_range(
            "KvsAll_hybrid.label_smoothing", float("-inf"), 1.0, max_inclusive=False
        )
        if self.label_smoothing < 0:
            if config.get("train.auto_correct"):
                config.log(
                    "Setting label_smoothing to 0, "
                    "was set to {}.".format(self.label_smoothing)
                )
                self.label_smoothing = 0
            else:
                raise Exception(
                    "Label_smoothing was set to {}, "
                    "should be at least 0.".format(self.label_smoothing)
                )
        elif self.label_smoothing > 0 and self.label_smoothing <= (
            1.0 / dataset.num_entities()
        ):
            if config.get("train.auto_correct"):
                # just to be sure it's used correctly
                config.log(
                    "Setting label_smoothing to 1/num_entities = {}, "
                    "was set to {}.".format(
                        1.0 / dataset.num_entities(), self.label_smoothing
                    )
                )
                self.label_smoothing = 1.0 / dataset.num_entities()
            else:
                raise Exception(
                    "Label_smoothing was set to {}, "
                    "should be at least {}.".format(
                        self.label_smoothing, 1.0 / dataset.num_entities()
                    )
                )

        config.log("Initializing 1-to-N training job...")
        self.type_str = "KvsAll_hybrid"

        if self.__class__ == TrainingJobKvsAllHybrid:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        super()._prepare()

        # determine enabled query types
        self._query_types = self.config.get("KvsAll_hybrid.query_types")
        self._query_types = [
            QueryType(query_type, value) for query_type, value in self._query_types.items() if value > 0.0
        ]

        # determine enabled multihop query types
        multihop_query_types = self.config.get("KvsAll_hybrid.multihop_query_types")
        if multihop_query_types:
            for query_type, value in multihop_query_types.items():
                if value > 0.0:
                    self._query_types.append(QueryType(query_type, value))

        if not len(self._query_types):
            raise ValueError("KvsAll hybrid has no enabled query types.")

        # precompute indexes for each query type
        # self._query_indexes: List[KvsAllIndex] = []
        self.num_examples = 0
        self._query_indexes = {}
        #' for each query type (ordered as in self.query_types), index right after last
        #' example of that type in the list of all examples (over all query types)
        # dict of form {query_type_name: last example}
        self._query_last_example = {}
        for query_type in self._query_types:
            index_name = f"{self.train_split}_{query_type.index}"
            # create index function if not already there
            if index_name not in self.dataset.index_functions:
                key_cols, val_cols = query_type.get_key_value_cols()
                num_hops = 1
                # check if query type is for more than 1 hop
                if query_type.name[-1].isnumeric():
                    num_hops = int(query_type.name[-1])
                self.dataset.index_functions[index_name] = IndexWrapper(
                    index_KvsAll,
                    split=self.train_split,
                    key=key_cols,
                    value=val_cols,
                    num_hops=num_hops,
                )
            index = self.dataset.index(index_name)
            self.num_examples += len(index)
            self._query_indexes[query_type.name] = index
            self._query_last_example[query_type.name] = self.num_examples

        # create dataloader
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
            """For a batch of size n, returns a dictionary of:

            - queries: nx2 tensor, row = query (sp, po, or so indexes)
            - label_coords: for each query, position of true answers (an Nx2 tensor,
              first columns holds query index, second colum holds index of label)
            - query_type_indexes (vector of size n holding the query type of each query)
            - triples (all true triples in the batch; e.g., needed for weighted
              penalties)

            """

            # count how many labels we have across the entire batch
            num_ones = 0
            for example_index in batch:
                start = 0
                for query_type in self._query_types:
                    query_type_index = self._query_indexes[query_type.name]
                    end = self._query_last_example[query_type.name]
                    if example_index < end:
                        example_index -= start
                        num_ones += query_type_index._values_offset[
                            example_index + 1
                        ]
                        num_ones -= query_type_index._values_offset[
                            example_index
                        ]
                        break
                    start = end

            # now create the batch elements
            # stores actual queries in batch, e.g. an instance of sp
            queries_batch = torch.zeros([len(batch), 2], dtype=torch.long)
            # map example in batch (by position in tensor) to corresponding index (by value in tensor)
            query_type_indexes_batch = torch.zeros([len(batch)], dtype=torch.long)
            # first col: row position in batch, second cold: col position in num_entities
            label_coords_batch = torch.zeros([num_ones, 2], dtype=torch.int)
            # positive triples in batch, e.g. each instance of an sp tuple
            #   and all its corresponding os make up set of triples
            triples_batch = torch.zeros([num_ones, 3], dtype=torch.long)
            current_index = 0
            for batch_index, example_index in enumerate(batch):
                start = 0
                for query_type_index, query_type in enumerate(self._query_types):
                    end = self._query_last_example[query_type.name]
                    if example_index < end:
                        example_index -= start
                        query_type_indexes_batch[batch_index] = query_type_index
                        queries = self._query_indexes[query_type.name]._keys
                        label_offsets = self._query_indexes[query_type.name]._values_offset
                        labels = self._query_indexes[query_type.name]._values
                        query_col_1 = query_type.input_slots[0]
                        query_col_2 = query_type.input_slots[1]
                        target_col = query_type.target_slot
                        placeholder_slot = query_type.name.find("^")
                        break
                    start = end
                queries_batch[batch_index,] = queries[example_index]
                start = label_offsets[example_index]
                end = label_offsets[example_index + 1]
                size = end - start
                label_coords_batch[
                    current_index : (current_index + size), 0
                ] = batch_index
                label_coords_batch[current_index : (current_index + size), 1] = labels[start:end]

                # create triples for weighted regularization
                for query_col in [query_col_1, query_col_2]:
                    if query_col == placeholder_slot:
                        # use ANY embedding index
                        any_index = self.model.pseudo_indices[placeholder_slot]
                        triples_batch[
                        current_index: (current_index + size), query_col
                        ] = any_index
                    else:
                        triples_batch[
                        current_index: (current_index + size), query_col
                        ] = queries[example_index][0]
                triples_batch[
                current_index : (current_index + size), target_col
                ] = labels[start:end]
                current_index += size

            # all done
            return {
                "queries": queries_batch,
                "label_coords": label_coords_batch,
                "query_type_indexes": query_type_indexes_batch,
                "triples": triples_batch,
            }

        return collate

    def _prepare_batch(
        self, batch_index, batch, result: TrainingJob._ProcessBatchResult
    ):
        # move labels to GPU for entire batch (else somewhat costly, but this should be
        # reasonably small)
        result.prepare_time -= time.time()
        batch["label_coords"] = batch["label_coords"].to(self.device)
        result.size = len(batch["queries"])
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
        queries_subbatch = batch["queries"][subbatch_slice].to(self.device)
        subbatch_size = len(queries_subbatch)
        label_coords_batch = batch["label_coords"]
        query_type_indexes_subbatch = batch["query_type_indexes"][subbatch_slice]

        # in this method, example refers to the index of an example in the batch, i.e.,
        # it takes values in 0,1,...,batch_size-1
        # stores positions in batch of examples for each query type
        examples_per_query_type = {}
        for query_type_index, query_type in enumerate(self._query_types):
            examples_per_query_type[query_type] = (
                (query_type_indexes_subbatch == query_type_index)
                .nonzero(as_tuple=False)
                .to(self.device)
                .view(-1)
            )

        labels_subbatch = kge.job.util.coord_to_sparse_tensor(
            subbatch_size,
            max(self.dataset.num_entities(), self.dataset.num_relations()),
            label_coords_batch,
            self.device,
            row_slice=subbatch_slice,
        ).to_dense()
        labels_for_query_type = {}
        for query_type, examples in examples_per_query_type.items():
            # if target slot is relations
            if query_type.target_slot == 1:
                labels_for_query_type[query_type] = labels_subbatch[
                    examples, : self.dataset.num_relations()
                ]
            else:
                labels_for_query_type[query_type] = labels_subbatch[
                    examples, : self.dataset.num_entities()
                ]

        if self.label_smoothing > 0.0:
            # as in ConvE: https://github.com/TimDettmers/ConvE
            for query_type, labels in labels_for_query_type.items():
                labels_for_query_type[query_type] = (
                    1.0 - self.label_smoothing
                ) * labels + 1.0 / labels.size(1)

        result.prepare_time += time.time()

        # forward/backward pass
        for query_type, examples in examples_per_query_type.items():
            if len(examples) > 0:
                result.prepare_time -= time.time()
                # get score function for current query type
                target_slot = query_type.name.find("_")
                score_func = getattr(self.model, query_type.score_fn)
                # get weight for current query type or use 1.0 as default
                query_weight = query_type.weight

                # get input and target indices from batch
                input_indices = [
                    queries_subbatch[examples, 0],
                    queries_subbatch[examples, 1],
                ]
                # set target slots to avoid using placeholder embeddings as
                # targets for score computations
                # TODO should this be here or elsewhere, e.g. at model level?
                if query_type.target_slot == 1:
                    target_indices = torch.arange(self.dataset.num_relations()).to(self.device)
                else:
                    target_indices = torch.arange(self.dataset.num_entities()).to(self.device)

                # use placeholder embeddings if query type asks for it
                placeholder_slot = query_type.name.find("^")
                if placeholder_slot != -1:
                    # get position of slot to replace
                    slot_position = query_type.input_slots.index(
                        placeholder_slot
                    )
                    # replace slot with placeholder embedding
                    input_indices[slot_position] = torch.tensor(
                        self.model.pseudo_indices[placeholder_slot]
                    ).expand(input_indices[slot_position].size(0)).to(self.device)
                result.prepare_time += time.time()

                # forward/backward pass
                result.forward_time -= time.time()
                scores = score_func(
                    input_indices[0], input_indices[1], target_indices
                )
                loss_value_query = self.loss(scores, labels_for_query_type[query_type])
                # note: average on batch_size, not on subbatch_size
                loss_value_query = query_weight * (loss_value_query / batch_size)
                result.avg_loss += loss_value_query.item()
                result.forward_time += time.time()
                result.backward_time -= time.time()
                if not self.is_forward_only:
                    loss_value_query.backward()
                result.backward_time += time.time()
