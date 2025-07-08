import math
import time
import sys

import torch
import kge.job
from kge.job import EvaluationJob, Job
from kge.job.util import QueryType
from kge import Config, Dataset
from collections import defaultdict
from kge.util.predictions_report import predictions_report_default
from kge.indexing import IndexWrapper, index_KvsAll


class RankingEvaluationJob(EvaluationJob):
    """ Ranking evaluation protocol """

    def __init__(self, config: Config, dataset: Dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)
        self.config.check(
            "ranking_evaluation.tie_handling.type",
            ["rounded_mean_rank", "best_rank", "worst_rank"],
        )

        # tie handling
        self.tie_handling = self.config.get("ranking_evaluation.tie_handling.type")
        self.tie_atol = float(self.config.get("ranking_evaluation.tie_handling.atol"))
        self.tie_rtol = float(self.config.get("ranking_evaluation.tie_handling.rtol"))

        # filtering
        self.filter_with_test = config.get("ranking_evaluation.filter_with_test")
        self.filter_splits = self.config.get("ranking_evaluation.filter_splits")
        if self.eval_split not in self.filter_splits:
            self.filter_splits.append(self.eval_split)

        # values of K
        max_k = min(
            self.dataset.num_entities(),
            max(self.config.get("ranking_evaluation.hits_at_k_s")),
        )
        self.hits_at_k_s = list(
            filter(lambda x: x <= max_k, self.config.get("ranking_evaluation.hits_at_k_s"))
        )

        #: Hooks after computing the ranks for each batch entry.
        #: Signature: hists, s, p, o, ranks, job, **kwargs
        self.hist_hooks = [hist_all]
        if config.get("ranking_evaluation.metrics_per.relation_type"):
            self.hist_hooks.append(hist_per_relation_type)
        if config.get("ranking_evaluation.metrics_per.argument_frequency"):
            self.hist_hooks.append(hist_per_frequency_percentile)

        # flag to check if model has pseudo embeddings
        self._pseudo_indices = False
        if hasattr(self.model, "pseudo_indices"):
            self._pseudo_indices = True

        # flag to compute geometric mean
        self._gmean = config.get("ranking_evaluation.geometric_mean")

        if self.__class__ == RankingEvaluationJob:
            for f in Job.job_created_hooks:
                f(self)

        # dump predictions
        # for now, we only support dumping top predictions after filtering
        self._predictions = config.get("ranking_evaluation.predictions.dump")
        self._predictions_top_k = config.get(
            "ranking_evaluation.predictions.top_k"
        )
        self._predictions_filename = config.get(
            "ranking_evaluation.predictions.filename"
        )
        self._predictions_use_strings = config.get(
            "ranking_evaluation.predictions.use_strings"
        )

    def _prepare(self):
        super()._prepare()
        """
        Determine enabled query types and construct corresponding indexes.
        """

        # eval triples
        self.triples = self.dataset.split(self.config.get("eval.split"))

        # determine enabled query types
        self._query_types = self.config.get("ranking_evaluation.query_types")
        self._query_types = [
            QueryType(query_type, value) for query_type, value in self._query_types.items() if value > 0.0
        ]

        # determine enabled multihop query types
        multihop_query_types = self.config.get("ranking_evaluation.multihop_query_types")
        if multihop_query_types:
            for query_type, value in multihop_query_types.items():
                if value > 0.0:
                    self._query_types.append(QueryType(query_type, value))

        if not len(self._query_types):
            raise ValueError("Ranking Evaluation has no enabled query types.")

        # precompute indexes for each query type
        for query_type in self._query_types:
            for split in self.filter_splits:
                index_name = f"{split}_{query_type.index}"
                self._create_index(index_name, query_type, split)
            if "test" not in self.filter_splits and self.filter_with_test:
                index_name = f"test_{query_type.index}"
                self._create_index(index_name, query_type, "test")

        # init data loader
        self.loader = torch.utils.data.DataLoader(
            self.triples,
            collate_fn=self._collate,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.config.get("eval.num_workers"),
            pin_memory=self.config.get("eval.pin_memory"),
        )

    def _create_index(self, index_name, query_type, split):
        """
        Adds index to indexes in object's dataset
        """
        if index_name not in self.dataset.index_functions:
            key_cols, val_cols = query_type.get_key_value_cols()
            num_hops = 1
            # check if query type is for more than 1 hop
            if query_type.name[-1].isnumeric():
                num_hops = int(query_type.name[-1])
            self.dataset.index_functions[index_name] = IndexWrapper(
                index_KvsAll,
                split=split,
                key=key_cols,
                value=val_cols,
                num_hops=num_hops,
            )
        self.dataset.index(index_name)

    def _collate(self, batch):
        """
        Returns given batch as well as a dictionary where keys are (enabled)
        query types and values are coordinates of true triples in all filter
        splits for each query type. If needed, a separate dictionary for the
        test split is also created to compute filtered_with_test metrics.
        """

        label_coords_per_query_type = {}
        test_label_coords_per_query_type = {}
        batch = torch.cat(batch).reshape((-1, 3))
        for query_type in self._query_types:
            label_coords = []
            for split in self.filter_splits:
                query_type_index = self.dataset.index(f"{split}_{query_type.index}")
                split_label_coords = kge.job.util.get_label_coords_from_spo_batch(
                    batch,
                    query_type,
                    query_type_index
                )
                label_coords.append(split_label_coords)
            label_coords = torch.cat(label_coords)
            label_coords_per_query_type[query_type.name] = label_coords

            if "test" not in self.filter_splits and self.filter_with_test:
                query_type_index = self.dataset.index(f"test_{query_type.index}")
                test_label_coords = kge.job.util.get_label_coords_from_spo_batch(
                    batch,
                    query_type,
                    query_type_index
                )
            else:
                test_label_coords = torch.zeros([0, 2], dtype=torch.long)
            test_label_coords_per_query_type[query_type.name] = test_label_coords

        return batch, label_coords_per_query_type, test_label_coords_per_query_type

    @torch.no_grad()
    def _evaluate(self):
        num_entities = self.dataset.num_entities()
        num_relations = self.dataset.num_relations()

        # we also filter with test data if requested
        filter_with_test = "test" not in self.filter_splits and self.filter_with_test

        # which rankings to compute (DO NOT REORDER; code assumes the order
        # given here)
        rankings = (
            ["_raw", "_filt", "_filt_test"] if filter_with_test else ["_raw", "_filt"]
        )

        # Initialize dictionaries that hold the overall histogram of ranks of
        # true answers. These histograms are used to compute relevant metrics.
        # The dictionary entry with key 'all' collects the overall statistics
        # and is the default.
        hists = dict()
        hists_filt = dict()
        hists_filt_test = dict()

        # create initial trace entry
        self.current_trace["epoch"] = dict(
            type="ranking_evaluation",
            scope="epoch",
            split=self.eval_split,
            filter_splits=self.filter_splits,
            epoch=self.epoch,
            batches=len(self.loader),
            size=len(self.triples),
        )

        # run pre-epoch hooks (may modify trace)
        for f in self.pre_epoch_hooks:
            f(self)

        # prepare main datastructure for dumping top k predictions
        # IMPORTANT: by not keeping track of corresponding triples, we assume
        #   the data loader is set to shuffle=False, which is the case in this
        #   evaluation job.
        # This is a dictionary of dictionaries as follows: {
        #   predictions: {query_type: [list of entities]},
        #   scores: {query_type: [list of scores]},
        #   rankings: {query_type: [list of rankings]},
        #   ties: {query_type: [list of ties]},
        # }
        if self._predictions:
            top_predictions = defaultdict(lambda: defaultdict(list))
            target_filtering = "_filt"
            if filter_with_test:
                target_filtering = "_filt_test"

        # let's go
        epoch_time = -time.time()
        cand_chunk_size_per_query_type = {}
        batch_chunk_size_per_query_type = {}
        for batch_number, batch_coords_per_query_type in enumerate(self.loader):
            # create initial batch trace (yet incomplete)
            self.current_trace["batch"] = dict(
                type="ranking_evaluation",
                scope="batch",
                split=self.eval_split,
                filter_splits=self.filter_splits,
                epoch=self.epoch,
                batch=batch_number,
                size=len(batch_coords_per_query_type[0]),
                batches=len(self.loader),
            )

            # run the pre-batch hooks (may update the trace)
            for f in self.pre_batch_hooks:
                f(self)

            # load batch components
            batch = batch_coords_per_query_type[0].to(self.device)
            s, p, o = batch[:, 0], batch[:, 1], batch[:, 2]
            label_coords_per_query_type = batch_coords_per_query_type[1]

            # initialize histograms for current batch
            batch_hists = dict()
            batch_hists_filt = dict()
            batch_hists_filt_test = dict()

            # default dictionary for storing ranks and num_ties for each query
            # type in rankings (eg. raw, filt) as list of len 2: [ranks, num_ties]
            ranks_and_ties_for_ranking = defaultdict(
                lambda: [
                    torch.zeros(s.size(0), dtype=torch.long, device=self.device),
                    torch.zeros(s.size(0), dtype=torch.long, device=self.device),
                ]
            )

            # prepare batch datastructure for dumping top k predictions
            # this is a dictionary of dictionaries similar to top_predictions
            if self._predictions:
                batch_predictions = defaultdict(dict)
                # init required tensors
                for query_type in self._query_types:
                    task = query_type.name
                    # entities tensors
                    batch_predictions["predictions"][task] = torch.ones(
                        len(batch), self._predictions_top_k
                    ).to(self.device) * float("-Inf")
                    # scores tensors
                    batch_predictions["scores"][task] = torch.ones(
                        len(batch), self._predictions_top_k
                    ).to(self.device) * float("-Inf")

            # compute rankings for each query type
            for query_type in self._query_types:
                task = query_type.name

                # get embedding indices of input slots used to compute scores
                input_indices = [
                    batch[:, query_type.input_slots[0]],
                    batch[:, query_type.input_slots[1]],
                ]

                # use placeholder embeddings if query type asks for it
                placeholder_slot = query_type.name.find("^")
                if placeholder_slot != -1:
                    # get position of slot to replace
                    replaced_input_slot = query_type.input_slots.index(placeholder_slot)
                    # replace slot with placeholder embedding
                    if self._pseudo_indices:
                        input_indices[replaced_input_slot] = torch.tensor(
                            self.model.pseudo_indices[placeholder_slot]
                        ).expand(input_indices[replaced_input_slot].size(0)).to(self.device)

                # set num candidates
                num_candidates = num_entities
                if query_type.target_slot == 1:
                    num_candidates = num_relations

                # initialize candidate chunk size for this query type if not done already
                if query_type not in cand_chunk_size_per_query_type:
                    cand_chunk_size_per_query_type[query_type] = num_candidates

                # initialize batch chunk size for this query type if not done already
                if query_type not in batch_chunk_size_per_query_type:
                    batch_chunk_size_per_query_type[query_type] = self.batch_size

                # get query type data
                input_slots = query_type.input_slots
                target_slot = query_type.target_slot
                score_fn = getattr(self.model, query_type.score_fn)

                # dictionary that holds sparse tensors of labels for each type
                # of ranking, e.g. filtered, filtered_with_test
                labels_for_ranking = defaultdict(lambda: None)

                # get coords of labels for current query type
                label_coords = label_coords_per_query_type[query_type.name].to(self.device)
                # construct a sparse label tensor of shape batch_size x num_entities
                # entries are either 0 (false) or infinity (true)
                labels = kge.job.util.coord_to_sparse_tensor(
                    len(batch), num_candidates, label_coords, self.device, float("Inf")
                )
                labels_for_ranking["_filt"] = labels

                # do the same for test data if needed
                if filter_with_test:
                    test_label_coords_per_query_type = batch_coords_per_query_type[2]
                    # get coords of test labels for current query type
                    test_label_coords = test_label_coords_per_query_type[query_type.name].to(self.device)
                    # create sparse labels tensor
                    test_labels = kge.job.util.coord_to_sparse_tensor(
                        len(batch),
                        num_candidates,
                        test_label_coords,
                        self.device,
                        float("Inf"),
                    )
                    labels_for_ranking["_filt_test"] = test_labels

                # compute true scores beforehand, since we can't get them from a
                # chunked score table
                # o_true_scores = self.model.score_spo(s, p, o, "o").view(-1)
                # s_true_scores = self.model.score_spo(s, p, o, "s").view(-1)
                # scoring with spo vs sp and po can lead to slight differences
                # for ties due to floating point issues.
                # We use score_sp and score_po to stay consistent with scoring
                # used for further evaluation.
                unique_target_slot, unique_target_slot_inverse = torch.unique(
                    batch[:, target_slot], return_inverse=True
                )
                # aggregate scores over all possible candidates if query
                # if query type requires ANY embeddings and model does not have them
                if placeholder_slot != -1 and not self._pseudo_indices:
                    score_fn_args = {
                        "input_indices": input_indices,
                        "target_slots": unique_target_slot
                    }
                    true_scores, cand_chunk_size, batch_chunk_size = self._aggregate_scores_over_candidates(
                        score_fn,
                        score_fn_args,
                        placeholder_slot,
                        replaced_input_slot,
                        init_cand_chunk_size=cand_chunk_size_per_query_type[query_type]
                    )
                    # update chunks size for this query type
                    cand_chunk_size_per_query_type[query_type] = cand_chunk_size
                    batch_chunk_size_per_query_type[query_type] = batch_chunk_size
                else:
                    # query does not require ANY embeddings or model has them
                    # if needed
                    true_scores = score_fn(
                        input_indices[0],
                        input_indices[1],
                        unique_target_slot
                    )

                target_slot_true_scores = torch.gather(
                    true_scores,
                    1,
                    unique_target_slot_inverse.view(-1, 1),
                ).view(-1)

                # calculate scores in chunks to not have the complete score
                # matrix in memory. A chunk here represents a range of
                # entity_values to score against
                if self.config.get("ranking_evaluation.chunk_size") > -1:
                    chunk_size = self.config.get("ranking_evaluation.chunk_size")
                else:
                    chunk_size = self.dataset.num_entities()

                # process chunk by chunk
                for chunk_number in range(math.ceil(num_candidates / chunk_size)):
                    chunk_start = chunk_size * chunk_number
                    chunk_end = min(chunk_size * (chunk_number + 1), num_candidates)

                    # compute scores of chunk
                    target_slots = torch.arange(
                        chunk_start, chunk_end, device=self.device
                    )
                    # aggregate scores over all possible candidates if query
                    # requires ANY embeddings and model does not have them
                    if placeholder_slot != -1 and not self._pseudo_indices:
                        score_fn_args = {
                            "input_indices": input_indices,
                            "target_slots": target_slots
                        }
                        scores, cand_chunk_size, batch_chunk_size = self._aggregate_scores_over_candidates(
                            score_fn,
                            score_fn_args,
                            placeholder_slot,
                            replaced_input_slot,
                            init_cand_chunk_size=cand_chunk_size_per_query_type[query_type]
                        )
                        # update chunk sizes for this query type
                        cand_chunk_size_per_query_type[query_type] = cand_chunk_size
                        batch_chunk_size_per_query_type[query_type] = batch_chunk_size
                    else:
                        # query does not require ANY embeddings or model has
                        # them if needed
                        scores = score_fn(
                            input_indices[0],
                            input_indices[1],
                            target_slots
                        )

                    # replace the precomputed true_scores with the ones
                    # occurring in the scores matrix to avoid floating point
                    # issues
                    target_slot_in_chunk_mask = (chunk_start <= batch[:, target_slot]) & \
                                                (batch[:, target_slot] < chunk_end)
                    target_slot_in_chunk = (batch[:, target_slot][target_slot_in_chunk_mask] - chunk_start).long()

                    # check that scoring is consistent up to configured
                    # tolerance. if this is not the case, evaluation metrics may
                    # be artificially inflated
                    close_check = torch.allclose(
                        scores[target_slot_in_chunk_mask, target_slot_in_chunk],
                        target_slot_true_scores[target_slot_in_chunk_mask],
                        rtol=self.tie_rtol,
                        atol=self.tie_atol,
                    )

                    if not close_check:
                        diff = torch.abs(
                            scores[target_slot_in_chunk_mask, target_slot_in_chunk]
                            - target_slot_true_scores[target_slot_in_chunk_mask]
                        )

                        self.config.log(
                            f"Tie-handling: mean difference between scores was: {diff.mean()}."
                        )
                        self.config.log(
                            f"Tie-handling: max difference between scores was: {diff.max()}."
                        )
                        error_message = "Error in tie-handling. The scores assigned to a triple by the SPO and SP_/_PO scoring implementations were not 'equal' given the configured tolerances. Verify the model's scoring implementations or consider increasing tie-handling tolerances."
                        if self.config.get("ranking_evaluation.tie_handling.warn_only"):
                            print(error_message, file=sys.stderr)
                        else:
                            raise ValueError(error_message)

                    # now compute the rankings (assumes order: None, _filt, _filt_test)
                    for ranking in rankings:
                        if labels_for_ranking[ranking] is None:
                            labels_chunk = None
                        else:
                            # densify the needed part of the sparse labels tensor
                            labels_chunk = self._densify_chunk_of_labels(
                                labels_for_ranking[ranking], chunk_start, chunk_end
                            )

                            # remove current example from labels
                            labels_chunk[target_slot_in_chunk_mask, target_slot_in_chunk] = 0

                        # compute partial ranking and filter the scores (sets
                        # scores of true labels to infinity)
                        (
                            rank_chunk,
                            num_ties_chunk,
                            scores_filt,
                        ) = self._filter_and_rank(
                            scores, labels_chunk, target_slot_true_scores
                        )

                        # from now on, use filtered scores
                        scores = scores_filt

                        # update rankings
                        ranks_and_ties_for_ranking[query_type.name + ranking][0] += rank_chunk
                        ranks_and_ties_for_ranking[query_type.name + ranking][1] += num_ties_chunk

                        # keep track of top k predictions
                        # currently only support for predictions after filtering
                        if self._predictions and ranking == target_filtering:
                            # get top k scores for current query type
                            top_scores_task = batch_predictions["scores"][task]
                            topk_predictions_batch = torch.topk(
                                torch.cat((scores, top_scores_task), dim=1),
                                self._predictions_top_k,
                                dim=1
                            )
                            # get candidates from top scores
                            chunk_candidates = torch.arange(chunk_start, chunk_end).to(self.device)
                            chunk_candidates = chunk_candidates.view(1, -1).expand(len(batch), -1)
                            top_predictions_task = batch_predictions["predictions"][task]
                            all_current_candidates = torch.cat(
                                (chunk_candidates, top_predictions_task),
                                dim=1
                            )
                            batch_predictions["predictions"][task] = torch.gather(
                                input=all_current_candidates,
                                dim=1,
                                index=topk_predictions_batch.indices
                            )
                            # save current top k for batch
                            batch_predictions["scores"][task] = topk_predictions_batch.values

                    # we are done with the chunk

                # We are done with all chunks; calculate final ranks from counts
                ranks = self._get_ranks(
                    ranks_and_ties_for_ranking[query_type.name + "_raw"][0],
                    ranks_and_ties_for_ranking[query_type.name + "_raw"][1],
                )
                ranks_filt = self._get_ranks(
                    ranks_and_ties_for_ranking[query_type.name + "_filt"][0],
                    ranks_and_ties_for_ranking[query_type.name + "_filt"][1],
                )

                # Update the histograms of raw ranks and filtered ranks
                for f in self.hist_hooks:
                    f(batch_hists, s, p, o, ranks, query_type, job=self)
                    f(batch_hists_filt, s, p, o, ranks_filt, query_type, job=self)

                # and the same for filtered_with_test ranks
                if filter_with_test:
                    ranks_filt_test = self._get_ranks(
                        ranks_and_ties_for_ranking[query_type.name + "_filt_test"][0],
                        ranks_and_ties_for_ranking[query_type.name + "_filt_test"][1],
                    )
                    for f in self.hist_hooks:
                        f(
                            batch_hists_filt_test,
                            s,
                            p,
                            o,
                            ranks_filt_test,
                            query_type,
                            job=self,
                        )

                # store batch rankings and ties of this query type
                if self._predictions:
                    if filter_with_test:
                        batch_predictions["rankings"][task] = ranks_filt_test
                        batch_predictions["ties"][task] = ranks_and_ties_for_ranking[query_type.name + "_filt_test"][1]
                    else:
                        batch_predictions["rankings"][task] = ranks_filt
                        batch_predictions["ties"][task] = ranks_and_ties_for_ranking[query_type.name + "_filt"][1]

                # optionally: trace ranks of each example
                if self.trace_examples:
                    entry = {
                        "type": "ranking_evaluation",
                        "scope": "example",
                        "split": self.eval_split,
                        "filter_splits": self.filter_splits,
                        "size": len(batch),
                        "batches": len(self.loader),
                        "epoch": self.epoch,
                    }
                    for i in range(len(batch)):
                        entry["batch"] = i
                        entry["s"], entry["p"], entry["o"] = (
                            s[i].item(),
                            p[i].item(),
                            o[i].item(),
                        )
                        if filter_with_test:
                            entry["rank_filtered_with_test"] = (
                                ranks_filt_test[i].item() + 1
                            )
                        self.trace(
                            event="example_rank",
                            task=query_type.name,
                            rank=ranks[i].item() + 1,
                            rank_filtered=ranks_filt[i].item() + 1,
                            **entry,
                        )

                # we're done with the current query type

            # Compute the batch metrics for the full histogram (key "all")
            metrics = self._compute_metrics(batch_hists["all"])
            metrics.update(
                self._compute_metrics(batch_hists_filt["all"],
                                      suffix="_filtered")
            )
            if filter_with_test:
                metrics.update(
                    self._compute_metrics(
                        batch_hists_filt_test["all"],
                        suffix="_filtered_with_test"
                    )
                )

            # # update batch trace with the results
            self.current_trace["batch"].update(metrics)

            # run the post-batch hooks (may modify the trace)
            for f in self.post_batch_hooks:
                f(self)

            # output, then clear trace
            if self.trace_batch:
                self.trace(**self.current_trace["batch"])
            self.current_trace["batch"] = None

            # output batch information to console
            self.config.print(
                (
                    "\r"  # go back
                    + "{}  batch:{: "
                    + str(1 + int(math.ceil(math.log10(len(self.loader)))))
                    + "d}/{}, mrr (filt.): {:4.3f} ({:4.3f}), "
                    + "hits@1: {:4.3f} ({:4.3f}), "
                    + "hits@{}: {:4.3f} ({:4.3f})"
                    + "\033[K"  # clear to right
                ).format(
                    self.config.log_prefix,
                    batch_number,
                    len(self.loader) - 1,
                    metrics["mean_reciprocal_rank"],
                    metrics["mean_reciprocal_rank_filtered"],
                    metrics["hits_at_1"],
                    metrics["hits_at_1_filtered"],
                    self.hits_at_k_s[-1],
                    metrics["hits_at_{}".format(self.hits_at_k_s[-1])],
                    metrics["hits_at_{}_filtered".format(self.hits_at_k_s[-1])],
                ),
                end="",
                flush=True,
            )

            # merge batch histograms into global histograms
            def merge_hist(target_hists, source_hists):
                for key, hist in source_hists.items():
                    if key in target_hists:
                        target_hists[key] = target_hists[key] + hist
                    else:
                        target_hists[key] = hist

            merge_hist(hists, batch_hists)
            merge_hist(hists_filt, batch_hists_filt)
            if filter_with_test:
                merge_hist(hists_filt_test, batch_hists_filt_test)

            # add top batch predictions of this query type to global predictions
            if self._predictions:
                for query_type in self._query_types:
                    task = query_type.name
                    top_predictions["scores"][task].append(batch_predictions["scores"][task])
                    top_predictions["predictions"][task].append(batch_predictions["predictions"][task])
                    # rankings + 1 for readability
                    top_predictions["rankings"][task].append(batch_predictions["rankings"][task] + 1)
                    # ties - 1 because code counts correct answer as tie with itself
                    top_predictions["ties"][task].append(batch_predictions["ties"][task] - 1)

        # we are done; compute final metrics
        self.config.print("\033[2K\r", end="", flush=True)  # clear line and go back
        for key, hist in hists.items():
            name = "_" + key if key != "all" else ""
            metrics.update(self._compute_metrics(hists[key], suffix=name))
            metrics.update(
                self._compute_metrics(hists_filt[key], suffix="_filtered" + name)
            )
            if filter_with_test:
                metrics.update(
                    self._compute_metrics(
                        hists_filt_test[key], suffix="_filtered_with_test" + name
                    )
                )

        # add geometric mean of metrics
        if self._gmean:
            metrics.update(self._compute_geometric_mean_of_metrics(metrics))
        epoch_time += time.time()

        # dump predictions
        if self._predictions:
            # concat all batch predictions, scores and rankings
            for query_type in self._query_types:
                task = query_type.name
                top_predictions["scores"][task] = torch.cat(
                    top_predictions["scores"][task],
                    dim=0
                )
                top_predictions["predictions"][task] = torch.cat(
                    top_predictions["predictions"][task],
                    dim=0
                )
                top_predictions["rankings"][task] = torch.cat(
                    top_predictions["rankings"][task],
                    dim=0
                )
                top_predictions["ties"][task] = torch.cat(
                    top_predictions["ties"][task],
                    dim=0
                )
            # dump 'em!
            self._dump_top_k_predictions(top_predictions)

        # update trace with results
        self.current_trace["epoch"].update(
            dict(epoch_time=epoch_time, event="eval_completed", **metrics,)
        )

    def _dump_top_k_predictions(self, top_predictions):
        """
        Dumps top k predictions made by the model.

        :param top_predictions: dict of dicts created during evaluation
        """

        # TODO here's where support for other formats besides KGxBoard
        #   should be added
        predictions_report_default(
            top_predictions,
            self.triples,
            self.config,
            self.dataset,
            self.eval_split,
            self._predictions_filename,
            self._predictions_use_strings,
            self._query_types,
        )

    def _compute_geometric_mean_of_metrics(self, metrics):
        """
        Computes the geometric mean of mean rank, mean_reciprocal_rank and
        hits@k of all query types found in the given metrics dictionary. This
        is done separately for each type of metric, i.e. raw, filtered and
        filtered_with_test.
        """

        # set metric names and variants
        # TODO this is hardcoded because we do this hardcoded when computing
        #   the metrics in _compute_metrics. We should generalize these to class
        #   properties.
        metric_names = ["mean_rank", "mean_reciprocal_rank"]
        for k in self.hits_at_k_s:
            metric_names.append("hits_at_" + str(k))
        metric_variants = ["", "_filtered"]
        if self.filter_with_test and self.eval_split != "test":
            metric_variants.append("_filtered_with_test")

        # compute gmean
        gmean_metrics = defaultdict(lambda: 1)
        n = len(self._query_types)
        for metric_name in metric_names:
            for metric_variant in metric_variants:
                base_metric = metric_name + metric_variant
                base_metric_gmean = metric_name + metric_variant + "_gmean"
                for query_type in self._query_types:
                    query_name = query_type.name
                    key = base_metric + "_" + query_name
                    gmean_metrics[base_metric_gmean] *= metrics[key]
                # finalize gmean
                gmean_metrics[base_metric_gmean] = gmean_metrics[base_metric_gmean] ** (1/n)

        return gmean_metrics

    # TODO reduce the overhead from tuning everytime
    #   See what to pass when calling this function so there is less
    #   need to tune.
    #   This will depend on the batch_size, num_candidates and len(target_slots)
    def _aggregate_scores_over_candidates(
            self, 
            score_fn, 
            score_fn_args, 
            placeholder_slot, 
            candidates_pos,
            init_cand_chunk_size=None
    ) -> torch.Tensor:
        """
        Returns the output of the given score function with given args
        when used by replacing the placeholder_slot with all possible
        candidates for each given input tuple. That is, it computes the
        scores of n * num_candidates triples instead of the usual n, where n
        is the number of input tuples for the score function (usually batch
        size or chunk size) and num_candidates is either num_entities or
        num_relations. The scores over all candidates are max aggregated, so
        that this function returns a tensor of the same size as what would be
        returned by a call to score_fn with the given args., i.e.
        input_indices[0].shape[0] x len(target_slots), where input_indices and
        target_slots are keys in score_fn_args.

        :param score_fn: score function
        :param score_fn_args: args for score function as dictionary of
        form {input_indices: torch.Tensor, target_slots: torch.Tensor}
        :param placeholder_slot: slot in triple that needs to be replaced with
        set of candidates
        :param candidates_pos: whether set of all candidates is used as
        first or second input in call to score_fn indicated by int in {0, 1}
        :param init_cand_chunk_size: initial value of candidates chunk size
        that is then tuned
        :return: tensor of max aggregated scores
        """

        input_indices = score_fn_args["input_indices"]
        target_slots = score_fn_args["target_slots"]
        num_candidates = self.dataset.num_entities()
        if placeholder_slot == 1:
            num_candidates = self.dataset.num_relations()
        # set position of inputs in input_indices tuple which we use, i.e. the
        # ones we don't replace with all candidates
        fixed_input_pos = (candidates_pos + 1) % 2

        # we iterate over batch
        if init_cand_chunk_size is not None:
            cand_chunk = init_cand_chunk_size
        else:
            cand_chunk = num_candidates
        done = False
        while not done:
            all_scores = []
            try:
                for i in range(len(input_indices[0])):
                    # process chunk by chunk
                    chunk_scores = []
                    for chunk_number in range(
                            math.ceil(num_candidates / cand_chunk)
                    ):
                        chunk_start = cand_chunk * chunk_number
                        chunk_end = min(
                            cand_chunk * (chunk_number + 1), num_candidates
                        )
                        cand_chunk_size = chunk_end - chunk_start

                        # set candidates
                        candidates = torch.arange(
                            start=chunk_start, end=chunk_end
                        ).to(self.device)

                        # prepare input indices for input tuple i
                        inputs = [input_indices[fixed_input_pos][i].expand(cand_chunk_size)]
                        inputs.insert(candidates_pos, candidates)

                        # compute and store scores
                        scores = score_fn(
                            inputs[0], inputs[1], target_slots
                        )

                        # aggregate this chunk
                        chunk_scores.append(
                            torch.unsqueeze(torch.max(scores, 0).values, 0)
                        )

                    # aggregate all chunks
                    chunk_scores = torch.cat(chunk_scores)
                    all_scores.append(
                        torch.unsqueeze(torch.max(chunk_scores, 0).values, 0)
                    )

                # done with computing and aggregating scores for this batch
                done = True
            except RuntimeError as e:
                # pass error if not OOM
                if "CUDA out of memory" not in str(e):
                    raise e
                # try rerunning with smaller chunk size
                self.config.log(
                    "Caught OOM exception when computing aggregated scores over all candidates; "
                    "trying to reduce the corresponding chunk size of candidates..."
                )
                if cand_chunk <= 0:
                    # cannot reduce further
                    self.config.log(
                        "Cannot reduce chunk size of candidates "
                        f"(current value: {cand_chunk})"
                    )
                    raise e

                cand_chunk = int(cand_chunk * 0.9)
                self.config.log(
                    "New size of chunk of candidates "
                    f"(current value: {cand_chunk})"
                )

        # TODO I return None for consistency with the other version of this function
        #   Fix this once you know which version works best
        return torch.cat(all_scores), cand_chunk, None

    # TODO keep testing to make sure it is indeed faster
    #   So far it isn't, likely because some memory isn't freed!
    #   See that tuning goes to batch_chunk_size 1 EVERY TIME!
    #   Consider tuning cand_chunk_size first!
    #   But think in general: why does this happen?
    #   Also, tuning should be done a minimal number of times,
    #   not everytime this function is called
    def _aggregate_scores_over_candidates_with_smart_memory_usage(
            self, 
            score_fn, 
            score_fn_args, 
            placeholder_slot, 
            candidates_pos,
            init_cand_chunk_size=None,
            init_batch_chunk_size=None
    ) -> torch.Tensor:
        """
        Returns the output of the given score function with given args
        when used by replacing the placeholder_slot with all possible
        candidates for each given input tuple. That is, it computes the
        scores of n * num_candidates triples instead of the usual n, where n
        is the number of input tuples for the score function (usually batch
        size or chunk size) and num_candidates is either num_entities or
        num_relations. The scores over all candidates are max aggregated, so
        that this function returns a tensor of the same size as what would be
        returned by a call to score_fn with the given args., i.e.
        input_indices[0].shape[0] x len(target_slots), where input_indices and
        target_slots are keys in score_fn_args.

        :param score_fn: score function
        :param score_fn_args: args for score function as dictionary of
        form {input_indices: torch.Tensor, target_slots: torch.Tensor}
        :param placeholder_slot: slot in triple that needs to be replaced with
        set of candidates
        :param candidates_pos: whether set of all candidates is used as
        first or second in put in call to score_fn indicated by int in {0, 1}
        :param init_cand_chunk_size: initial value of candidates chunk size
        that is then tuned
        :param init_batch_chunk_size: initial value of batch chunk size that is tuned
        :return: tensor of max aggregated scores
        """

        input_indices = score_fn_args["input_indices"]
        target_slots = score_fn_args["target_slots"]
        num_candidates = self.dataset.num_entities()
        if placeholder_slot == 1:
            num_candidates = self.dataset.num_relations()
        # set position of inputs in input_indices tuple which we use, i.e. the
        # ones we don't replace with all candidates
        fixed_input_pos = (candidates_pos + 1) % 2

        # set initial sizes to be tuned
        batch_size = len(input_indices[0])
        if init_batch_chunk_size is not None:
            batch_chunk_size = init_batch_chunk_size
        else:
            batch_chunk_size = batch_size
        if init_cand_chunk_size is not None:
            cand_chunk_size = init_cand_chunk_size
        else:
            cand_chunk_size = num_candidates
        done = False

        # gogogo
        while not done:
            all_scores = []
            try:
                # process batch chunk by chunk
                for batch_chunk_number in range(
                        math.ceil(batch_size / batch_chunk_size)
                ):
                    batch_chunk_start = batch_chunk_size * batch_chunk_number
                    batch_chunk_end = min(
                        batch_chunk_size * (batch_chunk_number + 1), batch_size
                    )
                    current_batch_chunk_size = batch_chunk_end - batch_chunk_start

                    # process candidates chunk by chunk
                    cand_chunk_scores = []
                    for cand_chunk_number in range(
                            math.ceil(num_candidates / cand_chunk_size)
                    ):
                        cand_chunk_start = cand_chunk_size * cand_chunk_number
                        cand_chunk_end = min(
                            cand_chunk_size * (cand_chunk_number + 1), num_candidates
                        )
                        current_cand_chunk_size = cand_chunk_end - cand_chunk_start

                        # set candidates
                        candidates = torch.arange(
                            start=cand_chunk_start, end=cand_chunk_end
                        ).to(self.device)

                        # prepare inputs from batch chunk for score function
                        input_indices_chunk = input_indices[fixed_input_pos][batch_chunk_start:batch_chunk_end]
                        inputs = [input_indices_chunk.unsqueeze(1).expand(-1, current_cand_chunk_size)]
                        # prepare candidates for score function
                        candidates = candidates.unsqueeze(0).expand(current_batch_chunk_size, -1)
                        inputs.insert(candidates_pos, candidates)

                        # compute and store scores
                        scores = score_fn(
                            inputs[0].reshape(-1), inputs[1].reshape(-1), target_slots
                        ).view(current_batch_chunk_size, current_cand_chunk_size, len(target_slots))

                        # aggregate this chunk
                        cand_chunk_scores.append(torch.max(scores, 1).values)

                    # aggregate all chunks
                    cand_chunk_scores = torch.stack(cand_chunk_scores, dim=0)
                    all_scores.append(torch.max(cand_chunk_scores, 0).values)
                # done with computing and aggregating scores for this batch
                done = True
            except RuntimeError as e:
                # pass error if not OOM
                # TODO how to catch this nicely?! This is nasty!
                if "CUDA out of memory" not in str(e) and "CUBLAS" not in str(e):
                    raise e
                # we first tune the number of examples per batch chunk
                # then the number of candidates per example
                self.config.log(
                    "Caught OOM exception when computing aggregated scores over all candidates; "
                    "trying to reduce the number of examples and number of candidates per example..."
                )
                # try rerunning with smaller batch chunk size
                if batch_chunk_size > 1:
                    batch_chunk_size = int(batch_chunk_size * 0.5)
                    self.config.log(
                        "New size of batch chunk "
                        f"({batch_chunk_size})"
                    )
                # reduce chunk of candidates per example if batch chunk already at 1 example
                else:
                    if cand_chunk_size <= 0:
                        # cannot reduce chunk of candidates further
                        self.config.log(
                            "Cannot reduce chunk size of candidates "
                            f"({cand_chunk_size})"
                        )
                        raise e

                    cand_chunk_size = int(cand_chunk_size * 0.9)
                    self.config.log(
                        "New size of chunk of candidates "
                        f"({cand_chunk_size})"
                    )

        return torch.cat(all_scores), cand_chunk_size, batch_chunk_size

    def _densify_chunk_of_labels(
            self, labels: torch.Tensor, chunk_start: int, chunk_end: int
    ) -> torch.Tensor:
        """
        Creates a dense chunk of a sparse label tensor.

        A chunk here is a range of entity values with 'chunk_start' being the
        lower bound and 'chunk_end' the upper bound. The resulting tensor
        contains the labels for the sp chunk and the po chunk.

        :param labels: sparse tensor containing the labels corresponding to the
        batch

        :param chunk_start: int start index of the chunk

        :param chunk_end: int end index of the chunk

        :return: batch_size x chunk_size dense tensor with labels for the chunk
        """

        num_entities = self.dataset.num_entities()
        indices = labels._indices()
        mask = (chunk_start <= indices[1, :]) & (indices[1, :] < chunk_end)
        indices_chunk = indices[:, mask]
        indices_chunk[1, :] = indices_chunk[1, :] - chunk_start
        dense_labels = torch.sparse.LongTensor(
            indices_chunk,
            # since all sparse label tensors have the same value we could also
            # create a new tensor here without indexing with:
            # torch.full([indices_chunk.shape[1]], float("inf"))
            # TODO not sure if conversion in line below is correct, so check!
            labels._values()[mask],
            torch.Size([labels.size()[0], (chunk_end - chunk_start)]),
        ).to_dense()
        return dense_labels

    def _filter_and_rank(
            self,
            scores: torch.Tensor,
            labels: torch.Tensor,
            true_scores: torch.Tensor,
    ):
        """
        Filters the current examples with the given labels and returns counts
        rank and num_ties for each true score.

        :param scores: batch_size x chunk_size tensor of scores

        :param labels: batch_size x chunk_size tensor of scores

        :param true_scores: batch_size x 1 tensor containing the scores of the
        actual target slots in batch

        :return: batch_size x 1 tensors rank and num_ties and
        filtered scores
        """

        if labels is not None:
            # remove current example from labels
            scores = scores - labels
        rank, num_ties = self._get_ranks_and_num_ties(scores, true_scores)
        return rank, num_ties, scores

    def _get_ranks_and_num_ties(
        self, scores: torch.Tensor, true_scores: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """
        Returns rank and number of ties of each true score in scores.

        :param scores: batch_size x entities tensor of scores

        :param true_scores: batch_size x 1 tensor containing the actual scores
        of the batch

        :return: batch_size x 1 tensors rank and num_ties
        """

        # process NaN values
        scores = scores.clone()
        scores[torch.isnan(scores)] = float("-Inf")
        true_scores = true_scores.clone()
        true_scores[torch.isnan(true_scores)] = float("-Inf")

        # Determine how many scores are greater than / equal to each true answer
        # (in its corresponding row of scores)
        is_close = torch.isclose(
            scores, true_scores.view(-1, 1),
            rtol=self.tie_rtol,
            atol=self.tie_atol
        )
        is_greater = scores > true_scores.view(-1, 1)
        num_ties = torch.sum(is_close, dim=1, dtype=torch.long)
        rank = torch.sum(is_greater & ~is_close, dim=1, dtype=torch.long)
        return rank, num_ties

    def _get_ranks(self, rank: torch.Tensor, num_ties: torch.Tensor) -> torch.Tensor:
        """
        Calculates the final rank from (minimum) rank and number of ties.

        :param rank: batch_size x 1 tensor with number of scores greater than
        the one of the true score

        :param num_ties: batch_size x tensor with number of scores equal as the
        one of the true score

        :return: batch_size x 1 tensor of ranks
        """

        if self.tie_handling == "rounded_mean_rank":
            return rank + num_ties // 2
        elif self.tie_handling == "best_rank":
            return rank
        elif self.tie_handling == "worst_rank":
            return rank + num_ties - 1
        else:
            raise NotImplementedError

    def _compute_metrics(self, rank_hist, suffix=""):
        """
        Computes desired matrix from rank histogram
        """

        metrics = {}
        n = torch.sum(rank_hist).item()

        ranks = torch.arange(
            1, self.dataset.num_entities() + 1
        ).float().to(self.device)
        metrics["mean_rank" + suffix] = (
            (torch.sum(rank_hist * ranks).item() / n) if n > 0.0 else 0.0
        )

        reciprocal_ranks = 1.0 / ranks
        metrics["mean_reciprocal_rank" + suffix] = (
            (torch.sum(rank_hist * reciprocal_ranks).item() / n) if n > 0.0 else 0.0
        )

        hits_at_k = (
            (
                torch.cumsum(
                    rank_hist[: max(self.hits_at_k_s)], dim=0, dtype=torch.float64
                )
                / n
            ).tolist()
            if n > 0.0
            else [0.0] * max(self.hits_at_k_s)
        )

        for i, k in enumerate(self.hits_at_k_s):
            metrics["hits_at_{}{}".format(k, suffix)] = hits_at_k[k - 1]

        return metrics


# HISTOGRAM COMPUTATION ########################################################


def __initialize_hist(hists, key, job):
    """
    If there is no histogram with given `key` in `hists`, add an empty one.
    """

    if key not in hists:
        hists[key] = torch.zeros(
            [job.dataset.num_entities()],
            device=job.config.get("job.device"),
            dtype=torch.float,
        )


def hist_all(hists, s, p, o, ranks, query_type, job):
    """
    Create histogram of all subject/object ranks (key: "all").

    `hists` a dictionary of histograms to update; only key "all" will be
    affected. `s`, `p`, `o` are true triples indexes for the batch. `ranks` are
    the rank of the true answer for the given query type obtained from a model.
    """

    __initialize_hist(hists, "all", job)
    __initialize_hist(hists, query_type.name, job)
    hist_query = hists[query_type.name]

    hist = hists["all"]
    ranks_unique, ranks_count = torch.unique(ranks, return_counts=True)
    hist.index_add_(0, ranks_unique, ranks_count.float())
    hist_query.index_add_(0, ranks_unique, ranks_count.float())


def hist_per_relation_type(hists, s, p, o, ranks, query_type, job):
    for rel_type, rels in job.dataset.index("relations_per_type").items():
        __initialize_hist(hists, rel_type, job)
        hist = hists[rel_type]
        __initialize_hist(hists, f"{rel_type}_" + query_type.name, job)
        hist_query = hists[f"{rel_type}_" + query_type.name]

        mask = [_p in rels for _p in p.tolist()]
        for r, m in zip(ranks, mask):
            if m:
                hists[rel_type][r] += 1
                if job.head_and_tail:
                    hist_query[r] += 1


def hist_per_frequency_percentile(hists, s, p, o, ranks, query_type, job):
    # initialize
    frequency_percs = job.dataset.index("frequency_percentiles")
    for arg, percs in frequency_percs.items():
        for perc, value in percs.items():
            __initialize_hist(hists, "{}_{}".format(arg, perc), job)

    # get target slot in string format for given query type
    SLOTS = ["subject", "relation", "object"]
    target_slot_str = SLOTS[query_type.target_slot]

    # IMPORTANT:
    #   In standard entity ranking, the semantics of the relation percentile
    #   histograms is different than subject and object, because relatio is
    #   never predicted, so the relation histogram keeps track of both s and o
    #   predictions whenever the relation in the triple was in each percentile
    #   Now, we generalize to having any kind of query type so that we always
    #   keep track of percentile histograms of target slots only.

    # go
    for perc in frequency_percs["subject"].keys():  # same for relation, object
        for r, m_s, m_r in zip(
            ranks,
            [id in frequency_percs[target_slot_str][perc] for id in s.tolist()],
            # [id in frequency_percs["relation"][perc] for id in p.tolist()],
        ):
            if m_s:
                hists["{}_{}".format(target_slot_str, perc)][r] += 1
            # if m_r:
            #     hists["{}_{}".format("relation", perc)][r] += 1
