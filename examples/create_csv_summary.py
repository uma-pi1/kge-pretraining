import argparse
from email.utils import encode_rfc2231
from multiprocessing.sharedctypes import Value
from tabnanny import check
import pandas
import os
import yaml
import numpy as np

from collections import defaultdict as ddict


# query types
QUERY_TYPES = [
    "gmean",    # geometric mean
    "sp_",      # object prediction
    "_po",      # subject prediction
    "s_o",      # relation prediction
    "s^_",      # subject neighborhood
    "_^o",      # object neighborhood
    "^p_",      # relation subjects
    "_p^",      # relation objects
    "s_^",      # outward links
    "^_o",      # inward links
]

# TODO the following two functions should be a SINGLE function! 
def parse_rank_eval_txt(file_path, metrics, test_data="False"):
    """ 
    Extracts relevant metrics from rank eval results in dumped TXT 
    """

    # iterate through lines in txt file
    checkpoint_metrics = ddict(float)
    with open(file_path) as file:
        for line in reversed(file.readlines()):
            line = line.strip()
            # store metrics in dict
            for metric in metrics:
                if metric in line and "metric" not in line:
                    # check that it's not filtered_with_test on test data
                    if test_data.lower() == "true" and "with_test" in line:
                        continue

                    line_split = line.split(":") 
                    metric_name = line_split[0].split()[-1]
                    value = float(line_split[-1])
                    query_type = metric_name[len(metric + "_"):]
                    if not query_type:
                        query_type = "amean"
                    checkpoint_metrics[metric + "_" + query_type] = value

    return checkpoint_metrics


def parse_downstream_task_txt(
        file_path, 
        metrics, 
        models="", 
        test_data="False", 
        test_data_n_times=False
    ):
    """ 
    Extracts relevant metrics from downstream task results in dumped TXT.
    The results of the given metrics are returned as a dictionary of form
    {metric: value}
    """

    # iterate through lines in txt file
    checkpoint_metrics = ddict(float)

    # dict to store performance per metric to then show best across DT models
    # should be of the form {metric: [value, value, value...]}
    best_metrics = ddict(list)
    best_stds = ddict(list)

    suffixes = ["mean", "std"]
    if test_data.lower() == "true" and test_data_n_times.lower() == "false":
        suffixes = [""]
    with open(file_path) as file:
        for line in reversed(file.readlines()):
            line = line.strip()
            # store metrics in dict
            for metric in metrics:
                for dt_model in models:
                    for suffix in suffixes:
                        for prefix in ["", "z-scores"]:
                            # construct metric you are after
                            metric_str = "_".join(
                                [prefix, metric, dt_model, suffix]
                                )
                            metric_str = metric_str.strip("_")
                            # if metric_str[0] == "_":
                            #     metric_str = metric_str[1:]

                            # see if current line may contain it
                            if "metric" not in line and metric_str + ":" in line:
                                # get metric in line
                                line_metric = line.strip().split(":")[0].split()[-1]
                                # compare source and target metrics, s
                                if metric_str == line_metric:
                                    line_split = line.split(":") 
                                    value = line_split[-1]
                                    if ".nan" not in value:
                                        value = float(line_split[-1])
                                    else:
                                        value = 0.0
                                    checkpoint_metrics[metric_str] = value

                                    # keep value to get best across dt models
                                    if suffix == "mean":
                                        # here dt_model is the current value you are adding to the list
                                        best_key = "_".join([prefix, metric, "max", suffix])
                                        best_key = best_key.strip("_")
                                        best_metrics[best_key].append(value)
                                    elif suffix == "std":
                                        # store with same name as mean to later find corresponding std
                                        best_key = "_".join([prefix, metric, "max", "mean"])
                                        best_key = best_key.strip("_")
                                        best_stds[best_key].append(value)

    # get best performance across dt models
    for k, v in best_metrics.items():
        # HACK to avoid max when I need min
        maximize = True
        for i in minimizing_metrics:
            if i in k:
                maximize = False
                break
        if maximize:
            # checkpoint_metrics[k] = np.array(v).max().item()
            best_position = np.array(v).argmax()
        else:
            # checkpoint_metrics[k] = np.array(v).min().item()
            best_position = np.array(v).argmin()

        # get max value and corresponding std
        max_value = v[best_position]
        max_std = best_stds[k][best_position]
        max_std_key = k.replace("mean", "std")
        checkpoint_metrics[k] = max_value
        checkpoint_metrics[max_std_key] = max_std

        # print()
        # print("DEBUGGING MAX STD STUFF")
        # print(file_path)
        # print("POS: ", best_position)
        # print("VALS:", v)
        # print("STDS:", best_stds[k])
        # print("OG MAX:", checkpoint_metrics[k])
        # print("MY MAX:", max_value)
        # print("MY STD:", max_std)
        # print("OG MAX:", type(checkpoint_metrics[k]))
        # print("MY MAX:", type(max_value))
        # print("VAL KEY:", k)
        # print("STD KEY:", max_std_key)
        # print()
        # input()

    return checkpoint_metrics


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--folder', 
        required=True, 
        help="folder with trials"
    )
    parser.add_argument(
        '--trial', 
        required=True, 
        help="trial folder"
    )
    parser.add_argument(
        '--eval', 
        required=True, 
        help="rank_eval, entity_classification, or regression"
    )
    parser.add_argument(
        '--task_name', 
        required=False, 
        help="name of task folder, e.g. entity_classification_#, date_of_birth"
    )
    parser.add_argument(
        '--models', 
        required=False, 
        help="comma separated names of downstream models used"
    )
    parser.add_argument(
        '--test_data', 
        required=True, 
        help="True or False"
    )
    parser.add_argument(
        '--test_data_n_times', 
        required=True, 
        help="True or False"
    )
    args, _ = parser.parse_known_args()

    def convert_str_to_bool(text):
        if text.lower() == "true":
            bool_value = True
        elif text.lower() == "false":
            bool_value = False
        else:
            raise ValueError("Unrecognizes value for boolean: ", text)

        return bool_value

    # get args
    folder = args.folder
    trial = args.trial
    test_data = args.test_data
    test_data_n_times = args.test_data_n_times
    if trial[-1] == "/":
        trial = trial[:-1]
    eval = args.eval
    if eval in ["entity_classification", "regression"]:
        task_name = args.task_name
        dt_models = args.models.split(",")
    else:
        task_name = eval

    # metrics
    if test_data.lower() == "true":
        er_metrics = [
                "mean_reciprocal_rank_filtered", 
                "hits_at_10_filtered"
            ]
    else:
        er_metrics = [
                "mean_reciprocal_rank_filtered_with_test", 
                "hits_at_10_filtered_with_test"
            ]
    if "rank_eval" in eval:
        if test_data.lower() == "True":
            metrics = [
                "mean_reciprocal_rank_filtered", 
                "hits_at_10_filtered"
            ]
        else:
            metrics = [
                "mean_reciprocal_rank_filtered_with_test", 
                "hits_at_10_filtered_with_test"
            ]
    elif eval == "entity_classification":
        metrics = [
            "accuracy",
            "weighted_f1",
            "weighted_precision",
            "weighted_recall",
        ]
    elif eval == "regression":
        metrics = [
            "mse",
            "rse",
            "ndcg_at_100",
            "spearman",
        ]
    else:
        raise ValueError("Unrecognized EVAL {}".format(eval))

    minimizing_metrics = [
        "rse",
        "mse"
    ]

    # hyperparameters
    hyperparameters = [
        "model", 
        "1vsAll_hybrid.query_weights.sp_", 
        "1vsAll_hybrid.query_weights._po", 
        "1vsAll_hybrid.query_weights.s_o", 
        "1vsAll_hybrid.query_weights.s^_", 
        "1vsAll_hybrid.query_weights._^o", 
        "1vsAll_hybrid.query_weights.^p_", 
        "1vsAll_hybrid.query_weights._p^", 
        "1vsAll_hybrid.query_weights.s_^",
        "1vsAll_hybrid.query_weights.^_o",
        "train.type",
        "lookup_embedder.dim",
        "train.batch_size",
        "train.loss",
        "train.optimizer",
        # "train.optimizer_args.lr",
        "train.optimizer.default.args.lr",
        "lookup_embedder.initialize",
        "lookup_embedder.regularize",
        "lookup_embedder.regularize_args.weighted",
        "###model###.entity_embedder.regularize_weight",
        "###model###.relation_embedder.regularize_weight",
        "###model###.entity_embedder.dropout",
        "###model###.relation_embedder.dropout",
    ]

    # gogogo
    print("Creating CSVs that summarizes all checkpoints per trial")
    print("FOLDER:  {}".format(folder.upper()))
    print("TRIAL:   {}".format(trial.upper()))
    print("EVAL:    {}".format(eval.upper()))
    print("TEST:    {}".format(test_data.upper()))
    if eval in ["entity_classification", "regression"]:
        print("TASK NAME:  {}".format(task_name.upper()))
        print("MODELS:  {}".format(dt_models))

        # don't print model name if single one
        # as done by LibKGE's output
        if len(dt_models) <= 1:
            dt_models = [""]

    # process trial!
    if test_data_n_times.lower() == "true":
        output_filename = "{}_summary_{}_{}_test_n_times.csv".format(
            task_name, folder, trial
        )
    elif test_data.lower() == "true":
        output_filename = "{}_summary_{}_{}_test.csv".format(
            task_name, folder, trial
        )
    else:
        output_filename = "{}_summary_{}_{}.csv".format(
            task_name, folder, trial
        )

    # skip if output file exists
    if os.path.isfile(os.path.join(output_filename)):
        print(
            "Skipping {}/{} as output file {} already exists.".format(
                folder, trial, output_filename
            )
        )
    else:
        # get trial hyperparameters
        trial_hyperparameters = {}
        with open("config.yaml") as config_file:
            # read config file for trial
            config_dict = yaml.load(config_file, Loader=yaml.SafeLoader)
            # read hyperparameters from config
            model = ""
            for i, hyperparam in enumerate(hyperparameters):
                split_hyp = hyperparam.split(".")
                hyp_dict = config_dict
                # get model
                if hyperparam == "model":
                    model = hyp_dict[hyperparam]
                    # get base model if model_with_placeholders
                    if model == "model_with_placeholders":
                        hyperparameters.insert(
                            i + 1, "model_with_placeholders.base_model.type"
                        )
                for hyp in split_hyp:
                    # use correct model
                    if hyp == "###model###":
                        hyp = model
                    hyp_dict = hyp_dict[hyp]
                    # update model if model_with_placeholders is used
                    if model == "model_with_placeholders" and \
                        split_hyp[0] == "model_with_placeholders" and \
                        hyp == "type":
                        model = hyp_dict
                trial_hyperparameters[hyperparam] = hyp_dict

        # get ER performance per checkpoint
        er_results = {}
        best_mrr = 0
        best_hits = 0
        with open("trace.yaml") as trace_file:
            # read trace file for trial
            for line in trace_file:
                # hack for Pytorch 1.10's object for torch version
                if "TorchVersion" in line:
                    continue
                entry = yaml.load(line, Loader=yaml.SafeLoader)
                if entry["event"] == "eval_completed":
                    er_results[str(entry["epoch"])] = (
                        entry[er_metrics[0]], entry[er_metrics[1]]
                    )
                    # keep track of MRR and HITS of best checkpoint
                    # selection was always done on MRR
                    if entry[er_metrics[0]] > best_mrr:
                        best_mrr = entry[er_metrics[0]]
                        best_hits = entry[er_metrics[1]]

        # add best ER results for best checkpoint
        er_results["best"] = (best_mrr, best_hits)

        # iterate through checkpoints
        entries_all_checkpoints = []
        for checkpoint in sorted(os.listdir(".")):
            # skip if not a relevant TXT dump
            if task_name not in checkpoint or ".txt" not in checkpoint:
                continue

            # hack to make rank eval work with ablaction runs
            if task_name == "rank_eval":
                if checkpoint.split("_")[2] == "no":
                    continue

            # skip if not an n_times_file
            if test_data_n_times.lower() == "true" and "_n_times" not in checkpoint:
                continue

            # skip if not the right split
            if test_data.lower() == "true" and "_test" not in checkpoint:
                continue

            # process current dumped TXT
            print("\t\tCHECKPOINT: {}".format(checkpoint))
            checkpoint_metrics = ddict(float)

            # add ER performance
            checkpoint_num = checkpoint.split("_")[-1][:-4]
            # skip checkpoint 00000 for backwards compatibility
            if checkpoint_num == "00000":
                continue
            if checkpoint_num != "best":
                checkpoint_num = checkpoint.split("_")[-1][:-4].lstrip("0")
            if checkpoint_num == "test":
                checkpoint_num = checkpoint.split("_")[-2]
                if checkpoint_num != "best":
                    checkpoint_num = checkpoint.split("_")[-2].lstrip("0")
            if checkpoint_num == "times":
                checkpoint_num = checkpoint.split("_")[-5]
                if checkpoint_num != "best":
                    checkpoint_num = checkpoint.split("_")[-5].lstrip("0")

            checkpoint_metrics["mrr_er"] = er_results[checkpoint_num][0]
            checkpoint_metrics["hits_er"] = er_results[checkpoint_num][1]

            # get metrics
            if "rank_eval" in eval:
                checkpoint_metrics.update(
                    parse_rank_eval_txt(checkpoint, metrics, test_data)
                    )
            else:
                checkpoint_metrics.update(
                    parse_downstream_task_txt(
                        checkpoint, metrics, dt_models, test_data, test_data_n_times
                        )
                    )

            # print(checkpoint_metrics)
            # input()

            # create entry for current checkpoint
            checkpoint_entry = [
                folder, 
                trial, 
                checkpoint.split("_")[-1][:-4], 
                model,
                task_name,
            ]

            # add performance metrics
            # er
            checkpoint_entry.append(checkpoint_metrics["mrr_er"])
            checkpoint_entry.append(checkpoint_metrics["hits_er"])
            # rank_eval
            if "rank_eval" in eval:
                for metric in metrics:
                    checkpoint_entry.append(
                        checkpoint_metrics[metric + "_amean"]
                        )
                    for query_type in QUERY_TYPES:
                        checkpoint_entry.append(
                            checkpoint_metrics[metric + "_" + query_type]
                        )
            # downstream tasks
            else:
                suffixes = ["mean", "std"]
                if test_data.lower() == "true" and test_data_n_times.lower() == "false":
                    suffixes = [""]
                for prefix in ["", "z-scores"]:
                    for metric in metrics:
                        for dt_model in dt_models + ["max"]:
                            for suffix in suffixes:
                                # if dt_model == "max" and suffix == "std":
                                #     continue
                                metric_str = "_".join(
                                    [prefix, metric, dt_model, suffix]
                                    )
                                metric_str = metric_str.strip("_")
                                # if metric_str[0] == "_":
                                #     metric_str = metric_str[1:]
                                checkpoint_entry.append(
                                    checkpoint_metrics[metric_str]
                                    )

            # add hyperparameter values
            for hyperparam in hyperparameters:
                checkpoint_entry.append(trial_hyperparameters[hyperparam])

            # add entry to all entries
            entries_all_checkpoints.append(checkpoint_entry)

        # create dataframe for this trial
        # TODO col name creation can be movred right up here, code looks same
        cols = [
            "folder", 
            "trial", 
            "checkpoint", 
            "model", 
            "task name", 
        ]

        # add er column names
        cols.append("mrr_er")
        cols.append("hits@10_er")
        # add metrics column names
        if "rank_eval" in eval:
            # mrr cols
            cols.append("mrr_amean")
            for query_type in QUERY_TYPES:
                cols.append("mrr_" + query_type)
            # hits cols
            cols.append("hits@10_amean")
            for query_type in QUERY_TYPES:
                cols.append("hits@10_" + query_type)
        else:
            suffixes = ["mean", "std"]
            if test_data.lower() == "true" and test_data_n_times.lower() == "false":
                suffixes = [""]
            for prefix in ["", "z-scores"]:
                for metric in metrics:
                    for dt_model in dt_models + ["max"]:
                        for suffix in suffixes:
                            # if dt_model == "max" and suffix == "std":
                            #     continue
                            metric_str = "_".join(
                                [prefix, metric, dt_model, suffix]
                                )
                            metric_str = metric_str.strip("_")
                            # if metric_str[0] == "_":
                            #     metric_str = metric_str[1:]
                            cols.append(metric_str)
                
        # add hyperparameters
        for hyperparam in hyperparameters:
            cols.append(hyperparam.replace("###model###", model))
        trial_df = pandas.DataFrame(entries_all_checkpoints, columns=cols)

        # dump dataframe
        trial_df.to_csv(
            os.path.join(output_filename), index=False
        )
        print("Dumped summary for trial {} to {}/{}/{}.".format(
            trial, 
            folder,
            trial,
            output_filename)
        )
