import json
import pandas


def predictions_report_default(
        top_predictions_dict,
        triples,
        config,
        dataset,
        split,
        output_filename,
        use_strings=False,
        query_types = None,
):
    """
    Dumps the JSON file with top k predictions as required by KGxBoard

    :param top_predictions: dict of dicts containing top scores, entities,
    rankings and ties
    :param triples: tensor of size nx3 with the n triples used for evaluation
    :param config: job's config object
    :param dataset: job's dataset object
    :param output_filename: name of file to dump content
    :param use_strings: print triples and predictions with readable strings, if
    available
    """

    config.log("Dumping top predictions in default format")
    # set names to be dataset IDs or readable strings
    if use_strings:
        ent_names = dataset.entity_strings
        rel_names = dataset.relation_strings
    else:
        ent_names = dataset.entity_ids
        rel_names = dataset.relation_ids

    predictions = top_predictions_dict["predictions"]
    scores = top_predictions_dict["scores"]
    rankings = top_predictions_dict["rankings"]
    ties = top_predictions_dict["ties"]

    if query_types is None:
        query_types = ["sub", "obj"]

    # construct examples
    examples = []
    for triple_pos in range(len(triples)):
        # create entry for each example and task
        for query_type in query_types:
            # set correct mapping between LibKGE IDs and dataset IDs.
            if not isinstance(query_type, str):
                task = query_type.name
                pred_names = ent_names
                if query_type.target_slot == 1:
                    pred_names = rel_names
            else:
                task = query_type
                pred_names = ent_names

            # create entry
            example_dict = {}
            example_dict["gold_head"] = ent_names(triples[triple_pos, 0]).item()
            example_dict["gold_predicate"] = rel_names(
                triples[triple_pos, 1]
            ).item()
            example_dict["gold_tail"] = ent_names(triples[triple_pos, 2]).item()
            example_dict["task"] = task
            example_dict["predictions"] = \
                pred_names(predictions[task][triple_pos, :].int()).tolist()
            example_dict["scores"] = scores[task][triple_pos, :].tolist()
            example_dict["true_rank"] = rankings[task][triple_pos].item()
            example_dict["ties"] = ties[task][triple_pos].item()
            example_dict["example_id"] = triple_pos

            # add current example
            examples.append(example_dict)

            # top k (I know...)
            top_k = scores[task].size()[1]

    # TODO stuff that is a good idea to put in metadata
    #   1. Filter splits information
    #   2. Tie handling policy
    #   3. Framework, i.e. libkge
    # construct main dict
    main_dict = {
        "metadata": {
            "model": config.get("model"),
            "dataset": config.get("dataset.name"),
            "split": split,
        },
        "examples": examples
    }

    # dump main dict to json
    with open(output_filename, "w") as file:
        file.write(json.dumps(main_dict, indent=2))
    config.log(
        "Dumped top {} predictions to file: {}".format(top_k, output_filename)
    )

    # dump config settings
    flattened_config_dict = pandas.json_normalize(
        config.options, sep="."
    ).to_dict(orient="records")[0]

    # remove unwanted config settings
    train_type = config.get("train.type")
    model = config.get("model")
    if model == "reciprocal_relations_model":
        model = config.get("reciprocal_relations_model.base_model.type")
    wanted_keys = {
        "random_seed",
        "dataset.name",
        "model",
        "lookup_embedder",
        model,
        "train",
        train_type,
        "eval",
        "entity_ranking",
        "valid",
    }
    config_dict = {}
    for k, v in flattened_config_dict.items():
        root_key = k.split(".")[0]
        if root_key in wanted_keys:
            config_dict[k] = v

    # assumes given output filename has an extension
    output_filename_split = output_filename.split(".")
    output_filename = "".join(output_filename_split[:-1]) + \
                      "_hyperparameters." + output_filename_split[-1]
    with open(output_filename, "w") as file:
        file.write(json.dumps(config_dict, indent=2))
    config.log(
        "Dumped flattened config settings to file: {}".format(output_filename)
    )
