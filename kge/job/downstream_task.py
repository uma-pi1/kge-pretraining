import numpy
import time
import torch
import pandas as pd
import scipy.stats
import numpy as np

from kge.job import EvaluationJob, Job
from kge import Config, Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans
from sklearn import metrics as skmetrics
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from skopt.space import Real, Categorical, Space
from skopt.sampler import Sobol


from collections import defaultdict as ddict


SUPPORTED_TASKS = ["entity_classification", "regression", "clustering"]
CLASSIFICATION_MODELS = ["knn", "logistic_regression", "random_forest", "mlp"]
REGRESSION_MODELS = ["linear_regression", "random_forest", "mlp"]
CLUSTERING_MODELS = ["kmeans"]
MINIMIZED_METRICS = ["mse", "rse"]
SUPPORTED_METRICS = {
    "entity_classification": [
        "accuracy",
        "weighted_f1",
        "weighted_precision",
        "weighted_recall",
    ],
    "regression": [
        "mse",
        "rse",
        # "ndcg@100",
        "spearman"
    ],
    "clustering": [
        "weighted_f1",
    ]
}


class DownstreamTaskJob(EvaluationJob):
    """ Downstream task evaluation protocol """

    def __init__(self, config: Config, dataset: Dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)

        if self.__class__ == DownstreamTaskJob:
            for f in Job.job_created_hooks:
                f(self)

        # type of task
        self._type_of_task = config.get(
            "downstream_task.type"
        )
        if self._type_of_task == "entity_classification":
            self._supported_models = CLASSIFICATION_MODELS
        elif self._type_of_task == "regression":
            self._supported_models = REGRESSION_MODELS
        elif self._type_of_task == "clustering":
            self._supported_models = CLUSTERING_MODELS
        else:
            raise ValueError("Unsupported downstream task: {}".format(
                self._type_of_task
                )
            )

        # task dataset
        self._downstream_dataset = config.get(
            "downstream_task.dataset"
        )

        # models
        self._models = config.get("downstream_task.models").split(",")

        # number of cross validation folds
        self._num_folds = config.get(
            "downstream_task.num_folds"
        )

        # combine training and validation data
        self._combine_train_with_valid = config.get(
            "downstream_task.combine_train_with_valid"
        )

        # flag for evaluating on test
        self._eval_on_test = config.get(
            "downstream_task.eval_on_test"
        )

        # num times to train given model and evaluated on test
        self._test_data_num_times = config.get(
            "downstream_task.test_data_num_times"
        )

        # flag for tuning or not
        self._tune_model = config.get(
            "downstream_task.tune_model"
        )

        # selection metric for tuning
        self._selection_metric = config.get("downstream_task.selection_metric")

        # random search samples
        self._num_random_samples = config.get(
            "downstream_task.num_random_samples"
        )

        # flag for sobol sampling in tuning
        self._sobol = config.get("downstream_task.sobol")

        # hyperparameters
        self._hyperparameters = config.get("downstream_task.hyperparameters")

        # flag for using z-scores if task is regression
        self._z_scores = config.get("downstream_task.z_scores")

        # flag for using logs if task is regression
        self._log = config.get("downstream_task.log")

        # number of parallel jobs for scikit-learn tuning
        self._n_jobs = config.get("downstream_task.n_jobs")

        # concatenate embeddings of entity descriptions to input embeddings
        self._use_description_embeddings = config.get(
            "downstream_task.use_description_embeddings"
        )

    def _prepare(self):
        super()._prepare()

        # check that selection metric matches task
        if self._selection_metric not in SUPPORTED_METRICS[self._type_of_task]:
            raise ValueError("Unsupported selection metric {} for task {}.".format(
                self._selection_metric,
                self._type_of_task
            ))

        # load splits
        self._train_set = self.dataset.downstream_split(
            "train", folder=self._downstream_dataset
        )
        self._valid_set = self.dataset.downstream_split(
            "valid", folder=self._downstream_dataset
        )
        self._test_set = self.dataset.downstream_split(
            "test", folder=self._downstream_dataset
        )

        # init downstream models
        # TODO move this init_models to where models are used, not here!
        #   E.g. for tuning, the Sobol tuner inits models.
        #   Similarly, the randomizedsearchCV inits models.
        #   So, the only other place where this is needed is when not tuning
        #   models.
        #   You can also load hyperparameters from the config in that spot.
        #   You are already iterating through models there anyway.
        self._init_models = {}
        self._search_space = {}
        for model in self._models:
            if model in self._supported_models:
                # get hyperparameters if there is no tuning
                hyperparameters = None
                if not self._tune_model:
                    try:
                        hyperparameters = self._hyperparameters[model]
                    except KeyError:
                        # HACK to greatly simplify deployment of clustering tests
                        # set KMeans K to num_classes in downstream dataset
                        if self._type_of_task == "clustering" and self._models == ["kmeans"]:
                            num_classes = len(self._train_set[0]) - 1
                            hyperparameters = {"n_clusters": num_classes}
                        else:
                            # TODO get rid of this check, simply use an empty dict
                            #   That will mean default settings in scikit-learn
                            #   For now I like this to avoid mistakes in experiments
                            raise ValueError("Hyperparameters for model {} must be provided if no tuning is done.".format(
                                model
                                )
                            )
                self._init_models[model], self._search_space[model] = \
                    self._init_downstream_model(model, hyperparameters)
            else:
                raise ValueError(
                    "Unsupported model: {}. Supported models: {}".format(
                        model, self._supported_models
                    )
                )

    # TODO this code needs some work!
    #   Tuning is mixed with CV.
    #   Keep an option for no tuning, i.e. handle empty search spaces.
    #   Check the space and if None, run the model directly,
    #       no RandomizedSearchCV or Sobol Tuner.
    #   But do cross validation in any case for now!
    #   Keep the sobol tuner option as well.

    @torch.no_grad()
    def _evaluate(self) -> dict:

        # create initial trace entry
        self.current_trace["epoch"] = dict(
            type="downstream_task",
            scope="epoch",
            epoch=self.epoch,
            split=self.eval_split,
            task=self._type_of_task,
            task_dataset=self._downstream_dataset,
            models=self._models,
            selection_metric=self._selection_metric,
            num_folds=self._num_folds,
            eval_on_test=self._eval_on_test,
            num_random_sample=self._num_random_samples,
            sobol=self._sobol,
            z_scores=self._z_scores,
            apply_log=self._log,
        )

        # run pre-epoch hooks (may modify trace)
        for f in self.pre_epoch_hooks:
            f(self)

        # loop over downstream models
        all_metrics = {}
        best_hyperparams = {}
        for model_name in self._models:
            dt_model = self._init_models[model_name]
            epoch_time = -time.time()
            # set single CV split
            tuning_split = torch.cat(
                [self._train_set, self._valid_set]
            )
            # get features and targets
            train_features, train_targets = \
                self._get_features_and_targets(tuning_split)

            # define scorers
            scorers = self._get_scorers()

            # tune
            if self._tune_model:
                if self._sobol:
                    search_results = self._sobol_tuner(
                        model_name=model_name,
                        search_space=self._search_space[model_name],
                        features=train_features,
                        targets=train_targets,
                        scorers=scorers,
                        selection_metric=self._selection_metric,
                        n_jobs=self._n_jobs,

                    )
                else:
                    search_results = self._wrapper_randomizedsearch(
                        dt_model=dt_model,
                        search_space=self._search_space[model_name],
                        features=train_features,
                        targets=train_targets,
                        scorers=scorers,
                        selection_metric=self._selection_metric,
                        n_jobs=self._n_jobs
                    )

                # store hyperparameters of best model
                for hyp_param, value in search_results["best_params"].items():
                    key = "{}_{}".format(model_name, hyp_param)
                    # best_hyperparams[key] = str(value)
                    best_hyperparams[key] = str(value)

                # get best model metrics if reporting on validation
                if not self._eval_on_test:
                    # get best model
                    results_df = pd.DataFrame(search_results["cv_results"])
                    best_model = \
                        results_df.iloc[search_results["best_index"], :]

                    # get best model metrics
                    metric_names = self._get_metric_names()
                    metrics = {}

                    for metric_name in metric_names:
                        for split in ["test", "train"]:
                            # first part of key must be metric name,
                            # because this is used later to flip signs
                            # of some metrics
                            key = "{}_{}".format(metric_name, model_name)
                            if split == "train":
                                key += "_{}".format(split)
                            # get mean and std of metric
                            for stat in ["mean", "std"]:
                                col_name = "_".join([stat, split, metric_name])
                                metrics["_".join([key, stat])] = \
                                    best_model[col_name].item()

                # get best estimator if reporting on test
                else:
                    best_estimator = search_results["best_estimator"]
            # if no tuning is required
            else:
                if not self._eval_on_test:
                    # fit model
                    cv_results = cross_validate(
                        estimator=dt_model,
                        X=train_features,
                        y=train_targets,
                        cv=self._num_folds,
                        scoring=scorers,
                        return_train_score=True,
                        return_estimator=False,
                        error_score="raise",
                        n_jobs=self._n_jobs,
                    )

                    # get model metrics
                    metric_names = self._get_metric_names()
                    metrics = {}
                    for metric_name in metric_names:
                        for split in ["test", "train"]:
                            in_key = "{}_{}".format(split, metric_name)
                            fold_values = np.array(cv_results[in_key])
                            # add mean and std of values
                            out_key = "{}_{}".format(metric_name, model_name)
                            if split == "train":
                                out_key += "_{}".format(split)
                            metrics[out_key + "_mean"] = fold_values.mean().item()
                            metrics[out_key + "_std"] = fold_values.std().item()
                else:
                    # fit on train (or train+valid if flag for that is True)
                    if not self._type_of_task == "clustering":
                        best_estimator = dt_model.fit(train_features, train_targets)
                    else:
                        best_estimator = dt_model.fit(train_features)

                # store hyperparameters for tracing
                model_params = dt_model.get_params()
                best_hyperparams = {}
                for parameter in self._hyperparameters[model_name].keys():
                    best_hyperparams[model_name + "_" + parameter] = \
                        str(model_params[parameter])

            # get final performance metrics on validation data
            if not self._eval_on_test:
                # store metrics in all metrics
                for metric, value in metrics.items():
                    # turn negatives into positives for minimized
                    # metrics (scikit-learn minimizes with sign flip)
                    if metric.split("_")[0] in MINIMIZED_METRICS and \
                            "std" not in metric:
                        value = value * -1
                    all_metrics[metric] = value
            # get final performance metrics on test data
            else:
                # get predict features
                predict_features, predict_targets = \
                    self._get_features_and_targets(self._test_set)
                # TODO this if else has repeated code! Fix this!
                if self._tune_model or self._test_data_num_times <= 1:
                    # predict
                    predictions = best_estimator.predict(
                        predict_features
                    )

                    # metrics
                    metrics = self._compute_metrics(
                        model_name, predict_targets, predictions
                    )

                    # store metrics in all metrics
                    for metric, value in metrics.items():
                        all_metrics[metric] = float(value)

                    # add metrics on train to check overfitting
                    predictions = best_estimator.predict(train_features)
                    metrics = self._compute_metrics(
                        model_name, train_targets, predictions
                    )
                    for metric, value in metrics.items():
                        all_metrics[metric + "_train"] = float(value)
                else:
                    test_data_metrics = ddict(list)
                    for i in range(self._test_data_num_times):
                        # fit
                        if not self._type_of_task == "clustering":
                            estimator = dt_model.fit(train_features, train_targets)
                        else:
                            estimator = dt_model.fit(train_features)

                        # predict
                        predictions = estimator.predict(
                            predict_features
                        )

                        # get metrics
                        metrics = self._compute_metrics(
                            model_name, predict_targets, predictions
                        )

                        # store metrics in test_data_metrics dict
                        for metric, value in metrics.items():
                            test_data_metrics[metric].append(float(value))

                        # add metrics on train to check overfitting
                        predictions = estimator.predict(train_features)
                        metrics = self._compute_metrics(
                            model_name, train_targets, predictions
                        )
                        for metric, value in metrics.items():
                            test_data_metrics[metric + "_train"].append(float(value))

                    # aggregate metrics and store in all_metrics
                    for metric, value in test_data_metrics.items():
                        mean = np.array(value).mean().item()
                        std = np.array(value).std().item()
                        all_metrics[metric] = mean
                        all_metrics[metric + "_mean"] = mean
                        all_metrics[metric + "_std"] = std
                        all_metrics["text_" + metric] = str(mean) + "+-" + str(std)
            epoch_time += time.time()

        # update trace with results
        self.current_trace["epoch"].update(
            dict(
                epoch_time=epoch_time,
                event="eval_completed",
                **all_metrics,
                **best_hyperparams,
            )
        )

    def _wrapper_randomizedsearch(
            self,
            dt_model,
            search_space,
            features,
            targets,
            scorers,
            selection_metric,
            n_jobs,
    ):
        """
        Tunes given DT model with scikit-learn's RandomizedSearchCV and returns
        the search_results data dict.

        :param dt_model: downstream model
        :param search space: search space for given downstream model
        :param features: features to train on
        :param targets: targets to train on
        :param scorers: metrics to compute
        :param selection_metric: which of the given metrics to use for selection
        :param n_jobs: number of jobs for scikit-learn RandomizedSearchCV
        :return: search_results dict
        """

        # init randomized search
        randcv = RandomizedSearchCV(
            estimator=dt_model,
            param_distributions=search_space,
            n_iter=self._num_random_samples,
            cv=self._num_folds,
            scoring=scorers,
            refit=selection_metric,
            return_train_score=True,
            error_score="raise",
            n_jobs=n_jobs,
        )

        # fit it for glory!
        results = randcv.fit(features, targets)

        # create output dict
        search_results = {}
        search_results["best_params"] = results.best_params_
        search_results["cv_results"] = results.cv_results_
        search_results["best_index"] = results.best_index_
        search_results["best_estimator"] = results.best_estimator_

        return search_results

    def _sobol_tuner(
            self,
            model_name,
            search_space,
            features,
            targets,
            scorers,
            selection_metric,
            n_jobs,

    ):
        """
        Manually implements the entire tuning, model selection, refitting steps
        done with RandomizedSearchCV, but with SOBOL sampling. Returns the
        same search_results data dict as the one in wrapper_randimizedcv.

        :param model_name: downstream model name
        :param search space: search space for given downstream model
        :param features: features to train on
        :param targets: targets to train on
        :param scorers: metrics to compute
        :param selection_metric: which of the given metrics to use for selection
        :param n_jobs: number of jobs for scikit-learn cross validate
        :return: search_results dict
        """

        # sobol sampling
        sobol = Sobol()
        trials = sobol.generate(
            search_space.dimensions, self._num_random_samples
        )
        hyp_names = search_space.dimension_names

        # run trials!
        params_list = []
        trial_results = []
        for trial in trials:
            # create dict of hyperparameters for this trial
            trial_params = {}
            for i in range(len(hyp_names)):
                # cast numpy bools to python bools as required by sklearn
                # skopt's space uses numpy.bool
                if isinstance(trial[i], numpy.bool_):
                    trial_params[hyp_names[i]] = bool(trial[i])
                else:
                    trial_params[hyp_names[i]] = trial[i]
            params_list.append(trial_params)

            # init model with hyperparameters from this trial
            dt_model, _ = self._init_downstream_model(model_name)
            dt_model.set_params(**trial_params)

            # run model!
            trial_results.append(cross_validate(
                estimator=dt_model,
                X=features,
                y=targets,
                cv=self._num_folds,
                scoring=scorers,
                return_train_score=True,
                return_estimator=False,
                error_score="raise",
                n_jobs=n_jobs,
                )
            )

        # process cv results
        raw_results = ddict(list)
        for trial_result in trial_results:
            for k, v in trial_result.items():
                raw_results[k].append(v)
        for k, v in raw_results.items():
            raw_results[k] = np.stack(v, axis=0)
        cv_results = {}
        # compute means+std of everything for each trial
        for k, v in raw_results.items():
            for i in range(self._num_folds):
                cv_results["split{}_{}".format(i, k)] = v[:, i]
            cv_results["mean_{}".format(k)] = np.average(v, axis=1)
            cv_results["std_{}".format(k)] = np.std(v, axis=1)
        # add params to cv_results
        cv_results["params"] = params_list

        # add ranks (as done in sklearn)
        cv_results["rank_test_{}".format(selection_metric)] = np.asarray(
            self._rankdata(
                cv_results["mean_test_{}".format(selection_metric)] * -1,
                method="min"
            ), dtype=np.int32
        )

        # find best estimator
        best_index = \
            cv_results["rank_test_{}".format(selection_metric)].argmin()
        best_params = params_list[best_index]

        # fit best estimator to entire data
        best_estimator, _ = self._init_downstream_model(model_name)
        best_estimator.set_params(**best_params)
        best_estimator.fit(features, targets)

        # create output dict
        search_results = {}
        search_results["cv_results"] = cv_results
        search_results["best_index"] = best_index
        search_results["best_params"] = best_params
        search_results["best_estimator"] = best_estimator

        return search_results

    # Taken as is from scipy's source: scipy/stats/stats.py#8631
    def _rankdata(self, a, method='average', *, axis=None):
        """Assign ranks to data, dealing with ties appropriately.

        By default (``axis=None``), the data array is first flattened, and a flat
        array of ranks is returned. Separately reshape the rank array to the
        shape of the data array if desired (see Examples).

        Ranks begin at 1.  The `method` argument controls how ranks are assigned
        to equal values.  See [1]_ for further discussion of ranking methods.

        Parameters
        ----------
        a : array_like
            The array of values to be ranked.
        method : {'average', 'min', 'max', 'dense', 'ordinal'}, optional
            The method used to assign ranks to tied elements.
            The following methods are available (default is 'average'):

              * 'average': The average of the ranks that would have been assigned to
                all the tied values is assigned to each value.
              * 'min': The minimum of the ranks that would have been assigned to all
                the tied values is assigned to each value.  (This is also
                referred to as "competition" ranking.)
              * 'max': The maximum of the ranks that would have been assigned to all
                the tied values is assigned to each value.
              * 'dense': Like 'min', but the rank of the next highest element is
                assigned the rank immediately after those assigned to the tied
                elements.
              * 'ordinal': All values are given a distinct rank, corresponding to
                the order that the values occur in `a`.
        axis : {None, int}, optional
            Axis along which to perform the ranking. If ``None``, the data array
            is first flattened.

        Returns
        -------
        ranks : ndarray
             An array of size equal to the size of `a`, containing rank
             scores.

        References
        ----------
        .. [1] "Ranking", https://en.wikipedia.org/wiki/Ranking

        Examples
        --------
        >>> from scipy.stats import rankdata
        >>> rankdata([0, 2, 3, 2])
        array([ 1. ,  2.5,  4. ,  2.5])
        >>> rankdata([0, 2, 3, 2], method='min')
        array([ 1,  2,  4,  2])
        >>> rankdata([0, 2, 3, 2], method='max')
        array([ 1,  3,  4,  3])
        >>> rankdata([0, 2, 3, 2], method='dense')
        array([ 1,  2,  3,  2])
        >>> rankdata([0, 2, 3, 2], method='ordinal')
        array([ 1,  2,  4,  3])
        >>> rankdata([[0, 2], [3, 2]]).reshape(2,2)
        array([[1. , 2.5],
              [4. , 2.5]])
        >>> rankdata([[0, 2, 2], [3, 2, 5]], axis=1)
        array([[1. , 2.5, 2.5],
               [2. , 1. , 3. ]])

        """
        if method not in ('average', 'min', 'max', 'dense', 'ordinal'):
            raise ValueError('unknown method "{0}"'.format(method))

        if axis is not None:
            a = np.asarray(a)
            if a.size == 0:
                # The return values of `normalize_axis_index` are ignored.  The
                # call validates `axis`, even though we won't use it.
                # use scipy._lib._util._normalize_axis_index when available
                np.core.multiarray.normalize_axis_index(axis, a.ndim)
                dt = np.float64 if method == 'average' else np.int_
                return np.empty(a.shape, dtype=dt)
            return np.apply_along_axis(rankdata, axis, a, method)

        arr = np.ravel(np.asarray(a))
        algo = 'mergesort' if method == 'ordinal' else 'quicksort'
        sorter = np.argsort(arr, kind=algo)

        inv = np.empty(sorter.size, dtype=np.intp)
        inv[sorter] = np.arange(sorter.size, dtype=np.intp)

        if method == 'ordinal':
            return inv + 1

        arr = arr[sorter]
        obs = np.r_[True, arr[1:] != arr[:-1]]
        dense = obs.cumsum()[inv]

        if method == 'dense':
            return dense

        # cumulative counts of each unique value
        count = np.r_[np.nonzero(obs)[0], len(obs)]

        if method == 'max':
            return count[dense]

        if method == 'min':
            return count[dense - 1] + 1

        # average method
        return .5 * (count[dense] + count[dense - 1] + 1)

    def _init_downstream_model(self, model_name, hyperparameters=None):
        """
        Returns initialized scikit-learn downstream task model as well
        as corresponding search space for hyperparameter tuning.

        :param model_name: name of model to initialize
        :return: model object, search space dict
        """

        # TODO search spaces should be taken from config
        #   but hardcoded for now to simplify deployment of experiments
        prefix = ""
        if self._type_of_task == "regression" and (self._z_scores or self._log):
            prefix = "regressor__"
        if model_name == "knn":
            model = KNeighborsClassifier()
            if self._sobol:
                # version for SOBOL
                search_space = Space([
                    Categorical(list(range(1, 11)), name=prefix + "n_neighbors"),
                ])
            else:
                # version for RandomizedSearchCV
                search_space = {
                    prefix + "n_neighbors": list(range(1, 11)),
                }
        elif model_name == "logistic_regression":
            model = OneVsRestClassifier(
                LogisticRegression(max_iter=1000)
            )
            if self._sobol:
                # version for SOBOL
                search_space = Space([
                    Real(100, 100000, name=prefix + "estimator__C"),
                ])
            else:
                # version for RandomizedSearchCV
                search_space = {
                    prefix + "estimator__C": scipy.stats.uniform(loc=100, scale=100000),
                }
        elif model_name == "mlp":
            if self._type_of_task == "entity_classification":
                model = MLPClassifier()
            elif self._type_of_task == "regression":
                model = MLPRegressor()
            if self._sobol:
                # version for SOBOL
                search_space = Space([
                    Categorical([(100,), (10,), (100, 100), (10, 10)], name=prefix + "hidden_layer_sizes"),
                    Real(0.00001, 0.01, "uniform", name=prefix + "alpha"),
                    Real(0.001, 0.01, "uniform", name=prefix + "learning_rate_init"),
                    Categorical(["adam", "lbfgs"], name=prefix + "solver"),
                    # max iter set as in Jain et al, ESWC21.
                    Categorical([1000], name=prefix + "max_iter"),
                    Categorical([False], name=prefix + "early_stopping"),
                    # Categorical([True], name=prefix + "early_stopping"),
                ])
            else:
                # version for RandomizedSearchCV
                search_space = {
                    prefix + "hidden_layer_sizes": [(100,), (10,), (100, 100), (10, 10)],
                    prefix + "alpha": scipy.stats.uniform(loc=0.00001, scale=0.01),
                    prefix + "learning_rate_init": scipy.stats.uniform(loc=0.001, scale=0.01),
                    prefix + "solver": ["adam", "lbfgs"],
                    # max iter set as in Jain et al, ESWC21.
                    prefix + "max_iter": [1000],
                    prefix + "early_stopping": [False],
                    # prefix + "early_stopping": [True],
                }
        elif model_name == "random_forest":
            if self._type_of_task == "entity_classification":
                model = RandomForestClassifier()
            elif self._type_of_task == "regression":
                model = RandomForestRegressor()
            if self._sobol:
                # version for SOBOL
                search_space = Space([
                    Categorical([100, 10, 50, 200], name=prefix + "n_estimators")
                ])
            else:
                # version for RandomizedSearchCV
                search_space = {
                    prefix + "n_estimators": [100, 10, 50, 200],
                }
        elif model_name == "linear_regression":
            model = Ridge(max_iter=1000)
            if self._sobol:
                # version for SOBOL
                search_space = Space([
                    Real(0.00001, 0.01, name=prefix + "alpha"),
                ])
            else:
                # version for RandomizedSearchCV
                search_space = {
                    prefix + "alpha": scipy.stats.uniform(loc=0.00001, scale=0.01),
                }
        elif model_name == "kmeans":
                model = KMeans(max_iter=1000)
                # for now we set K to default value in scikit-learn
                # our tests will manually set K and not tune anything
                if self._sobol:
                    # version for SOBOL
                    search_space = Space([
                        Categorical(list(range(1, 11)), name=prefix + "n_clusters"),
                    ])
                else:
                    # version for RandomizedSearchCV
                    search_space = {
                        prefix + "n_clusters": list(range(1, 11)),
                    }
        else:
            raise ValueError(
                "Unsupported downstream model for {}: {}".format(
                    self._type_of_task, model_name
                )
            )

        # wrap model with transformers if required
        if self._type_of_task == "regression" and self._z_scores:
            model = TransformedTargetRegressor(
                regressor=model, transformer=StandardScaler()
            )
        elif self._type_of_task == "regression" and self._log:
            model = TransformedTargetRegressor(
                regressor=model, func=np.log, inverse_func=np.exp
            )

        # set hyperparameters if given
        if hyperparameters is not None:
            # set model hyperparameters
            # TODO remove this hack and figure out how to get tuples and bools from terminal parameters
            # convert hidden_layer_sizes to tuple
            for hyp_name, hyp_value in hyperparameters.items():
                # if "hidden_layer_sizes" in hyperparameters and isinstance(hyperparameters["hidden_layer_sizes"], str):
                #     str_value = hyperparameters["hidden_layer_sizes"].strip()[1:-1]
                #     hyperparameters["hidden_layer_sizes"] = tuple(map(int, str_value.split(',')))
                # if "early_stopping" in hyperparameters and isinstance(hyperparameters["early_stopping"], str):
                #     str_value = hyperparameters["early_stopping"]
                #     hyperparameters["early_stopping"] = str_value in ["True", "true"]
                if "hidden_layer_sizes" in hyp_name and isinstance(hyperparameters[hyp_name], str):
                    str_value = hyperparameters[hyp_name].strip()[1:-1]
                    str_value_split = str_value.split(",")
                    first_layer = int(str_value_split[0])
                    if str_value_split[1]:
                        tuple_ = (first_layer, int(str_value_split[1]))
                    else:
                        tuple_ = (first_layer, )
                    hyperparameters[hyp_name] = tuple_
                if "early_stopping" in hyp_name and isinstance(hyperparameters[hyp_name], str):
                    str_value = hyperparameters[hyp_name]
                    hyperparameters[hyp_name] = str_value in ["True", "true"]

            # add prefix required by model scikit-learn wrappers if not given by user
            hyps_with_prefix = {}
            for k, v in hyperparameters.items():
                if prefix not in k:
                    hyps_with_prefix[prefix + k] = v
                else:
                    hyps_with_prefix[k] = v
            # rewrite user given hyperparameters so we later trace values from
            # the instantiated model instead of tracing what the user provided
            hyperparameters = dict(hyps_with_prefix)
            self._hyperparameters[model_name] = hyperparameters
            # set hyperparameters (finally...)
            model.set_params(**hyperparameters)

        return model, search_space

    @staticmethod
    def _set_targets_for_multilabel_task(targets):
        """
        Prepares target values to format required by scikit-learn
        """

        targets_multilabel = []
        for row in targets:
            # targets_multilabel.append(row.tolist())
            targets_multilabel.append(row.int().tolist())
        return targets_multilabel

    def _get_features_and_targets(self, split):
        """
        Returns embedding features and labels from given downstream split
        """

        # move to CPU so conversion to numpy is automatic
        features = self.model._entity_embedder.embed(
            split[:, 0].to(self.device)
        ).to("cpu")
        targets = split[:, 1:]

        # set targets for multilabel classification
        if targets.size()[1] > 1:
            targets = self._set_targets_for_multilabel_task(
                targets
            )

        # avoid scikit-learn warning
        if torch.is_tensor(targets):
            targets = targets.view(-1)

        return features, targets

    def _get_scorers(self):
        """
        Initializes scorers to be used in random search.
        """

        if self._type_of_task == "entity_classification":
            scorers = {
                "accuracy": make_scorer(skmetrics.accuracy_score),
                "weighted_f1": make_scorer(
                    skmetrics.f1_score,
                    average="weighted",
                ),
                "weighted_precision": make_scorer(
                    skmetrics.precision_score,
                    average="weighted",
                ),
                "weighted_recall": make_scorer(
                    skmetrics.recall_score,
                    average="weighted",
                ),
            }
        elif self._type_of_task == "regression":
            scorers = {
                "mse": make_scorer(
                    skmetrics.mean_squared_error,
                    greater_is_better=False
                ),
                "rse": make_scorer(
                    self._relative_squared_error,
                    greater_is_better=False
                ),
                # "ndcg_at_100": make_scorer(self._ndcg_wrapper),
                "spearman": make_scorer(self._spearman_wrapper),
            }
        elif self._type_of_task == "clustering":
            scorers = {
                "weighted_f1": make_scorer(
                    self._clustering_weighted_f1,
                ),
            }
        else:
            raise ValueError("Unrecognized type of downstream task: {}".format(
                self._type_of_task
                )
            )

        return scorers

    @staticmethod
    def _relative_squared_error(targets, predictions, details=False):
        """
        Computes relative squared error (RSE) of given predictions and targets.

        :param predictions: torch tensor of size n x 1 with n model predictions
        :param targets: torch tensor of size n x 1 with the n target values
        :param details: flag to return numerator and denominator separately
        :return: RSE value (also numerator and denominator if details = True)
        """

        # turn predictions to torch tensor
        predictions = torch.from_numpy(predictions)

        # turn targets to float (scikit-learn gives long)
        targets = targets.float()

        # avoid accidental broadcasting
        targets = targets.view(-1)
        predictions = predictions.view(-1)

        # numerator
        squared_error = torch.sum(
            torch.square(predictions - targets.view(-1))
        )
        # denominator
        targets_from_mean = torch.sum(
            torch.square(torch.mean(targets.view(-1)) - targets.view(-1))
        )

        if details:
            return squared_error / targets_from_mean, \
                   squared_error, \
                   targets_from_mean
        else:
            return squared_error / targets_from_mean

    @staticmethod
    def _clustering_weighted_f1(targets, predictions):
        """
        Computes weighted F1 based on clustering assignments. Given a clustering
        assignment, we assign to each member of a cluster, the majority class
        in that cluster. Then we treat these class assignments as predicted
        classes and compute weighted F1

        :param predictions: torch tensor of size n x 1 with n model predictions
        :param targets: torch tensor of size n x 1 with the n target values
        :return: weighted F1 value
        """

        # TODO why weighted F1 if we treat this as single class classification?
        #   That is because we predict the majority class per cluster, i.e. a
        #   single label. For actual multilabel datasets, this is less smart
        #   than what the EC classifiers are doing, which is providing
        #   predictions for multilabel classification

        # turn targets to numpy for smart indexing
        targets = np.array(targets)

        # create new predictions array with format required by scikit-learn
        new_predictions = np.zeros((targets.shape[0], targets.shape[1]), int)

        # turn cluster assignments into classification predictions
        # we turn majority class in each cluster to predicted class for all
        # members of that cluster
        clusters = np.unique(predictions)
        for cluster in clusters:
            # get members of this cluster
            cluster_members = np.nonzero(predictions == cluster)[0]
            # get majority class for this cluster
            cluster_labels = targets[cluster_members]
            majority_class = np.argmax(np.sum(cluster_labels, axis=0))
            # set predictions to majority class
            new_predictions[cluster_members, majority_class] = 1

        return skmetrics.f1_score(targets, new_predictions, average="weighted")

    @staticmethod
    def _spearman_wrapper(targets, predictions):
        """
        Wrapper so spearman correlation can be used with as scorer.
        """

        return scipy.stats.spearmanr(targets, predictions)[0]

    @staticmethod
    def _ndcg_wrapper(targets, predictions):
        """
        Wrapper to use NGDC@100 without as scorer without a warning
        """

        return skmetrics.ndcg_score(
            targets.reshape(1, -1),
            predictions.reshape(1, -1),
            k=100
        )

    def _get_metric_names(self):
        """
        Returns metric names depending on type of task
        """

        metric_names = []
        if self._type_of_task == "entity_classification":
            metric_names = [
                "accuracy",
                "weighted_f1",
                "weighted_precision",
                "weighted_recall",
            ]
        elif self._type_of_task == "regression":
            metric_names = [
                "mse",
                "rse",
                # "ndcg_at_100",
                "spearman",
            ]
        elif self._type_of_task == "clustering":
            metric_names = [
                "weighted_f1",
            ]

        return metric_names

    def _compute_metrics(self, model_name, targets, predictions):
        """
        Computes metrics depending on type of task
        """

        if self._type_of_task == "entity_classification":
            metrics = {
                "accuracy_{}".format(model_name): skmetrics.accuracy_score(
                    targets, predictions
                ),
                "weighted_precision_{}".format(model_name):
                    skmetrics.precision_score(
                        targets, predictions, average="weighted"
                    ),
                "weighted_recall_{}".format(model_name):
                    skmetrics.recall_score(
                        targets, predictions, average="weighted"
                    ),
                "weighted_f1_{}".format(model_name):
                    skmetrics.f1_score(
                        targets, predictions, average="weighted"
                    ),
            }
        elif self._type_of_task == "regression":
            rse, rse_num, rse_den = self._relative_squared_error(
                targets, predictions, details=True
            )
            metrics = {
                "mse_{}".format(model_name): skmetrics.mean_squared_error(
                    targets, predictions
                ),
                "rse_{}".format(model_name): rse,
                "rse_numerator_{}".format(model_name): rse_num,
                "rse_denominator_{}".format(model_name): rse_den,
                # "ndcg@100_{}".format(model_name): self._ndcg_wrapper(
                #     targets, predictions
                # ),
                "spearman_{}".format(model_name): self._spearman_wrapper(
                    targets, predictions
                ),
            }
        elif self._type_of_task == "clustering":
            # TODO here you have to send what you need, i.e. true labels
            metrics = {
                "weighted_f1_{}".format(model_name):
                    self._clustering_weighted_f1(
                        targets, predictions
                    )
            }
        else:
            raise ValueError("Unrecognized type of task: {}".format(
                self._type_of_task
                )
            )

        return metrics
