from kge import Config, Dataset
from kge.model.kge_model import KgeModel

SLOTS = [0, 1, 2, 3]
S, P, O, B = SLOTS


class ModelWithPlaceholders(KgeModel):
    """
    Modifies a base model to have extra placeholder embeddings to be used with
    classification training objectives, e.g. s*?.
    """

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        self._init_configuration(config, configuration_key)

        # Initialize base model
        # We use a dataset with an extra relation embedding for the placeholder
        # for relations, and two extra embeddings for the placeholders for
        # subject and object slots
        alt_dataset = dataset.shallow_copy()
        alt_dataset._num_relations = dataset.num_relations() + 1
        alt_dataset._num_entities = dataset.num_entities() + 2
        base_model = KgeModel.create(
            config=config,
            dataset=alt_dataset,
            configuration_key=self.configuration_key + ".base_model",
            init_for_load_only=init_for_load_only,
        )

        # Initialize this model
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=base_model.get_scorer(),
            create_embedders=False,
            init_for_load_only=init_for_load_only,
        )
        self._base_model = base_model
        # TODO change entity_embedder assignment to sub and obj embedders when
        # support for that is added
        self._entity_embedder = self._base_model.get_s_embedder()
        self._relation_embedder = self._base_model.get_p_embedder()

        # indices for placeholder embeddings
        self.pseudo_indices = {
            S: self._base_model.dataset.num_entities() - 2,
            P: self._base_model.dataset.num_relations() - 1,
            O: self._base_model.dataset.num_entities() - 1
        }

    def prepare_job(self, job, **kwargs):
        self._base_model.prepare_job(job, **kwargs)

    def penalty(self, **kwargs):
        return self._base_model.penalty(**kwargs)
