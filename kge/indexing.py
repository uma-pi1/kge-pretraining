import torch
import numba
import numpy as np
from typing import Iterator, List, Tuple, Dict


class KvsAllIndex:
    """Construct an index from keys (e.g., sp) to all its values (o).

    Keys are tuples, values are PyTorch tensors.

    Internally stores list of unique keys, list of values, and starting offset of each
    key in values in PyTorch tensors. Access by key is enabled using an index on top of
    these tensors. The tensors can also be used directly (e.g., in KvsAll training)

    """

    def __init__(
        self,
        triples: torch.Tensor,
        key_cols: List,
        value_col: int,
        default_factory: type,
        num_hops=1,
    ):
        """
        Args:
            triples: data
            key_cols: the two columns used as keys
            value_col: column used as value
            default_factory: default return type
        """
        self.key_cols = key_cols
        self.value_col = value_col

        # sort triples, extract unique keys, all values, and starting offset of each key
        triples_sorted = KvsAllIndex.sort_triples_by_keys(triples, key_cols, value_col)
        keys, values_offset = np.unique(
            triples_sorted[:, key_cols], axis=0, return_index=True
        )
        values_offset = np.append(values_offset, len(triples_sorted))

        # create dictionary from key (as a tuple) to key index
        self.default_index_of_key = -1
        self._index_of_key = self._create_index_of_key_dict(keys)

        # expand index to include n relation-specific hops
        if num_hops > 1:
            print("Creating {}-hop index from cols {} to cols {}".format(
                num_hops, key_cols, value_col
                )
            )

            # get relevant slot positions to construct new query instances
            if len(key_cols) > 1:
                # get 2nd key col, i.e. the relation slot in sp or po
                key_slot_2 = key_cols.index(1)
                # key col 1 is the other slot, i.e. the entity slot in sp or po
                key_slot_1 = (key_cols.index(1) + 1) % 2
            else:
                # there is a single key col, eg in s_to_o indexes
                key_slot_1 = 0
                key_slot_2 = None

            # set 1 hop neighborhood values and offsets to expand hops
            # values will be added to new hops, so we use a lists (friendlier than torch tensor)
            one_hop_offsets = values_offset
            one_hop_values = list(triples_sorted[:, self.value_col].to(dtype=torch.int32).numpy())
            # one_hop_values = list(triples_sorted[:, self.value_col].clone().to(dtype=torch.int32).numpy())

            # set values and offsets to be expanded to  n + 1 hop neighborhood
            # we start with 1 hop, but detached because they will change
            # but we need to keep the 1 hop neighborhoods
            new_hop_offsets = one_hop_offsets.copy()
            new_hop_values = one_hop_values.copy()

            # loop over hops
            for hop_number in range(num_hops - 1):
                # set neighborhood from previous hop, this must remain fixed
                previous_hop_offsets = new_hop_offsets.copy()
                previous_hop_values = new_hop_values.copy()

                # loop over existing (fixed set of) keys
                for current_pos, current_key in enumerate(keys):
                    # get previous hop neighborhood of current key
                    current_offset = previous_hop_offsets[current_pos]
                    next_offset = previous_hop_offsets[current_pos + 1]
                    current_vals = previous_hop_values[current_offset:next_offset]

                    # keep set of current vals, which will be updated with every
                    # new entry in the neighborhood to avoid repetition
                    current_vals_set = set(current_vals)

                    # loop over values for current key
                    for val in current_vals:
                        # construct new key for new hop
                        if key_slot_2 is not None:
                            new_key = [0, 0]
                            new_key[key_slot_2] = int(current_key[key_slot_2])
                            new_key[key_slot_1] = int(val.item())
                            new_key = tuple(new_key)
                        else:
                            # -1 is used as dummy key to keep key type consistent
                            # for numba
                            new_key = [-1, -1]
                            new_key[key_slot_1] = int(val.item())
                            new_key = tuple(new_key)

                        # get 1 hop neighborhood of new key
                        try:
                            new_key_pos = self._index_of_key[new_key]
                        except KeyError:
                            # this tuple does not even have 1-hop neighbors
                            continue
                        new_offset = one_hop_offsets[new_key_pos]
                        new_next_offset = one_hop_offsets[new_key_pos + 1]
                        new_vals = one_hop_values[new_offset:new_next_offset]

                        # compute set difference between current and new values
                        # current also includes values being added for this hop
                        # current_vals_set = set(new_hop_values[current_offset:next_offset])
                        new_vals_set = set(new_vals)
                        vals_diff = new_vals_set.difference(current_vals_set)

                        # add new values to values of current key, not new key!
                        # we insert new values at the end of set of already
                        # expanded values
                        upd_next_offset = new_hop_offsets[current_pos + 1]
                        new_hop_values[upd_next_offset:upd_next_offset] = list(vals_diff)

                        # modify all necessary offsets accordingly
                        new_hop_offsets[current_pos + 1:] += len(vals_diff)

                        # add new entries to set of current vals
                        current_vals_set.update(vals_diff)

        # convert data structures to pytorch and keep them
        self._keys = torch.from_numpy(keys)
        if num_hops > 1:
            self._values_offset = torch.from_numpy(new_hop_offsets).int()
            self._values = torch.tensor(new_hop_values)
        else:
            self._values_offset = torch.from_numpy(values_offset).int()
            self._values = triples_sorted[:, self.value_col].clone()  # to drop reference to triples_sorted

        self.default_factory = default_factory

    @staticmethod
    @numba.njit(boundscheck=True)
    # @numba.njit()
    def _create_index_of_key_dict(keys: np.ndarray) -> Dict[Tuple[int, int], int]:
        """
        Creates a dictionary mapping keys to the key_index needed for value-lookup
        Args:
            keys: [n, 2], int32, Tensor containing keys

        Returns:
            Dictionary with key-tuple as key and key_index as value
        """

        keys = keys.astype(np.int32)
        index_of_key = dict()
        for key_index in range(len(keys)):
            # keys may be one or two slots, e.g. (s,) or (s,p)
            if keys.shape[1] > 1:
                index_of_key[(keys[key_index, 0].item(), keys[key_index, 1].item())] = key_index
            else:
                # we use dummy 2nd dim, because numba requires that all keys be the same type
                index_of_key[(keys[key_index, 0].item(), -1)] = key_index

        return index_of_key

    def __getstate__(self):
        """We can not pickle a numba dict. Remove from state dict"""
        state = self.__dict__.copy()
        del state["_index_of_key"]
        return state

    def __setstate__(self, state):
        """Numba dict was not pickled. Restore here"""
        self.__dict__.update(state)
        self._index_of_key = self._create_index_of_key_dict(self._keys.numpy())

    def __getitem__(self, key, default_return_value=None) -> torch.Tensor:
        key = self._convert_key_to_correct_type(key)
        key_index = self._index_of_key.get(key, self.default_index_of_key)
        if key_index < 0:
            if default_return_value is None:
                return self.default_factory()
            return default_return_value
        return self._values_of(key_index)

    @staticmethod
    def _convert_key_to_correct_type(key):
        """Numba dict expects np.int32. Ensure all inputs are converted correctly"""
        key_type = type(key[0])
        if key_type is np.int32:
            return key
        if key_type is np.int64:
            return key[0].astype(np.int32), key[1].astype(np.int32)
        if key_type is int:
            return np.int32(key[0]), np.int32(key[1])
        if key_type is torch.Tensor:
            return np.int32(key[0].item()), np.int32(key[1].item())
        raise ValueError("Input type for key attribute in KvsAllIndex needs to be Tuple[np.int32,np.int32]")

    def _values_of(self, key_index) -> torch.Tensor:
        start = self._values_offset[key_index]
        end = self._values_offset[key_index + 1]
        return self._values[start:end]

    @staticmethod
    @numba.njit()
    def _get_all_impl(
            keys: np.ndarray,
            index_of_key: Dict[Tuple[int, int], int],
            values: np.ndarray,
            values_offset: np.ndarray
    ):
        """
        Looks up all values corresponding to keys and outputs them in a single tensor
        Args:
            keys: [n, 2] Tensors with keys to look up
            index_of_key: dict mapping keys to key_index. Provide self._index_of_keys
            values: [n,] Tensor containing values. Provide self._values.numpy()
            values_offset: [n,] Tensor mapping key_index to offset in value tensor.
                Provide self._values_offset.numpy()

        Returns: Tensor[m, 2] with all values for all keys concatenated in [:, 1] and
            the corresponding input position of the key in [:, 0]

        """
        key_index = np.empty((len(keys)), dtype=np.int32)
        total_length = 0
        for i in range(len(keys)):
            index = index_of_key.get((keys[i, 0].item(), keys[i, 1].item()), -1)
            key_index[i] = index
            if index < 0:
                continue
            total_length += len(
                values[values_offset[key_index[i].item()]:values_offset[key_index[i].item()+1]]
            )
        result = np.empty((total_length, 2), dtype=np.int32)
        current_index = 0
        for i in range(len(key_index)):
            if key_index[i].item() < 0:
                continue
            res = (values[values_offset[key_index[i]]:values_offset[key_index[i]+1]])
            len_res = len(res)
            result[current_index: current_index+len_res, 0] = i
            result[current_index: current_index + len_res, 1] = res
            current_index += len_res
        return result

    def __len__(self):
        return len(self._keys)

    def get(self, key, default_return_value=None) -> torch.Tensor:
        return self.__getitem__(key, default_return_value)

    def get_all(self, keys):
        # keys need to be int32 otherwise numba won't find any matches in the dict
        keys = keys.int()
        return torch.from_numpy(
            self._get_all_impl(
                keys.numpy(), self._index_of_key,
                self._values.numpy(), self._values_offset.numpy()
            )
        )

    def keys(self) -> Iterator[Tuple[int, int]]:
        return self._index_of_key.keys()

    def values(self) -> Iterator[torch.Tensor]:
        return [self._values_of(key_index) for key_index in self._index_of_key.values()]

    def items(self) -> Iterator[Tuple[Tuple[int, int], torch.Tensor]]:
        return zip(self.keys(), self.values())

    @staticmethod
    def sort_triples_by_keys(
        triples: torch.Tensor, key_cols: List, value_col: int
    ) -> torch.Tensor:
        """Sorts triples by key_cols, then value_col."""
        # using numpy, since torch has no stable sort
        triples = triples.numpy()
        triples_sorted = triples[np.argsort(triples[:, value_col])]
        for key in key_cols[::-1]:
            triples_sorted = triples_sorted[
                np.argsort(triples_sorted[:, key], kind="stable")
            ]
        return torch.from_numpy(triples_sorted)


def index_KvsAll(dataset: "Dataset", split: str, key: str, value: str, num_hops=1):
    """Return an index for the triples in split (''train'', ''valid'', ''test'')
    from the specified key (''sp'' or ''po'' or ''so'') to the indexes of the
    remaining constituent (''o'' or ''s'' or ''p'' , respectively.)

    The index maps from `tuple' to `torch.LongTensor`.

    The index is cached in the provided dataset under name `{split}_sp_to_o` or
    `{split}_po_to_s`, or `{split}_so_to_p`. If this index is already present, does not
    recompute it.

    """

    slots_str = {
        "s": 0,
        "p": 1,
        "o": 2,
    }
    key_cols = []
    for slot in key:
        try:
            key_cols.append(slots_str[slot])
        except KeyError:
            raise KeyError("Unrecognized slot in query type with key {}".format(key))
    value_col = slots_str[value]

    name = split + "_" + key + "_to_" + value
    if num_hops > 1:
        name += "_{}_hops".format(num_hops)
    if not dataset._indexes.get(name):
        triples = dataset.split(split)
        dataset._indexes[name] = KvsAllIndex(
            triples, key_cols, value_col, torch.IntTensor, num_hops
        )

    dataset.config.log(
        "{} distinct {} pairs in {}".format(len(dataset._indexes[name]), key, split),
        prefix="  ",
    )

    return dataset._indexes.get(name)


def index_relation_types(dataset):
    """Classify relations into 1-N, M-1, 1-1, M-N.

    According to Bordes et al. "Translating embeddings for modeling multi-relational
    data.", NIPS13.

    Adds index `relation_types` with list that maps relation index to ("1-N", "M-1",
    "1-1", "M-N").

    """
    if "relation_types" not in dataset._indexes:
        # 2nd dim: num_s, num_distinct_po, num_o, num_distinct_so, is_M, is_N
        relation_stats = torch.zeros((dataset.num_relations(), 6))
        for index, p in [
            (dataset.index("train_sp_to_o"), 1),
            (dataset.index("train_po_to_s"), 0),
        ]:
            for prefix, labels in index.items():
                relation_stats[prefix[p], 0 + p * 2] = relation_stats[
                    prefix[p], 0 + p * 2
                ] + len(labels)
                relation_stats[prefix[p], 1 + p * 2] = (
                    relation_stats[prefix[p], 1 + p * 2] + 1.0
                )
        relation_stats[:, 4] = (relation_stats[:, 0] / relation_stats[:, 1]) > 1.5
        relation_stats[:, 5] = (relation_stats[:, 2] / relation_stats[:, 3]) > 1.5
        relation_types = []
        for i in range(dataset.num_relations()):
            relation_types.append(
                "{}-{}".format(
                    "1" if relation_stats[i, 4].item() == 0 else "M",
                    "1" if relation_stats[i, 5].item() == 0 else "N",
                )
            )

        dataset._indexes["relation_types"] = relation_types

    return dataset._indexes["relation_types"]


def index_relations_per_type(dataset):
    if "relations_per_type" not in dataset._indexes:
        relations_per_type = {}
        for i, k in enumerate(dataset.index("relation_types")):
            relations_per_type.setdefault(k, set()).add(i)
        dataset._indexes["relations_per_type"] = relations_per_type
    else:
        relations_per_type = dataset._indexes["relations_per_type"]

    dataset.config.log("Loaded relation index")
    for k, relations in relations_per_type.items():
        dataset.config.log(
            "{} relations of type {}".format(len(relations), k), prefix="  "
        )

    return relations_per_type


def index_frequency_percentiles(dataset, recompute=False):
    """
    :return: dictionary mapping from
    {
        'subject':
        {25%, 50%, 75%, top} -> set of entities
        'relations':
        {25%, 50%, 75%, top} -> set of relations
        'object':
        {25%, 50%, 75%, top} -> set of entities
    }
    """
    if "frequency_percentiles" in dataset._indexes and not recompute:
        return
    subject_stats = torch.zeros((dataset.num_entities(), 1))
    relation_stats = torch.zeros((dataset.num_relations(), 1))
    object_stats = torch.zeros((dataset.num_entities(), 1))
    for (s, p, o) in dataset.split("train"):
        subject_stats[s] += 1
        relation_stats[p] += 1
        object_stats[o] += 1
    result = dict()
    for arg, stats, num in [
        (
            "subject",
            [
                i
                for i, j in list(
                    sorted(enumerate(subject_stats.tolist()), key=lambda x: x[1])
                )
            ],
            dataset.num_entities(),
        ),
        (
            "relation",
            [
                i
                for i, j in list(
                    sorted(enumerate(relation_stats.tolist()), key=lambda x: x[1])
                )
            ],
            dataset.num_relations(),
        ),
        (
            "object",
            [
                i
                for i, j in list(
                    sorted(enumerate(object_stats.tolist()), key=lambda x: x[1])
                )
            ],
            dataset.num_entities(),
        ),
    ]:
        for percentile, (begin, end) in [
            ("25%", (0.0, 0.25)),
            ("50%", (0.25, 0.5)),
            ("75%", (0.5, 0.75)),
            ("top", (0.75, 1.0)),
        ]:
            if arg not in result:
                result[arg] = dict()
            result[arg][percentile] = set(stats[int(begin * num) : int(end * num)])
    dataset._indexes["frequency_percentiles"] = result


class IndexWrapper:
    """Wraps a call to an index function so that it can be pickled"""

    def __init__(self, fun, **kwargs):
        self.fun = fun
        self.kwargs = kwargs

    def __call__(self, dataset: "Dataset", **kwargs):
        self.fun(dataset, **self.kwargs)


def _invert_ids(dataset, obj: str):
    if not f"{obj}_id_to_index" in dataset._indexes:
        ids = dataset.load_map(f"{obj}_ids")
        inv = {v: k for k, v in enumerate(ids)}
        dataset._indexes[f"{obj}_id_to_index"] = inv
    else:
        inv = dataset._indexes[f"{obj}_id_to_index"]
    dataset.config.log(f"Indexed {len(inv)} {obj} ids", prefix="  ")


def create_default_index_functions(dataset: "Dataset"):
    for split in dataset.files_of_type("triples"):
        for key, value in [("sp", "o"), ("po", "s"), ("so", "p")]:
            # self assignment needed to capture the loop var
            dataset.index_functions[f"{split}_{key}_to_{value}"] = IndexWrapper(
                index_KvsAll, split=split, key=key, value=value
            )
    dataset.index_functions["relation_types"] = index_relation_types
    dataset.index_functions["relations_per_type"] = index_relations_per_type
    dataset.index_functions["frequency_percentiles"] = index_frequency_percentiles

    for obj in ["entity", "relation"]:
        dataset.index_functions[f"{obj}_id_to_index"] = IndexWrapper(
            _invert_ids, obj=obj
        )

    # TODO this is lazy and should not be created by default, only if the job
    #   requires it, so perhaps move this to a different function which will be
    #   called by a train of eval job that uses query types
    # map query types with ANY embeddings to corresponding index functions
    for split in dataset.files_of_type("triples"):
        query_types = [("s", "p"), ("s", "o"), ("p", "s"), ("p", "o"), ("o", "s"), ("o", "p")]
        for key, value in query_types:
            # self assignment needed to capture the loop var
            dataset.index_functions[f"{split}_{key}_to_{value}"] = IndexWrapper(
                index_KvsAll, split=split, key=key, value=value
            )

@numba.njit
def where_in(x, y, not_in=False):
    """Retrieve the indices of the elements in x which are also in y.

    x and y are assumed to be 1 dimensional arrays.

    :params: not_in: if True, returns the indices of the of the elements in x
    which are not in y.

    """
    # np.isin is not supported in numba. Also: "i in y" raises an error in numba
    # setting njit(parallel=True) slows down the function
    list_y = set(y)
    return np.where(np.array([i in list_y for i in x]) != not_in)[0]
