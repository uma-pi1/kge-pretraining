import torch
from torch import Tensor
from typing import List, Union

SLOTS = [0, 1, 2]
S, P, O = SLOTS


class QueryType:
    """
    Stores information about supported query types for training/evaluation
    """

    def __init__(self, query_type, weight=1.0):
        # check that query type is supported
        target_slot = query_type.find("_")
        if target_slot not in SLOTS:
            raise ValueError("{} not a valid query type".format(query_type))

        # Input slots for different target slots
        # Semantics: {target_slot: (input_slot_1, input_slot_2)}
        query_type_input_slots = {
            S: (1, 2),
            P: (0, 2),
            O: (0, 1),
        }

        # Score functions for different target slots
        # Semantics: {target_slot: score_function_str}
        query_type_score_functions = {
            S: "score_po",
            P: "score_so",
            O: "score_sp",
        }

        # Indexes for different target slots
        # Semantics: {query_type: index_str}
        query_type_indexes = {
            "sp_": "sp_to_o",
            "_po": "po_to_s",
            "s_o": "so_to_p",
            "s^_": "s_to_o",
            "_^o": "o_to_s",
            "^p_": "p_to_o",
            "_p^": "p_to_s",
            "s_^": "s_to_p",
            "^_o": "o_to_p",
        }

        self.name = query_type
        self.weight = weight
        self.target_slot = target_slot
        self.input_slots = query_type_input_slots[target_slot]
        self.score_fn = query_type_score_functions[target_slot]
        # add number of hops to index name if query type is multihop
        if query_type[-1].isnumeric():
            base_index = query_type_indexes[query_type[:-1]]
            self.index = "_".join([base_index, query_type[-1], "hops"])
        else:
            self.index = query_type_indexes[query_type]

    def get_key_value_cols(self):
        """
        Returns tuple of form (key_cols, value_cols) extracted from query_type.
        For example, for query type sp_o, returns ("sp", "o").
        """
        key_value_cols = {
            "sp_": ("sp", "o"),
            "_po": ("po", "s"),
            "s_o": ("so", "p"),
            "s^_": ("s", "o"),
            "_^o": ("o", "s"),
            "^p_": ("p", "o"),
            "_p^": ("p", "s"),
            "s_^": ("s", "p"),
            "^_o": ("o", "p"),
        }

        # remote number of hops
        base_query_type = self.name[:-1]

        return key_value_cols[base_query_type]


def get_label_coords_from_spo_batch(
        batch: Union[Tensor, List[Tensor]], query_type: QueryType, index: dict
) -> torch.Tensor:
    """
    Given a set of triples, a query type object and an index, lookup matches for
    input slots to target slots of given query type in given index.

    Each row in batch holds an (s,p,o) triple. Returns the non-zero coordinates
    of a 2-way binary tensor with one row per triple and num_entities columns.
    """

    if type(batch) is list:
        batch = torch.cat(batch).reshape((-1, 3)).int()

    # if key is two slots
    if len(query_type.index.split("_")[0]) > 1:
        input_slots = query_type.input_slots
        key = batch[:, [input_slots[0], input_slots[1]]]
    # if key is one slot
    else:
        slots_str = {
            "s": 0,
            "p": 1,
            "o": 2,
        }
        key_slot = slots_str[query_type.index.split("_")[0]]
        key = batch[:, key_slot]
        # add dummy dim
        dummy = torch.ones(len(key), dtype=torch.int32) * -1
        key = torch.stack((key, dummy), 1)

    return index.get_all(key)


def get_sp_po_coords_from_spo_batch(
    batch: Union[Tensor, List[Tensor]], num_entities: int, sp_index: dict, po_index: dict
) -> torch.Tensor:
    """Given a set of triples , lookup matches for (s,p,?) and (?,p,o).

    Each row in batch holds an (s,p,o) triple. Returns the non-zero coordinates
    of a 2-way binary tensor with one row per triple and 2*num_entites columns.
    The first half of the columns correspond to hits for (s,p,?); the second
    half for (?,p,o).

    """
    if type(batch) is list:
        batch = torch.cat(batch).reshape((-1, 3)).int()
    sp_coords = sp_index.get_all(batch[:, [0, 1]])
    po_coords = po_index.get_all(batch[:, [1, 2]])
    po_coords[:, 1] += num_entities
    coords = torch.cat(
        (
            sp_coords,
            po_coords
        )
    )

    return coords


def coord_to_sparse_tensor(
    nrows: int, ncols: int, coords: Tensor, device: str, value=1.0, row_slice=None
):
    if row_slice is not None:
        if row_slice.step is not None:
            # just to be sure
            raise ValueError()

        coords = coords[
            (coords[:, 0] >= row_slice.start) & (coords[:, 0] < row_slice.stop), :
        ]
        coords[:, 0] -= row_slice.start
        nrows = row_slice.stop - row_slice.start

    if device == "cpu":
        labels = torch.sparse.FloatTensor(
            coords.long().t(),
            torch.ones([len(coords)], dtype=torch.float, device=device) * value,
            torch.Size([nrows, ncols]),
        )
    else:
        labels = torch.cuda.sparse.FloatTensor(
            coords.long().t(),
            torch.ones([len(coords)], dtype=torch.float, device=device) * value,
            torch.Size([nrows, ncols]),
            device=device,
        )

    return labels
