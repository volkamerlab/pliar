from enum import StrEnum


class C(StrEnum):
    ACTIVITY_ID = "activity_id"
    MASKED_RESNR = "masked_residue"
    RESNR = "residue_number"
    DELTA = "delta"
    PRED = "predicted_value"
    ATTR_RANK = "attr_rank"
    ISOLATED_RANK = "isolated_attr_rank"
    INTR_TYPE = "interaction_type"
    IS_RELEVANT = "is_relevant"
    NUM_RELEVANT = "num_relevant"
    NUM_RESIDUES = "num_residues"
