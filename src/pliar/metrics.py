import polars as pl
from .columns import C


num_non_key = pl.col(C.NUM_RESIDUES) - pl.col(C.NUM_RELEVANT)
# how many non-key residues are ranked below each key residue?
num_correctly_ranked_pairs = (num_non_key - pl.col(C.ISOLATED_RANK) - 1).clip(0).sum()
# how many non-key residues are there in total?
normalizer = (num_non_key).sum()
rank_auroc = (num_correctly_ranked_pairs / normalizer).alias("rank_auroc")


def precision_at(k: int):
    recalled = pl.col(C.ISOLATED_RANK) <= k
    return recalled.mean().alias(f"precision_at_{k}")
