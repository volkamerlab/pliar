import polars as pl
from .columns import C

num_non_key = (pl.col.num_residues - pl.col.num_relevant).clip(1)
rank_auroc = (
    ((num_non_key - pl.col.isolated_attr_rank - 1).clip(0) / num_non_key)
    .mean()
    .alias("rank_auroc")
)


def precision_at(k: int):
    recalled = pl.col(C.ISOLATED_RANK) <= k
    return recalled.mean().alias(f"precision_at_{k}")
