import logging
from pathlib import Path

import polars as pl

from .columns import C
from .metrics import rank_auroc, precision_at

_REPO = Path(__file__).parent.parent.parent
_PLI_REFERENCE_FILE = _REPO / "data" / "processed" / "pli_reference.csv"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def main(
    clean_prediction_file: Path,
    masked_prediction_file: Path,
    plip_explanation_file: Path = _PLI_REFERENCE_FILE,
    outdir: Path = _REPO,
    prefix: str = "",
):
    logger.info("Loading data...")
    clean_predictions = pl.read_csv(clean_prediction_file)
    masked_predictions = pl.read_csv(masked_prediction_file)
    plip_explanations = pl.read_csv(plip_explanation_file)
    plip_explanations = plip_explanations.with_columns(
        pl.when(pl.col("H-Bond (P-Acc)") > 0)
        .then(pl.lit("H-Bond (PA)"))
        .when(pl.col("H-Bond (P-Don)") > 0)
        .then(pl.lit("H-Bond (PD)"))
        .when(pl.col("Hydroph. Intr.") > 0)
        .then(pl.lit("Hydrophobic"))
        .when(pl.col("Pi-Cation") > 0)
        .then(pl.lit("Pi-Cation"))
        .when(pl.col("Pi Stack.") > 0)
        .then(pl.lit("Pi-Stack"))
        .when(pl.col("Salt Bridge") > 0)
        .then(pl.lit("Salt Bridge"))
        .alias("interaction_type")
    )
    activity_id_subset = plip_explanations[C.ACTIVITY_ID].unique()

    logger.info("Computing prediction deltas...")
    delta = (
        clean_predictions.join(
            masked_predictions,
            on="activity_id",
            suffix="_masked",
        )
        .filter(pl.col(C.ACTIVITY_ID).is_in(activity_id_subset))
        .with_columns((pl.col(C.PRED) - pl.col(f"{C.PRED}_masked")).alias(C.DELTA))
    )
    data = delta.join(
        plip_explanations,
        how="left",
        left_on=[C.ACTIVITY_ID, C.MASKED_RESNR],
        right_on=[C.ACTIVITY_ID, C.RESNR],
    )

    assert data[C.ACTIVITY_ID].value_counts().max()["count"].item() <= 85

    logger.info("Computing ranks...")
    # highest attribution first
    data = data.with_columns(
        # attribution ranks within complexes
        pl.col(C.DELTA)
        .rank("average", descending=True)
        .over(C.ACTIVITY_ID)
        .alias(C.ATTR_RANK),
        # is the residue relevant (has positive importance)
        pl.when(
            pl.col("residue_importance").is_not_null()
            & (pl.col("residue_importance") > 0)
        )
        .then(True)
        .otherwise(False)
        .alias(C.IS_RELEVANT),
    )
    data = data.with_columns(
        # number of relevant residues per complex
        pl.col(C.IS_RELEVANT).sum().over(C.ACTIVITY_ID).alias(C.NUM_RELEVANT),
        # total number of residues per complex
        pl.count().over(C.ACTIVITY_ID).alias(C.NUM_RESIDUES),
    )
    data = data.with_columns(
        # isolated attribution rank (rank only relative to "irrelevant" residues)
        (pl.col(C.ATTR_RANK) - pl.col(C.NUM_RELEVANT) + 1).alias(C.ISOLATED_RANK),
    )
    rank_auroc_data = data.group_by(C.ACTIVITY_ID).agg(rank_auroc)
    precision = data.group_by(C.ACTIVITY_ID).agg(
        precision_at(1), precision_at(3), precision_at(5), precision_at(10)
    )
    rank_auroc_by_interaction = data.group_by("interaction_type").agg(rank_auroc)
    logger.info("Writing results to disk...")
    data.drop_nulls().write_csv(outdir / f"{prefix}attribution_ranking.csv")
    rank_auroc_data.write_csv(outdir / f"{prefix}attribution_ranking_auroc.csv")
    precision.write_csv(outdir / f"{prefix}attribution_ranking_precision.csv")
    rank_auroc_by_interaction.write_csv(
        outdir / f"{prefix}attribution_ranking_auroc_by_interaction.csv"
    )
