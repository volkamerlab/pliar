import logging
from pathlib import Path

import polars as pl

from .columns import C
from .metrics import rank_auroc

_REPO = Path(__file__).parent.parent.parent
_PLI_REFERENCE_FILE = _REPO / "data" / "processed" / "pli_reference.csv"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# debugging
def describe_size(df, col):
    print(col, df.group_by(col).agg(pl.len().alias("size")).select("size").describe())


def main(
    clean_prediction_file: Path,
    masked_prediction_file: Path,
    plip_explanation_file: Path = _PLI_REFERENCE_FILE,
    outdir: Path = _REPO,
    prefix: str = "",
):
    """
    Evaluate the R-AUROC PLI-alignment of a model based on it's predictions over unaltered (clean)
    kinase-ligand complexes and modified inputs, where KLIFS residues were masked one-at-a-time.

    For the schema of the clean and masked prediction tabular data consider the README.md file.

    Args:
        clean_prediction_file (Path): Predictions on clean inputs
        masked_prediction_file (Path): Predictions on masked inputs.
        plip_explanation_file (Path, optional): Source of reference explantions. Defaults to _PLI_REFERENCE_FILE.
        outdir (Path, optional): Where write the results to. Defaults to _REPO.
        prefix (str, optional): Prefix that will be appended to stems of output files. Defaults to "".
    """
    logger.info("Loading data...")
    logger.info(f"PLI reference: {plip_explanation_file.absolute()}")
    plip_explanations = pl.scan_csv(plip_explanation_file)
    logger.info(f"Clean predictions: {clean_prediction_file.absolute().resolve()}")
    clean_predictions = (
        pl.scan_csv(clean_prediction_file)
        .join(plip_explanations.select("activity_id"), on="activity_id", how="semi")
        .collect()
    )
    logger.info(f"Masked predictions: {masked_prediction_file.absolute().resolve()}")
    masked_predictions = (
        pl.scan_csv(masked_prediction_file)
        .join(plip_explanations.select("activity_id"), on="activity_id", how="semi")
        .collect()
    )
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
    ).collect()

    logger.info("Computing prediction deltas...")
    delta = clean_predictions.join(
        masked_predictions, on="activity_id", suffix="_masked", how="inner"
    ).with_columns((pl.col(C.PRED) - pl.col(f"{C.PRED}_masked")).alias(C.DELTA))
    data = delta.join(
        plip_explanations,
        how="left",
        left_on=[C.ACTIVITY_ID, C.MASKED_RESNR],
        right_on=[C.ACTIVITY_ID, C.RESNR],
    )

    assert (
        num_entries := data[C.ACTIVITY_ID].value_counts().max()["count"].item()
    ) <= 85, num_entries

    logger.info("Computing ranks and metrics...")
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
    ).filter(pl.col(C.IS_RELEVANT))
    rank_auroc_data = data.group_by(C.ACTIVITY_ID).agg(rank_auroc)
    rank_auroc_by_interaction = data.group_by("interaction_type").agg(rank_auroc)
    logger.info("Writing results to disk...")
    attribution_ranks = outdir / f"{prefix}attribution_ranking.csv"
    attribution_rank_auroc = outdir / f"{prefix}attribution_ranking_auroc.csv"
    auroc_by_interaction = (
        outdir / f"{prefix}attribution_ranking_auroc_by_interaction.csv"
    )

    logger.info(" ~ attribution ranks: %s" % str(attribution_ranks))
    logger.info(" ~ attribution rank auroc: %s" % str(attribution_rank_auroc))
    logger.info(
        " ~ attribution rank auroc by interaction type: %s" % str(auroc_by_interaction)
    )
    data.drop_nulls().write_csv(attribution_ranks)
    rank_auroc_data.write_csv(attribution_rank_auroc)
    rank_auroc_by_interaction.write_csv(auroc_by_interaction)
