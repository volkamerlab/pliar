import logging
import shutil
from enum import StrEnum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path as MplPath
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


class COLS(StrEnum):
    ACTIVITY_ID = "activities.activity_id"
    KLIFS_ID = "similar.klifs_structure_id"
    SEQUENCE = "structure.pocket_sequence"
    DUNBRACK = "abreviated_dunbrack_state"
    DFG = "dfg_state"
    DUNBRACK_CONF = "dunbrack_conf"
    DUNBRACK_ACTIVE = "dunbrack_active"
    DUNBRACK_SIMPLIFIED = "dunbrack_simplified"
    REFERENCE_PREDICTION = "reference_pred"
    MASKED_PREDICTION = "masked_pred"
    DELTA = "delta"
    RESIDUE_IMPORTANCE = "residue_importance"
    UNIPROT_ID = "UniprotID"
    ALIGNMENT = "cosine_similarity"
    SMILES = "compound_structures.canonical_smiles"
    ACTIVITY_VALUE = "activities.standard_value"
    RESNR = "RESNR"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

HERE = Path(__file__).parent
DATA_PATH = HERE.parent.parent / Path("data")
FIGURE_PATH = DATA_PATH / Path("figures")
PLIP_PATH = DATA_PATH / "plip"
if not FIGURE_PATH.exists():
    FIGURE_PATH.mkdir(parents=True)


def write_figure(filename, fig=None, dpi=200, format=["svg", "pdf", "png"]):
    if fig is None:
        fig = plt.gcf()
    filename = filename.replace(" ", "_")
    filepath = FIGURE_PATH / filename
    if not isinstance(format, list):
        format = [format]
    for fmt in format:
        if fmt.startswith("."):
            fmt = fmt[1:]
        filepath = filepath.with_suffix(f".{fmt}")
        logger.info(f"Writing figure to {filepath} ...")
        fig.savefig(
            filepath,
            dpi=dpi,
            bbox_inches="tight",
            format=fmt,
        )


TALK_RENAMING = {
    "similar.klifs_structure_id": "KLIFS structure",
    "activities.standard_value": "pIC50",
    "num_interactions": "# PLIP interactions",
    "UniprotID": "Uniprot ID",
}


def talk_friendly_column_names(df):
    subset = {k: v for k, v in TALK_RENAMING.items() if k in df.columns}
    return df.rename(columns=subset)


kinase_regions = dict(
    Hinge=[46, 47, 48],
    DFG=[81, 82, 83],
    SaltBridgeRegion=[17, 24],
    GK=[45],
    X=[80],
    # glycine_rich_loop=[4, 5, 6, 7, 8, 9],
)

residue_number_to_region = {
    res: region for region, residues in kinase_regions.items() for res in residues
}
residue_number_to_region


palette = sns.color_palette("Dark2", len(kinase_regions) + 1)
region_colors = pd.Series(palette, index=list(kinase_regions.keys()) + ["other"])
region_colors["other"] = (0.75, 0.75, 0.75)

residue_colors = []
for resnr in range(1, 86):
    region = residue_number_to_region.get(resnr, "other")
    residue_colors.append(region_colors[region])


def show_residue_palette():
    plt.bar(range(1, 86), [1] * 85, color=residue_colors)


interaction_cm = plt.cm.get_cmap("Set3", 6)
interaction_colors = interaction_cm.colors


def fisher_transform(corr: np.ndarray):
    return (1 / 2) * np.log((1 + corr) / (1 - corr))


def inv_fisher_transform(z: np.ndarray):
    return np.tanh(2 * z)


def fisher_z(r):
    return np.arctanh(np.clip(r, -0.9999, 0.9999))


def inverse_fisher_z(z):
    return np.tanh(z)


def fisher_average(r, n):
    z = fisher_z(r)
    weights = n - 3
    z_avg = np.sum(weights * z) / np.sum(weights)
    return inverse_fisher_z(z_avg)


def fisher_std(r, n):
    weights = n - 3
    se_z = 1 / np.sqrt(np.sum(weights))
    return se_z


def make_fisher_agg(corr_col="r", size_col="n"):
    """
    Returns a function that can be used in groupby().apply(),
    using custom column names for correlation and sample size.
    """

    def fisher_agg(group):
        r = group[corr_col].values
        n = group[size_col].values
        return pd.Series(
            {"fisher_avg_r": fisher_average(r, n), "fisher_se": fisher_std(r, n)}
        )

    return fisher_agg


def get_kinodata3d_source_df() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH / "processed" / "kinodata3d_metadata.csv")


def add_kinase_coloring(ax, draw_legend, fontsize=18):
    # color xticks based on residue number region
    for label in ax.get_xticklabels():
        resnr = int(label.get_text())
        region = residue_number_to_region.get(resnr, "other")
        index = (list(kinase_regions.keys()) + ["other"]).index(region)
        label.set_color(region_colors.iloc[index])

    if draw_legend:
        # add a legend that maps region to color
        legend_elements = [
            plt.Line2D([0], [0], color=region_colors.iloc[i], lw=4, label=region)
            for i, region in enumerate(list(kinase_regions.keys()) + ["other"])
        ]
        ax.legend(handles=legend_elements, title="Residue region", fontsize=fontsize)


def radar_factory(num_vars, frame="circle"):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return MplPath(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=MplPath.unit_regular_polygon(num_vars),
                )
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def radar_plot(
    data=None,
    factor=None,
    y=None,
    hue=None,
    factor_order=None,
    hue_order=None,
    theta=None,
    ax=None,
    estimator="mean",
    palette="tab10",
):
    if data is not None:
        assert isinstance(data, pl.DataFrame)
        data = data.select(pl.col(factor), pl.col(y), pl.col(hue))
    if data is None:
        data = pl.DataFrame(
            {
                "factor": factor,
                "y": y,
                "hue": hue,
            }
        )
        factor = "factor"
        y = "y"
        hue = "hue"
    if factor_order is None:
        factor_order = data.select(pl.col(factor)).unique().to_series().to_list()
    if hue_order is None:
        hue_order = data.select(pl.col(hue)).unique().to_series().to_list()
    data = data.filter(
        pl.col(factor).is_in(factor_order) & pl.col(hue).is_in(hue_order)
    )
    palette = sns.color_palette(palette, n_colors=data.n_unique(pl.col(hue)))
    if theta is None:
        theta = radar_factory(len(factor_order), frame="polygon")
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection="radar"))
    for hue_name, subdata in sorted(
        data.group_by(pl.col(hue)), key=lambda g: hue_order.index(g[0][0])
    ):
        color = palette.pop(0)
        if estimator == "mean":
            means = (
                subdata.group_by(pl.col(factor))
                .agg(pl.col(y).mean())
                .sort(pl.col(factor).map_elements(factor_order.index, return_dtype=int))
                .select(pl.col(y))
                .to_numpy()
                .flatten()
            )
            ax.plot(theta, means, label=hue_name, color=color)
            ax.fill(theta, means, facecolor=color, alpha=0.25)
        elif estimator == "median":
            medians = (
                subdata.group_by(pl.col(factor))
                .agg(pl.col(y).median())
                .sort(pl.col(factor).map_elements(factor_order.index, return_dtype=int))
                .select(pl.col(y))
                .to_numpy()
                .flatten()
            )
            q25 = (
                subdata.group_by(pl.col(factor))
                .agg(pl.col(y).quantile(0.25))
                .sort(pl.col(factor).map_elements(factor_order.index, return_dtype=int))
                .select(pl.col(y))
                .to_numpy()
                .flatten()
            )
            q75 = (
                subdata.group_by(pl.col(factor))
                .agg(pl.col(y).quantile(0.75))
                .sort(pl.col(factor).map_elements(factor_order.index, return_dtype=int))
                .select(pl.col(y))
                .to_numpy()
                .flatten()
            )
            ax.plot(theta, q25, label=hue_name, color=color, linestyle="--")
            ax.plot(theta, medians, label=hue_name, color=color)
            ax.plot(theta, q75, label=hue_name, color=color, linestyle="--")
            ax.fill(theta, q75, facecolor=color, alpha=0.25)
        else:
            raise NotImplementedError

    ax.set_varlabels(factor_order)
    return ax
