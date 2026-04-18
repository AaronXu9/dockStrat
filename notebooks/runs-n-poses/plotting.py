from collections import defaultdict
from io import BytesIO

import cmap
import matplotlib.pyplot as plt
import numpy as np
try:
    import ost
    if hasattr(ost, "PushVerbosityLevel"):
        ost.PushVerbosityLevel(-1)  # disable OST logging
except Exception:
    pass
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.metrics import accuracy_score, roc_curve

# ost.PushVerbosityLevel(-1)
import warnings

warnings.filterwarnings("ignore")
from matplotlib import gridspec
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


RMSD_THRESHOLD = 2
LDDT_PLI_THRESHOLD = 0.8
SIMILARITY_METRIC = "sucos_shape_pocket_qcov"
SIMILARITY_BINS = [0, 20, 30, 40, 50, 60, 70, 80, 100]
NUM_CCDS_THRESHOLD = 100

PALETTE = list(cmap.Colormap("tol:vibrant").iter_colors())

COLORS = {
    "af3": PALETTE[0],
    "af3_no_template": PALETTE[1],
    "protenix": PALETTE[2],
    "chai": PALETTE[3],
    "boltz1x": PALETTE[5],
    "boltz": PALETTE[4],
    "boltz2": list(cmap.Colormap("tol:muted").iter_colors())[3],
    "rfaa": list(cmap.Colormap("tol:muted").iter_colors())[0],
    "best": PALETTE[-1],
}
SHAPES = {
    "af3": "o",
    "protenix": "s",
    "chai": "D",
    "boltz1x": "X",
    "boltz": "X",
    "boltz2": "v",
    "best": "*",
    "af3_no_template": "P",
    "rfaa": "H",
}
METHODS = [
    "af3",
    "af3_no_template",
    "protenix",
    "chai",
    "boltz",
    "boltz1x",
    "rfaa",
    "boltz2",
]
COMMON_SUBSET_METHODS = ["af3", "protenix", "chai", "boltz"]
NAME_MAPPING = {
    "af3": "AlphaFold3",
    "protenix": "Protenix",
    "chai": "Chai-1",
    "boltz1x": "Boltz-1x",
    "boltz": "Boltz-1",
    "boltz2": "Boltz-2",
    "rfaa": "RosettaFold-all-atom",
    "af3_no_template": "AlphaFold3 (no template)",
}
METRIC_NAMES_1 = {
    "protein_fident_weighted_sum": "Protein\nsequence\nidentity",
    "pocket_qcov": "Pocket\ncoverage",
}

METRIC_NAMES_2 = {
    "topological_tanimoto": "Topological\nfingerprint\nsimilarity",
    "morgan_tanimoto": "Morgan\nfingerprint\nsimilarity",
    "sucos_shape": "Ligand\nSuCOS",
}


def plot_success_by_similarity_pb(
    results_df,
    lddt_pli_column,
    rmsd_column,
    pb_success_column,
    color="black",
    similarity_metric="sucos_shape_pocket_qcov",
    rmsd_threshold=RMSD_THRESHOLD,
    lddt_pli_threshold=LDDT_PLI_THRESHOLD,
    similarity_bins=SIMILARITY_BINS,
    ax=None,
    label="",
    x_offset=0,
    bar_width=0.15,
    fontsize=8,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    bin_data_success = []
    bin_data_pb = []
    bin_sizes = []

    for i in range(len(similarity_bins) - 1):
        mask = (results_df[similarity_metric] >= similarity_bins[i]) & (
            results_df[similarity_metric] < similarity_bins[i + 1]
        )
        bin_sizes.append(len(results_df[mask]))

        mask_success = mask & results_df[lddt_pli_column].notna()
        success_vals = (results_df[mask_success][rmsd_column] < rmsd_threshold) & (
            results_df[mask_success][lddt_pli_column] > lddt_pli_threshold
        )
        bin_data_success.append(np.mean(success_vals.astype(float)) * 100)

        pb_success_vals = (
            (results_df[mask_success][rmsd_column] < rmsd_threshold)
            & (results_df[mask_success][lddt_pli_column] > lddt_pli_threshold)
            & (results_df[mask_success][pb_success_column] == 1)
        )
        bin_data_pb.append(np.mean(pb_success_vals.astype(float)) * 100)

    x = np.arange(len(bin_sizes)) + x_offset

    bin_data_non_pb = [
        success - pb for success, pb in zip(bin_data_success, bin_data_pb)
    ]

    ax.bar(
        x,
        bin_data_pb,
        width=bar_width,
        color=color,
        edgecolor=color,
        zorder=3,
        label=label,
    )
    ax.bar(
        x,
        bin_data_non_pb,
        width=bar_width,
        bottom=bin_data_pb,
        color="white",
        hatch="///",
        edgecolor=color,
        zorder=3,
        label=label,
    )

    for i, (pb_val, total_val) in enumerate(zip(bin_data_pb, bin_data_success)):
        if pb_val > 0:
            ax.text(
                x[i],
                pb_val - 2,
                f"{pb_val:.0f}%",
                ha="center",
                va="center",
                fontsize=fontsize,
                color="white",
            )
        if total_val > 0:
            ax.text(
                x[i],
                total_val + 1,
                f"{total_val:.0f}%",
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )

    ax.set_xlabel("Similarity to the training set", fontsize=12, fontweight="bold")
    ax.set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
    ax.set_xticks(x - x_offset)
    ax.set_xticklabels(
        [
            f"{similarity_bins[i]}-{similarity_bins[i + 1]}\n(n={bin_sizes[i]:,})"
            for i in range(len(similarity_bins) - 1)
        ],
        rotation=45,
        fontsize=12,
    )
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax, bin_sizes


def make_one_plot_pb(
    df,
    ax,
    similarity_metric=SIMILARITY_METRIC,
    similarity_bins=SIMILARITY_BINS,
    lddt_pli_threshold=LDDT_PLI_THRESHOLD,
    rmsd_threshold=RMSD_THRESHOLD,
    methods=COMMON_SUBSET_METHODS,
    bar_width=0.23,
    fontsize=8,
):
    df = df[df["sucos_shape"].notna() & df["ligand_is_proper"]].reset_index(drop=True)

    total_width = len(methods) * bar_width
    start_offset = -total_width / 2 + bar_width / 2

    bin_sizes = None
    for i, method in enumerate(methods):
        x_offset = start_offset + i * bar_width
        ax, sizes = plot_success_by_similarity_pb(
            df,
            ax=ax,
            lddt_pli_column=f"lddt_pli_{method}",
            rmsd_column=f"rmsd_{method}",
            pb_success_column=f"pb_success_{method}",
            label=NAME_MAPPING[method],
            color=COLORS[method],
            rmsd_threshold=rmsd_threshold,
            lddt_pli_threshold=lddt_pli_threshold,
            similarity_metric=similarity_metric,
            similarity_bins=similarity_bins,
            x_offset=x_offset,
            bar_width=bar_width,
            fontsize=fontsize,
        )
        if bin_sizes is None:
            bin_sizes = sizes

    solid_patch = Rectangle((0, 0), 1, 1, facecolor="gray", edgecolor="black")
    hatched_patch = Rectangle(
        (0, 0), 1, 1, facecolor="white", hatch="///", edgecolor="black"
    )
    all_handles = [
        solid_patch,
        hatched_patch,
    ]
    title = f"RMSD ≤ {rmsd_threshold}Å"
    if lddt_pli_threshold > 0:
        title += f" & LDDT-PLI ≥ {lddt_pli_threshold}"
    all_labels = [
        f"{title} & PB-Valid",
        f"{title} only",
    ]

    ax.legend(all_handles, all_labels, loc="center left", frameon=False)
    ax.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.set_xlim(
        -total_width / 2 - bar_width / 2,
        len(similarity_bins) - 2 + total_width / 2 + bar_width / 2,
    )


def plot_success_by_similarity(
    results_df,
    lddt_pli_column,
    rmsd_column,
    color="black",
    shape="o",
    similarity_metric="sucos_shape_pocket_qcov",
    rmsd_threshold=RMSD_THRESHOLD,
    lddt_pli_threshold=LDDT_PLI_THRESHOLD,
    similarity_bins=SIMILARITY_BINS,
    ax=None,
    label="",
    bootstrap=True,
    linestyle="-",
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    bin_data = []
    bin_sizes = []
    bin_cis = []
    for i in range(len(similarity_bins) - 1):
        mask = (results_df[similarity_metric] >= similarity_bins[i]) & (
            results_df[similarity_metric] < similarity_bins[i + 1]
        )
        bin_sizes.append(len(results_df[mask]))

        mask_success = mask & results_df[lddt_pli_column].notna()
        success_vals = (results_df[mask_success][rmsd_column] < rmsd_threshold) & (
            results_df[mask_success][lddt_pli_column] > lddt_pli_threshold
        )
        bin_data.append(np.mean(success_vals.astype(float)) * 100)

        n_bootstrap = 1000
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(
                success_vals.astype(float), size=len(success_vals), replace=True
            )
            bootstrap_means.append(np.mean(bootstrap_sample) * 100)
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        bin_cis.append((ci_lower, ci_upper))

    x = np.arange(len(bin_sizes))
    bin_cis = np.array(bin_cis)

    ax.plot(
        x,
        bin_data,
        marker=shape,
        markersize=10,
        linestyle=linestyle,
        linewidth=2,
        color=color,
        label=label,
    )
    if bootstrap:
        ax.fill_between(
            x,
            bin_cis[:, 0],
            bin_cis[:, 1],
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel("Similarity to the training set", fontsize=12, fontweight="bold")
    ax.set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{similarity_bins[i]}-{similarity_bins[i + 1]}\n(n={bin_sizes[i]:,})"
            for i in range(len(similarity_bins) - 1)
        ],
        rotation=45,
        fontsize=12,
    )
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=12)

    ax.grid(True, linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def DrawMolsZoomed(
    mols, molsPerRow=3, subImgSize=(200, 200), legends=None, row_labels=None
):
    nRows = len(mols) // molsPerRow
    if len(mols) % molsPerRow:
        nRows += 1
    label_height = 40 if row_labels else 0
    fullSize = (
        molsPerRow * subImgSize[0],
        nRows * subImgSize[1] + nRows * label_height,
    )
    full_image = Image.new("RGBA", fullSize, "white")

    current_row = -1

    for ii, mol in enumerate(mols):
        if mol.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol)
        column = ii % molsPerRow
        row = ii // molsPerRow

        if row_labels and row != current_row:
            current_row = row
            label_img = Image.new(
                "RGBA", (molsPerRow * subImgSize[0], label_height), "white"
            )
            draw = ImageDraw.Draw(label_img)
            font = ImageFont.load_default(size=20)
            draw.text(
                (10, 5),
                row_labels[row],
                fill=(0, 0, 0),
                font=font,
            )
            full_image.paste(label_img, box=(0, row * (subImgSize[1] + label_height)))

        offset = (
            column * subImgSize[0],
            row * (subImgSize[1] + label_height) + label_height,
        )
        d2d = rdMolDraw2D.MolDraw2DCairo(subImgSize[0], subImgSize[1])
        legend = legends[ii] if legends is not None else None
        d2d.DrawMolecule(mol, legend=legend)
        d2d.FinishDrawing()
        sub = Image.open(BytesIO(d2d.GetDrawingText()))
        full_image.paste(sub, box=offset)
    return full_image


def make_one_plot(
    df,
    ax,
    title,
    similarity_metric=SIMILARITY_METRIC,
    similarity_bins=SIMILARITY_BINS,
    lddt_pli_threshold=LDDT_PLI_THRESHOLD,
    rmsd_threshold=RMSD_THRESHOLD,
    methods=COMMON_SUBSET_METHODS,
    legend_loc="lower right",
    ylabel="Success Rate (%)",
    xlabel="Similarity to the training set",
):
    df = df[df["sucos_shape"].notna() & df["ligand_is_proper"]].reset_index(drop=True)
    for method in methods:
        plot_success_by_similarity(
            df,
            ax=ax,
            lddt_pli_column=f"lddt_pli_{method}",
            rmsd_column=f"rmsd_{method}",
            label=NAME_MAPPING[method],
            shape=SHAPES[method],
            color=COLORS[method],
            rmsd_threshold=rmsd_threshold,
            lddt_pli_threshold=lddt_pli_threshold,
            similarity_metric=similarity_metric,
            similarity_bins=similarity_bins,
        )

    plot_success_by_similarity(
        df,
        ax=ax,
        lddt_pli_column="lddt_pli_max",
        rmsd_column="rmsd_min",
        label="Best",
        shape="*",
        color=COLORS["best"],
        rmsd_threshold=rmsd_threshold,
        lddt_pli_threshold=lddt_pli_threshold,
        similarity_metric=similarity_metric,
        similarity_bins=similarity_bins,
    )
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.text(0.015, 0.99, title, transform=ax.transAxes, fontsize=14, fontweight="bold")
    if legend_loc is not None:
        ax.legend(loc=legend_loc, frameon=False)


def make_distribution_plot(
    df,
    ax,
    accuracy_metric,
    threshold,
    similarity_metric=SIMILARITY_METRIC,
    similarity_bins=SIMILARITY_BINS,
    methods=COMMON_SUBSET_METHODS,
    log=False,
    scatter_spacing=0.2,
    violin_color="#a2a4f2",
    add_xlabel=True,
):
    plot_data = []
    df = df[df["ligand_is_proper"] & df["sucos_shape"].notna()].reset_index(drop=True)
    for i in range(len(similarity_bins) - 1):
        mask = (df[similarity_metric] >= similarity_bins[i]) & (
            df[similarity_metric] < similarity_bins[i + 1]
        )
        bin_size = mask.sum()

        bin_data = {
            "bin_range": f"{similarity_bins[i]}-{similarity_bins[i + 1]}",
            "n_samples": bin_size,
            "position": i,
        }

        all_values = []
        for method in methods:
            values = df[mask][f"{accuracy_metric}_{method}"].dropna()
            if log:
                values = np.log10(values)
            bin_data[method] = values
            all_values.extend(values)
        bin_data["mean"] = np.mean(all_values)
        bin_data["median"] = np.median(all_values)
        plot_data.append(bin_data)

    for i, bin_data in enumerate(plot_data):
        pos = i + 1
        parts = ax.violinplot(
            np.concatenate([bin_data[method] for method in methods]),
            positions=[pos],
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(violin_color)
            pc.set_alpha(0.3)
            pc.set_edgecolor(violin_color)

        method_offsets = {
            method: pos + (j - len(methods) / 2 + 0.5) * scatter_spacing
            for j, method in enumerate(methods)
        }

        for method in methods:
            x_jitter = np.random.normal(0, 0.02, size=len(bin_data[method]))
            ax.scatter(
                [method_offsets[method] + xj for xj in x_jitter],
                bin_data[method],
                color=COLORS[method],
                alpha=0.4,
                s=5,
                marker=SHAPES[method],
            )
        ax.plot(
            pos,
            bin_data["median"],
            "s",
            color="white",
            markersize=10,
            label="Median" if i == 0 else "",
            markeredgecolor="black",
            markeredgewidth=1,
        )
        ax.plot(
            pos,
            bin_data["mean"],
            "D",
            color="white",
            markersize=8,
            label="Mean" if i == 0 else "",
            markeredgecolor="black",
            markeredgewidth=1,
        )
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    if add_xlabel:
        ax.set_xlabel("Similarity to the training set", fontsize=12, fontweight="bold")
        ax.set_xticks([i for i in range(len(plot_data) + 1)])
        ax.set_xticklabels(
            [""] + [f"{d['bin_range']}\n(n={d['n_samples']})" for d in plot_data],
            rotation=0,
        )
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                color="white",
                marker="s",
                label="Median",
                markersize=8,
                linestyle="none",
                markeredgecolor="black",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="D",
                color="white",
                label="Mean",
                markersize=8,
                linestyle="none",
                markeredgecolor="black",
            ),
        ]
        ax.legend(handles=legend_elements, loc="lower left", frameon=False, ncols=2)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
    ax.set_xlim(-0.025, len(plot_data) + 0.5)
    ax.set_ylabel(
        f"{accuracy_metric.upper().replace('_', '-')}", fontsize=12, fontweight="bold"
    )

    label = threshold if accuracy_metric == "lddt_pli" else f"{threshold}Å"
    if log:
        threshold = np.log10(threshold)
    ax.axhline(
        y=threshold, color="black", linestyle="--", alpha=0.5, linewidth=2, zorder=-1
    )
    ax.text(
        0.3,
        threshold + 0.03,
        label,
        verticalalignment="center",
        horizontalalignment="right",
        fontsize=12,
        fontweight="bold",
        color="black",
    )
    if log:
        ax.set_yticklabels([f"{10**x:.2f}" for x in ax.get_yticks()])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_main_figure(
    df,
    filename,
    similarity_metric=SIMILARITY_METRIC,
    similarity_bins=SIMILARITY_BINS,
    log=False,
    lddt_pli_threshold=LDDT_PLI_THRESHOLD,
    rmsd_threshold=RMSD_THRESHOLD,
    methods=COMMON_SUBSET_METHODS,
    bar_width=0.2,
    figsize=(14, 18),
    scatter_spacing=0.2,
    legend_loc="upper left",
    pb_fontsize=9.5,
):
    fig = plt.figure(figsize=figsize)
    labels = ["A)", "B)", "C)", "D)", "E)"]
    gs = plt.GridSpec(
        7,
        2,
        hspace=0,
        wspace=0.15,
        height_ratios=[1.2, 0.35, 0.6, 0.05, 0.6, 0.2, 1],
    )

    ax_success_both = fig.add_subplot(gs[0, 0])
    ax_success_rmsd = fig.add_subplot(gs[0, 1])
    ax_filler_1 = fig.add_subplot(gs[1, 0])
    ax_filler_1.set_visible(False)
    ax_lddt_pli = fig.add_subplot(gs[2, :])
    ax_filler_2 = fig.add_subplot(gs[3, 0])
    ax_filler_2.set_visible(False)
    ax_rmsd = fig.add_subplot(gs[4, :])
    ax_filler_3 = fig.add_subplot(gs[5, 0])
    ax_filler_3.set_visible(False)
    ax_posebusters = fig.add_subplot(gs[6, :])

    make_one_plot(
        df,
        ax_success_both,
        title=labels[0],
        similarity_metric=similarity_metric,
        similarity_bins=similarity_bins,
        lddt_pli_threshold=lddt_pli_threshold,
        rmsd_threshold=rmsd_threshold,
        methods=methods,
        legend_loc=legend_loc,
    )
    make_one_plot(
        df,
        ax_success_rmsd,
        title=labels[1],
        similarity_metric=similarity_metric,
        similarity_bins=similarity_bins,
        lddt_pli_threshold=0,
        rmsd_threshold=rmsd_threshold,
        methods=methods,
        legend_loc=None,
        ylabel=f"RMSD < {rmsd_threshold}Å Success Rate (%)",
    )
    distribution_axes = [ax_lddt_pli, ax_rmsd]
    accuracy_metrics = ["lddt_pli", "rmsd"]
    thresholds = [lddt_pli_threshold, rmsd_threshold]
    logs = [False, True]
    add_xlabels = [False, True]
    for i, (ax, accuracy_metric, threshold, log, add_xlabel) in enumerate(
        zip(distribution_axes, accuracy_metrics, thresholds, logs, add_xlabels)
    ):
        ax.text(
            0.005,
            0.9,
            labels[2 + i],
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
        )
        make_distribution_plot(
            df,
            ax,
            accuracy_metric,
            threshold,
            similarity_metric=similarity_metric,
            similarity_bins=similarity_bins,
            methods=methods,
            log=log,
            scatter_spacing=scatter_spacing,
            add_xlabel=add_xlabel,
        )

    make_one_plot_pb(
        df,
        ax_posebusters,
        methods=methods,
        similarity_bins=similarity_bins,
        similarity_metric=similarity_metric,
        lddt_pli_threshold=lddt_pli_threshold,
        rmsd_threshold=rmsd_threshold,
        bar_width=bar_width,
        fontsize=pb_fontsize,
    )
    ax_posebusters.text(
        0.005,
        0.95,
        labels[4],
        transform=ax_posebusters.transAxes,
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(filename, dpi=300, bbox_inches="tight")


def make_one_plot_binned(
    df,
    ax,
    title,
    column,
    bins,
    column_name,
    fontsize=12,
    methods=COMMON_SUBSET_METHODS,
    legend_loc="lower right",
):
    bin_data = defaultdict(list)
    bin_sizes = []
    df = df[df["ligand_is_proper"] & df["sucos_shape"].notna()].reset_index(drop=True)
    for i in range(len(bins) - 1):
        mask = (df[column] >= bins[i]) & (df[column] < bins[i + 1])
        bin_sizes.append(len(df[mask]))
        for method in methods:
            mask_inner = mask & (df[f"rmsd_{method}"].notna())
            b = (df[mask_inner][f"rmsd_{method}"] < RMSD_THRESHOLD) & (
                df[mask_inner][f"lddt_pli_{method}"] > LDDT_PLI_THRESHOLD
            )
            bin_data[method].append(b.mean() * 100)
        mask_inner = mask & (df["rmsd_min"].notna())
        b = (df[mask_inner]["rmsd_min"] < RMSD_THRESHOLD) & (
            df[mask_inner]["lddt_pli_max"] > LDDT_PLI_THRESHOLD
        )
        bin_data["best"].append(b.mean() * 100)

    x = np.arange(len(bin_sizes))
    n_bootstrap = 1000
    bootstrap_data = defaultdict(list)

    for method in methods:
        for i in range(len(bins) - 1):
            mask = (df[column] >= bins[i]) & (df[column] < bins[i + 1])
            mask_inner = mask & (df[f"rmsd_{method}"].notna())
            data = df[mask_inner]
            boot_means = []

            for _ in range(n_bootstrap):
                boot_sample = data.sample(n=len(data), replace=True)
                b = (boot_sample[f"rmsd_{method}"] < RMSD_THRESHOLD) & (
                    boot_sample[f"lddt_pli_{method}"] > LDDT_PLI_THRESHOLD
                )
                boot_means.append(b.mean() * 100)

            ci_lower = np.percentile(boot_means, 2.5)
            ci_upper = np.percentile(boot_means, 97.5)
            bootstrap_data[method].append((ci_lower, ci_upper))
    for i in range(len(bins) - 1):
        mask = (df[column] >= bins[i]) & (df[column] < bins[i + 1])
        mask_inner = mask & (df["rmsd_min"].notna())
        data = df[mask_inner]
        boot_means = []

        for _ in range(n_bootstrap):
            boot_sample = data.sample(n=len(data), replace=True)
            b = (boot_sample["rmsd_min"] < RMSD_THRESHOLD) & (
                boot_sample["lddt_pli_max"] > LDDT_PLI_THRESHOLD
            )
            boot_means.append(b.mean() * 100)

        ci_lower = np.percentile(boot_means, 2.5)
        ci_upper = np.percentile(boot_means, 97.5)
        bootstrap_data["best"].append((ci_lower, ci_upper))

    for method in methods:
        ci_lower = [ci[0] for ci in bootstrap_data[method]]
        ci_upper = [ci[1] for ci in bootstrap_data[method]]

        ax.fill_between(x, ci_lower, ci_upper, alpha=0.2, color=COLORS[method])
        ax.plot(
            x,
            bin_data[method],
            marker=SHAPES[method],
            markersize=10,
            linestyle="-",
            linewidth=2,
            label=NAME_MAPPING[method],
            color=COLORS[method],
        )
    ci_lower = [ci[0] for ci in bootstrap_data["best"]]
    ci_upper = [ci[1] for ci in bootstrap_data["best"]]

    ax.fill_between(x, ci_lower, ci_upper, alpha=0.2, color=COLORS["best"])
    ax.plot(
        x,
        bin_data["best"],
        marker=SHAPES["best"],
        markersize=10,
        linestyle="-",
        linewidth=2,
        label="Best",
        color=COLORS["best"],
    )

    ax.set_xlabel(column_name, fontsize=12, fontweight="bold")
    ax.set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{bins[i]}-{bins[i + 1]}\n(n={bin_sizes[i]:,})"
            for i in range(len(bins) - 1)
        ],
        rotation=45,
        fontsize=fontsize,
    )
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=12)

    ax.grid(True, linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.015, 0.99, title, transform=ax.transAxes, fontsize=14, fontweight="bold")
    if legend_loc is not None:
        ax.legend(loc=legend_loc, frameon=False)


def make_figure_prevalent_distinct_clustered(
    df,
    cluster_df,
    filename,
    similarity_metric=SIMILARITY_METRIC,
    similarity_bins=SIMILARITY_BINS,
    lddt_pli_threshold=LDDT_PLI_THRESHOLD,
    rmsd_threshold=RMSD_THRESHOLD,
    methods=COMMON_SUBSET_METHODS,
    num_ccds_threshold=NUM_CCDS_THRESHOLD,
    figsize=(20, 6),
):
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(
        1,
        3,
        hspace=0,
        wspace=0.15,
        width_ratios=[1, 1, 1],
    )

    ax_bar_plot = fig.add_subplot(gs[0, 0])
    ax_uncommon = fig.add_subplot(gs[0, 1])
    ax_clustered = fig.add_subplot(gs[0, 2])
    make_one_plot_binned(
        df,
        ax_bar_plot,
        "A) No. of training ligands",
        "num_training_systems_with_similar_ccds",
        [0, 2, 10, 100, 500, 2500, 60000],
        "No. of training systems with similar ligands",
        fontsize=10,
    )

    make_one_plot(
        df[df["num_training_systems_with_similar_ccds"] < num_ccds_threshold],
        ax_uncommon,
        title="B) With distinct ligands",
        similarity_metric=similarity_metric,
        similarity_bins=similarity_bins,
        lddt_pli_threshold=lddt_pli_threshold,
        rmsd_threshold=rmsd_threshold,
        methods=methods,
        legend_loc=None,
    )
    make_one_plot(
        cluster_df[
            cluster_df["num_training_systems_with_similar_ccds"] < num_ccds_threshold
        ],
        ax_clustered,
        title="C) Clustered & distinct",
        similarity_metric=similarity_metric,
        similarity_bins=similarity_bins,
        lddt_pli_threshold=lddt_pli_threshold,
        rmsd_threshold=rmsd_threshold,
        methods=methods,
        legend_loc=None,
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")


def distribution_plots(
    ax4,
    df,
    score,
    score_name,
    similarity_metric,
    similarity_bins,
    methods,
    cluster_representatives,
    log=False,
    violin_color="#a2a4f2",
):
    plot_data = []
    for i in range(len(similarity_bins) - 1):
        mask = (df[similarity_metric] >= similarity_bins[i]) & (
            df[similarity_metric] <= similarity_bins[i + 1]
        )
        bin_size = mask.sum()

        bin_data = {
            "bin_range": f"{similarity_bins[i]}-{similarity_bins[i + 1]}",
            "n_samples": bin_size,
            "position": i,
        }

        all_values = []
        for method in methods:
            values = df[mask][f"{score}_{method}"].dropna()
            if log:
                values = np.log10(values)
            bin_data[method] = values
            all_values.extend(values)
        bin_data["mean"] = np.mean(all_values)
        bin_data["median"] = np.median(all_values)
        plot_data.append(bin_data)

    for i, bin_data in enumerate(plot_data):
        pos = i
        parts = ax4.violinplot(
            np.concatenate([bin_data[method] for method in methods]),
            positions=[pos],
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(violin_color)
            pc.set_alpha(0.3)
            pc.set_edgecolor(violin_color)
        method_offsets = {
            method: pos + (j - len(methods) / 2 + 0.5) * 0.2
            for j, method in enumerate(methods)
        }
        for method in methods:
            x_jitter = np.random.normal(0, 0.02, size=len(bin_data[method]))

            ax4.scatter(
                [method_offsets[method] + xj for xj in x_jitter],
                bin_data[method],
                color=COLORS[method],
                alpha=0.4,
                s=5,
                marker=SHAPES[method],
            )
        ax4.plot(
            pos,
            bin_data["median"],
            "s",
            color="white",
            markersize=10,
            label="Median" if i == 0 else "",
            markeredgecolor="black",
            markeredgewidth=1,
        )
        ax4.plot(
            pos,
            bin_data["mean"],
            "D",
            color="white",
            markersize=8,
            label="Mean" if i == 0 else "",
            markeredgecolor="black",
            markeredgewidth=1,
        )
    ax4.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax4.set_xlabel("Similarity to the training set", fontsize=12, fontweight="bold")
    ax4.set_ylabel(
        f"{score_name}",
        fontsize=12,
        fontweight="bold",
    )
    ax4.set_xticks([i for i in range(len(plot_data))])
    ax4.set_xticklabels(
        [f"{d['bin_range']}\n(n={d['n_samples']})" for d in plot_data],
        rotation=0,
    )


def other_metrics(
    df,
    cluster_df,
    filename,
    metric_names_1=METRIC_NAMES_1,
    metric_names_2=METRIC_NAMES_2,
    metric_1_threshold={"protein_fident_weighted_sum": 40},
    metric_2_threshold={"morgan_tanimoto": 85},
    similarity_bins=SIMILARITY_BINS,
    similarity_metric=SIMILARITY_METRIC,
    methods=COMMON_SUBSET_METHODS,
    violin_color="#a2a4f2",
):
    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(
        3, 2, height_ratios=[0.7, 0.5, 0.5], width_ratios=[1, 1], hspace=0.3, wspace=0.1
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    ax4 = fig.add_subplot(gs[2, :])

    ax1.text(0.005, 0.91, "A", transform=ax1.transAxes, fontsize=14, fontweight="bold")
    ax2.text(0.005, 0.91, "B", transform=ax2.transAxes, fontsize=14, fontweight="bold")
    ax3.text(0.005, 0.91, "E", transform=ax3.transAxes, fontsize=14, fontweight="bold")
    ax4.text(0.005, 0.91, "F", transform=ax4.transAxes, fontsize=14, fontweight="bold")
    for ax in [ax1, ax2, ax3, ax4]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    df = df[df["ligand_is_proper"] & df["sucos_shape"].notna()].reset_index(drop=True)
    colors = list(cmap.Colormap("tol:bright").iter_colors())
    plot_data = []
    log = False
    cluster_representatives = set(cluster_df["group_key"])
    for i in range(len(similarity_bins) - 1):
        mask = (df[similarity_metric] >= similarity_bins[i]) & (
            df[similarity_metric] <= similarity_bins[i + 1]
        )
        bin_size = mask.sum()

        bin_data = {
            "bin_range": f"{similarity_bins[i]}-{similarity_bins[i + 1]}",
            "n_samples": bin_size,
            "position": i,
        }

        for metric in list(metric_names_1.keys()) + list(metric_names_2.keys()):
            bin_data[metric] = df[mask][metric].dropna()
            bin_data[f"{metric}_rest"] = df[
                mask & (df["group_key"].isin(cluster_representatives))
            ][metric].dropna()
            bin_data[f"{metric}_representative"] = df[
                mask & (df["group_key"].isin(cluster_representatives))
            ][metric].dropna()
            bin_data[f"{metric}_mean"] = np.mean(bin_data[metric])
            bin_data[f"{metric}_median"] = np.median(bin_data[metric])

        plot_data.append(bin_data)

    space_between_metrics = 0.3
    space_between_bins = 0.5
    xtick_positions = []

    for i, bin_data in enumerate(plot_data):
        pos = i
        for j, metric in enumerate(metric_names_1):
            offset = j * space_between_metrics + pos * space_between_bins
            parts = ax1.violinplot(
                [bin_data[metric]],
                positions=[pos + offset],
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(colors[j])
                pc.set_alpha(0.3)
                pc.set_edgecolor(colors[j])
            x_jitter = np.random.normal(0, 0.05, size=len(bin_data[f"{metric}_rest"]))
            ax1.scatter(
                [pos + offset + xj for xj in x_jitter],
                bin_data[f"{metric}_rest"],
                color=colors[j],
                alpha=0.4,
                s=5,
                marker="o",
            )
            x_jitter = np.random.normal(
                0, 0.05, size=len(bin_data[f"{metric}_representative"])
            )
            ax1.scatter(
                [pos + offset + xj for xj in x_jitter],
                bin_data[f"{metric}_representative"],
                color=colors[j],
                alpha=1,
                s=10,
                marker="o",
                edgecolor="#7c7c7d",
                linewidth=0.5,
            )
            ax1.plot(
                pos + offset,
                bin_data[f"{metric}_median"],
                "s",
                color=colors[j],
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=1,
            )
            ax1.plot(
                pos + offset,
                bin_data[f"{metric}_mean"],
                "D",
                color=colors[j],
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=1,
            )
            if metric in metric_1_threshold:
                ax1.axhline(
                    metric_1_threshold[metric],
                    color=colors[j],
                    linestyle="--",
                    linewidth=1.5,
                )
        xtick_positions.append(
            pos + offset - space_between_metrics * len(metric_names_1) / 2
        )

    xtick_positions_2 = []
    for i, bin_data in enumerate(plot_data):
        pos = i
        for j, metric in enumerate(metric_names_2):
            offset = j * space_between_metrics + pos * space_between_bins
            parts = ax2.violinplot(
                [bin_data[metric]],
                positions=[pos + offset],
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(colors[j + len(metric_names_1)])
                pc.set_alpha(0.3)
                pc.set_edgecolor(colors[j + len(metric_names_1)])
            x_jitter = np.random.normal(0, 0.05, size=len(bin_data[f"{metric}_rest"]))
            ax2.scatter(
                [pos + offset + xj for xj in x_jitter],
                bin_data[f"{metric}_rest"],
                color=colors[j + len(metric_names_1)],
                alpha=0.4,
                s=5,
                marker="o",
            )
            x_jitter = np.random.normal(
                0, 0.05, size=len(bin_data[f"{metric}_representative"])
            )
            ax2.scatter(
                [pos + offset + xj for xj in x_jitter],
                bin_data[f"{metric}_representative"],
                color=colors[j + len(metric_names_1)],
                alpha=1,
                s=10,
                marker="o",
                edgecolor="#7c7c7d",
                linewidth=0.5,
            )
            ax2.plot(
                pos + offset,
                bin_data[f"{metric}_median"],
                "s",
                color=colors[j + len(metric_names_1)],
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=1,
            )
            ax2.plot(
                pos + offset,
                bin_data[f"{metric}_mean"],
                "D",
                color=colors[j + len(metric_names_1)],
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=1,
            )
            if metric in metric_2_threshold:
                ax2.axhline(
                    metric_2_threshold[metric],
                    color=colors[j + len(metric_names_1)],
                    linestyle="--",
                    linewidth=1,
                )
        xtick_positions_2.append(
            pos + offset - space_between_metrics * len(metric_names_2) / 2
        )

    ax1.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Similarity to the training set", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Metric Value", fontsize=12, fontweight="bold")
    ax1.set_xticks(xtick_positions)
    ax1.set_xticklabels(
        [f"{d['bin_range']}\n(n={d['n_samples']})" for d in plot_data],
        rotation=0,
        ha="center",
    )

    ax2.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Similarity to the training set", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Metric Value", fontsize=12, fontweight="bold")
    ax2.set_xticks(xtick_positions_2)
    ax2.set_xticklabels(
        [f"{d['bin_range']}\n(n={d['n_samples']})" for d in plot_data],
        rotation=0,
        ha="center",
    )

    metric_legend_elements_both = []
    j = 0
    for metric_names in [metric_names_1, metric_names_2]:
        metric_legend_elements = []
        for metric in metric_names:
            metric_legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=colors[j],
                    label=metric_names[metric],
                    marker="o",
                    markersize=8,
                    linestyle="none",
                )
            )
            j += 1
        metric_legend_elements_both.append(metric_legend_elements)
    stat_legend_elements = [
        plt.Line2D(
            [0],
            [0],
            color=COLORS["best"],
            marker="s",
            label="Median",
            markersize=8,
            linestyle="none",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color=COLORS["best"],
            label="Mean",
            markersize=8,
            linestyle="none",
        ),
    ]

    ax1.legend(
        handles=metric_legend_elements_both[0] + stat_legend_elements,
        loc="lower right",
        frameon=False,
        ncol=len(metric_legend_elements_both[0] + stat_legend_elements),
    )
    ax2.legend(
        handles=metric_legend_elements_both[1],
        loc="lower right",
        frameon=False,
        ncol=len(metric_legend_elements_both[1]),
    )
    ax1.set_xlim(-1.1, max(xtick_positions) + 1)
    ax2.set_xlim(-1.1, max(xtick_positions_2) + 1)
    ax1.set_ylim(-15, 105)
    ax2.set_ylim(-15, 105)

    distribution_plots(
        ax3,
        df,
        "pred_pocket_f1",
        "Pocket F1 score",
        similarity_metric,
        similarity_bins,
        methods,
        cluster_representatives,
        log,
        violin_color,
    )
    distribution_plots(
        ax4,
        df,
        "lddt_lp",
        "LDDT-LP",
        similarity_metric,
        similarity_bins,
        methods,
        cluster_representatives,
        log,
        violin_color,
    )

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            color=COLORS[method],
            label=NAME_MAPPING[method],
            marker=SHAPES[method],
            markersize=8,
            linestyle="none",
        )
        for method in methods
    ]
    legend_elements.extend([
        plt.Line2D(
            [0],
            [0],
            color="white",
            marker="s",
            label="Median",
            markersize=8,
            linestyle="none",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="white",
            label="Mean",
            markersize=8,
            linestyle="none",
        ),
    ])
    ax4.legend(handles=legend_elements, loc="lower left", frameon=False)
    if log:
        ax4.set_yticklabels([f"{10**x:.2f}" for x in ax4.get_yticks()])
    ax4.set_xlim(-0.95, len(plot_data) - 0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")


def make_figure_boltz2_time(
    df,
    annotated_df,
    all_similarity_scores,
    filename,
    similarity_bins=SIMILARITY_BINS,
    rmsd_threshold=RMSD_THRESHOLD,
    lddt_pli_threshold=LDDT_PLI_THRESHOLD,
):
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    ax, ax2 = axs
    ax.text(0.005, 1, "A", transform=ax.transAxes, fontsize=14, fontweight="bold")
    ax2.text(0.005, 1, "B", transform=ax2.transAxes, fontsize=14, fontweight="bold")
    systems = set(df[df["lddt_pli_boltz2"].notna()]["system_id"])

    df = df[(df["system_id"].isin(systems)) & (df["sucos_shape"].notna())].reset_index(
        drop=True
    )
    results_df = df[df["sucos_shape"].notna() & df["ligand_is_proper"]].reset_index(
        drop=True
    )

    similarity_metric = "sucos_shape_pocket_qcov"
    rmsd_column = "rmsd_boltz"
    lddt_pli_column = "lddt_pli_boltz"
    bootstrap = True
    linestyle = "-"
    shape = SHAPES["boltz"]
    color = COLORS["boltz"]
    label = "Boltz-1 (2021)"
    bin_data = []
    bin_sizes = []
    bin_cis = []
    for i in range(len(similarity_bins) - 1):
        mask = (results_df[similarity_metric] >= similarity_bins[i]) & (
            results_df[similarity_metric] < similarity_bins[i + 1]
        )
        bin_sizes.append(len(results_df[mask]))

        mask_success = mask & results_df[lddt_pli_column].notna()
        success_vals = (results_df[mask_success][rmsd_column] < rmsd_threshold) & (
            results_df[mask_success][lddt_pli_column] > lddt_pli_threshold
        )
        bin_data.append(np.mean(success_vals.astype(float)) * 100)

        n_bootstrap = 1000
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(
                success_vals.astype(float), size=len(success_vals), replace=True
            )
            bootstrap_means.append(np.mean(bootstrap_sample) * 100)
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        bin_cis.append((ci_lower, ci_upper))

    x = np.arange(len(bin_sizes))
    bin_cis = np.array(bin_cis)

    ax.plot(
        x,
        bin_data,
        marker=shape,
        markersize=14,
        linestyle=linestyle,
        linewidth=4,
        color=color,
        label=label,
    )
    if bootstrap:
        ax.fill_between(
            x,
            bin_cis[:, 0],
            bin_cis[:, 1],
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel("Similarity to the training set", fontsize=12, fontweight="bold")
    ax.set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    xticklabels = [
        f"{similarity_bins[i]}-{similarity_bins[i + 1]}\nn={bin_sizes[i]:,}"
        for i in range(len(similarity_bins) - 1)
    ]

    similarity_metric = "sucos_shape_pocket_qcov_2023"
    rmsd_column = "rmsd_boltz2"
    lddt_pli_column = "lddt_pli_boltz2"
    linestyle = "-"
    shape = SHAPES["boltz2"]
    color = COLORS["boltz2"]
    label = "Boltz-2 (2023)"
    bin_data = []
    bin_sizes = []
    bin_cis = []
    for i in range(len(similarity_bins) - 1):
        mask = (results_df[similarity_metric] >= similarity_bins[i]) & (
            results_df[similarity_metric] < similarity_bins[i + 1]
        )
        bin_sizes.append(len(results_df[mask]))

        mask_success = mask & results_df[lddt_pli_column].notna()
        success_vals = (results_df[mask_success][rmsd_column] < rmsd_threshold) & (
            results_df[mask_success][lddt_pli_column] > lddt_pli_threshold
        )
        bin_data.append(np.mean(success_vals.astype(float)) * 100)

        n_bootstrap = 1000
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(
                success_vals.astype(float), size=len(success_vals), replace=True
            )
            bootstrap_means.append(np.mean(bootstrap_sample) * 100)
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        bin_cis.append((ci_lower, ci_upper))

    x = np.arange(len(bin_sizes))
    bin_cis = np.array(bin_cis)

    ax.plot(
        x,
        bin_data,
        marker=shape,
        markersize=14,
        linestyle=linestyle,
        linewidth=4,
        color=color,
        label=label,
    )
    if bootstrap:
        ax.fill_between(
            x,
            bin_cis[:, 0],
            bin_cis[:, 1],
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel("Similarity to the training set", fontsize=14, fontweight="bold")
    ax.set_ylabel("Success Rate (%)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    xticklabels = [
        f"{xticklabels[i]}\nm={bin_sizes[i]:,}" for i in range(len(similarity_bins) - 1)
    ]
    ax.set_xticklabels(xticklabels, rotation=0)
    ax.set_ylim(0, 100)

    ax.grid(True, linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=12, frameon=False)
    ax.tick_params(axis="both", which="major", labelsize=14)
    dates = pd.date_range(start="2021-01-01", end="2024-01-01", freq="3M")

    data = []
    for date in dates:
        systems_for_date = set(
            annotated_df[annotated_df["release_date"] > date]["system_id"]
        )
        sim_df = (
            all_similarity_scores[
                all_similarity_scores["query_system"].isin(systems_for_date)
                & (all_similarity_scores["target_release_date"] < date)
            ]
            .sort_values(by="sucos_shape_pocket_qcov", ascending=False)
            .groupby("group_key")
            .head(1)
        )

        bin_counts = (
            pd.cut(sim_df["sucos_shape_pocket_qcov"], bins=SIMILARITY_BINS)
            .value_counts()
            .sort_index()
        )
        data.append({
            "date": date,
            "counts": bin_counts,
            "raw_counts": bin_counts,
            "total_systems": len(systems_for_date),
        })

    plot_data = pd.DataFrame([
        {
            "date": d["date"],
            **{f"{int(b.left)}-{int(b.right)}": c for b, c in d["raw_counts"].items()},
        }
        for d in data
    ])
    ax2 = plot_data.plot(
        x="date",
        kind="area",
        ax=ax2,
        cmap=cmap.Colormap("tol:sunset_r"),
    )
    for i, row in plot_data.iterrows():
        ax2.text(
            row["date"],
            plot_data.iloc[i][1:].sum() + 10,
            data[i]["total_systems"],
            ha="left",
            va="bottom",
            fontsize=12,
        )
        first_4_bins = plot_data.iloc[i][["0-20", "20-30", "30-40", "40-50"]].sum()
        y_pos = (
            plot_data.iloc[i]["0-20"]
            + plot_data.iloc[i]["20-30"]
            + plot_data.iloc[i]["30-40"]
            + plot_data.iloc[i]["40-50"]
        )
        ax2.text(
            row["date"],
            y_pos,
            f"{int(first_4_bins)}",
            ha="left",
            va="bottom",
            fontsize=12,
        )

    ax2.set_xlabel("Training cutoff date", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Number of systems", fontsize=14, fontweight="bold")
    ax2.tick_params(axis="both", which="major", labelsize=14)
    ax2.tick_params(axis="x", which="minor", labelsize=12)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax2.legend(
        title="Similarity Range",
        loc="upper right",
        bbox_to_anchor=(1, 1),
        fontsize=12,
        title_fontsize=12,
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")


def common_ligands_stats(df, filename):
    plt.figure(figsize=(9, 2), dpi=300)
    cutoffs = [
        0,
        1,
        2,
        3,
        5,
        10,
        25,
        50,
        75,
        100,
        500,
        2500,
        10000,
        60000,
    ]
    counts = []
    for cutoff in cutoffs:
        count = (
            df[df["ligand_is_proper"] & (df["sucos_shape"].notna())].drop_duplicates(
                "system_id"
            )["num_training_systems_with_similar_ccds"]
            < cutoff
        ).mean() * 100
        counts.append(count)

    plt.plot(range(len(cutoffs)), counts, "o-", color="black")

    for i, (cutoff, count) in enumerate(zip(cutoffs, counts)):
        plt.annotate(
            str(cutoff),
            (i, count),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=6,
        )

    idx_100 = cutoffs.index(100)
    plt.axvline(x=idx_100, color="red", linestyle="--", alpha=0.5)

    plt.xscale("log")
    plt.xticks([], [])
    plt.xlabel("No. training systems with similar ligands")
    plt.ylabel("% of systems")
    plt.savefig(filename, dpi=300, bbox_inches="tight")


def common_ligands_molecules(df, filename):
    mask = (
        (df["num_training_systems_with_similar_ccds"] > NUM_CCDS_THRESHOLD)
        & (df["ligand_is_proper"])
        & (df["sucos_shape"].notna())
    )
    smiles_100 = dict(
        zip(
            df[mask]["ligand_ccd_code"],
            df[mask]["ligand_smiles"],
        )
    )
    molecules_100 = [Chem.MolFromSmiles(smiles) for smiles in smiles_100.values()]
    labels_100 = [code for code in smiles_100.keys()]

    image = DrawMolsZoomed(
        molecules_100,
        molsPerRow=16,
        subImgSize=(200, 200),
        legends=labels_100,
    )
    image.save(filename)
    return image


def make_figure_stratifications(df, cluster_df, filename):
    fig = plt.figure(figsize=(22, 24))
    gs = plt.GridSpec(
        4,
        3,
        hspace=0.4,
        wspace=0.2,
    )
    axs = []
    for i in range(4):
        for j in range(3):
            axs.append(fig.add_subplot(gs[i, j]))

    make_one_plot_binned(
        cluster_df,
        axs[0],
        "A) No. of rotatable bonds (clustered)",
        "ligand_num_rot_bonds",
        [0, 2, 4, 6, 8, 10, 12, 32],
        "No. of rotatable bonds in the ligand",
    )
    make_one_plot_binned(
        cluster_df,
        axs[1],
        "B) Molecular weight (clustered)",
        "ligand_molecular_weight",
        [0, 200, 300, 400, 500, 600, 800],
        "Molecular weight of the ligand",
        legend_loc=None,
    )

    make_one_plot_binned(
        cluster_df,
        axs[2],
        "C) No. of pocket residues (clustered)",
        "ligand_num_pocket_residues",
        [0, 20, 25, 30, 35, 70],
        "No. of pocket residues",
        legend_loc=None,
    )

    bins = [0, 20, 25, 30, 35, 70]
    labels = "DEFGH"
    for i in range(len(bins) - 1):
        make_one_plot(
            df[
                (df["ligand_num_pocket_residues"] >= bins[i])
                & (df["ligand_num_pocket_residues"] < bins[i + 1])
            ],
            axs[i + 3],
            title=f"{labels[i]}) With {bins[i]}-{bins[i + 1]} pocket residues",
            legend_loc=None,
        )

    make_one_plot(
        df[(df["num_ligand_chains"] == 1)],
        ax=axs[8],
        title="I) Single ligand systems",
        legend_loc=None,
    )
    make_one_plot(
        df[(df["num_proper_ligand_chains"] > 1)],
        ax=axs[9],
        title="J) Multi-proper ligand systems",
        legend_loc=None,
    )
    make_one_plot(
        df[(df["num_protein_chains"] > 1)],
        ax=axs[10],
        title="K) Multi-protein systems",
        legend_loc=None,
    )
    make_one_plot(
        df,
        ax=axs[11],
        title="L) Closest by PLI-QCOV",
        similarity_metric="pli_qcov_pli_qcov",
        legend_loc=None,
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")


def pocket_residues(df, filename):
    fig = plt.figure(figsize=(20, 12))
    gs = plt.GridSpec(
        2,
        3,
        hspace=0.3,
        wspace=0.3,
    )
    axs = []
    for i in range(2):
        for j in range(3):
            axs.append(fig.add_subplot(gs[i, j]))
    bins = [0, 20, 25, 30, 35, 40, 70]
    for i in range(len(bins) - 1):
        make_one_plot(
            df[
                (df["ligand_num_pocket_residues"] >= bins[i])
                & (df["ligand_num_pocket_residues"] < bins[i + 1])
            ],
            axs[i],
            title=f"With {bins[i]}-{bins[i + 1]} pocket residues",
        )
    plt.savefig(filename, dpi=300, bbox_inches="tight")


def cluster_representatives_table(annotated_df):
    similarity_bin_labels = [
        f"{SIMILARITY_BINS[i]}-{SIMILARITY_BINS[i + 1]}"
        for i in range(len(SIMILARITY_BINS) - 1)
    ]
    annotated_df_clean = annotated_df[
        annotated_df["ligand_is_proper"] & (annotated_df["sucos_shape"].notna())
    ].reset_index(drop=True)
    annotated_df_clean["similarity_bin"] = pd.cut(
        annotated_df_clean[SIMILARITY_METRIC].fillna(0),
        bins=SIMILARITY_BINS,
        labels=similarity_bin_labels,
        include_lowest=True,
    )
    annotated_df_clean["cluster_size"] = annotated_df_clean.groupby("cluster")[
        "group_key"
    ].transform("nunique")
    labels = {}
    for group, values in annotated_df_clean.groupby("cluster")["similarity_bin"]:
        labels[group] = ", ".join(
            f"{bin} ({count})"
            for bin, count in values.value_counts().items()
            if count > 0
        )
    annotated_df_clean["cluster_similarity_bins"] = annotated_df_clean["cluster"].map(
        labels
    )
    print(
        " & ".join(["Cluster size", "PDB ID", "Keywords", "Similarity bins"]) + " \\\\"
    )
    for _, c in (
        annotated_df_clean.sort_values(SIMILARITY_METRIC)
        .groupby("cluster")
        .head(1)
        .sort_values("cluster_size", ascending=False)
        .head(10)[
            [
                "cluster",
                "cluster_size",
                "group_key",
                "entry_keywords",
                "cluster_similarity_bins",
            ]
        ]
    ).iterrows():
        print(
            " & ".join([
                str(int(c.cluster_size)),
                c.group_key[:4].upper(),
                c.entry_keywords,
                c.cluster_similarity_bins,
            ])
            + " \\\\"
            + "\n\\midrule\n"
        )


def confidence_plot(
    dfs,
    full_datasets,
    annotated_df,
    filename,
    methods=COMMON_SUBSET_METHODS,
    system_ids=None,
):
    fig = plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], figure=fig, hspace=0.4)
    axs = [
        plt.subplot(gs[0, 0]),
        plt.subplot(gs[0, 1]),
        plt.subplot(gs[1, 0]),
        plt.subplot(gs[1, 1]),
        plt.subplot(gs[2, 0]),
        plt.subplot(gs[2, 1]),
    ]

    colors = {
        "top": "black",
        "best": "blue",
        "worst": "red",
        "top_5": "gray",
        "best_5": "orange",
        "random": "green",
        "random_5": "purple",
    }
    shapes = {
        "top": "o",
        "best": "D",
        "worst": "X",
        "top_5": "p",
        "best_5": "d",
        "random": "s",
        "random_5": "v",
    }
    label_mapping = {
        "top": "Top-ranked (5 seeds)",
        "best": "Best-scored (5 seeds)",
        "worst": "Worst-scored (5 seeds)",
        "top_5": "Top-ranked (1 seed)",
        "best_5": "Best-scored (1 seed)",
        "random": "Random (5 seeds)",
        "random_5": "Random (1 seed)",
    }

    def bootstrap_success_rate(data, n_bootstrap=100):
        if len(data) == 0:
            return 0, 0
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample) * 100)
        return np.mean(bootstrap_means), np.std(bootstrap_means)

    labels = "ABCDEFG"
    for m, method in enumerate(methods):
        ax = axs[m]

        label = labels[m]
        ax.text(
            0.05,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
        )

        bin_data = defaultdict(list)
        bin_sizes = defaultdict(list)
        bin_errors = defaultdict(list)

        for df_name in dfs:
            chosen_df = dfs[df_name][
                dfs[df_name]["sucos_shape"].notna() & dfs[df_name]["ligand_is_proper"]
            ]
            if system_ids is not None:
                chosen_df = chosen_df[chosen_df["system_id"].isin(system_ids)]
            for i in range(len(SIMILARITY_BINS) - 1):
                mask = (chosen_df[SIMILARITY_METRIC] >= SIMILARITY_BINS[i]) & (
                    chosen_df[SIMILARITY_METRIC] < SIMILARITY_BINS[i + 1]
                )

                mask_inner = mask & (chosen_df[f"rmsd_{method}"].notna())
                bin_sizes[df_name].append(len(chosen_df[mask_inner]))
                success = (chosen_df[mask_inner][f"rmsd_{method}"] < RMSD_THRESHOLD) & (
                    chosen_df[mask_inner][f"lddt_pli_{method}"] > LDDT_PLI_THRESHOLD
                )
                mean, std = bootstrap_success_rate(success)
                bin_data[df_name].append(mean)
                bin_errors[df_name].append(std)

        x = np.arange(len(bin_sizes["top"]))

        random_5_means = np.array([bin_data[f"random_5_{i + 1}"] for i in range(5)])
        random_5_mean = np.mean(random_5_means, axis=0)
        random_5_std = np.std(random_5_means, axis=0)
        ax.errorbar(
            x,
            random_5_mean,
            yerr=random_5_std,
            marker=shapes["random_5"],
            markersize=5,
            linestyle="--",
            linewidth=2,
            label=label_mapping["random_5"],
            color=colors["random_5"],
            capsize=5,
        )

        top_5_means = np.array([bin_data[f"top_5_{i + 1}"] for i in range(5)])
        top_5_mean = np.mean(top_5_means, axis=0)
        top_5_std = np.std(top_5_means, axis=0)
        ax.errorbar(
            x,
            top_5_mean,
            yerr=top_5_std,
            marker=shapes["top_5"],
            markersize=5,
            linestyle="-",
            linewidth=2,
            label=label_mapping["top_5"],
            color=colors["top_5"],
            capsize=5,
        )

        best_5_means = np.array([bin_data[f"best_5_{i + 1}"] for i in range(5)])
        best_5_mean = np.mean(best_5_means, axis=0)
        best_5_std = np.std(best_5_means, axis=0)
        ax.errorbar(
            x,
            best_5_mean,
            yerr=best_5_std,
            marker=shapes["best_5"],
            markersize=5,
            linestyle="--",
            linewidth=2,
            label=label_mapping["best_5"],
            color=colors["best_5"],
            capsize=5,
        )

        for df_name in ["random", "best", "worst", "top"]:
            ax.plot(
                x,
                bin_data[df_name],
                marker=shapes[df_name],
                markersize=5,
                linestyle="-",
                linewidth=2,
                label=label_mapping[df_name],
                color=colors[df_name],
                zorder=10,
            )

        if m > 1:
            ax.set_xlabel("Similarity to training set", fontsize=12, fontweight="bold")
        ax.set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                f"{SIMILARITY_BINS[i]}-{SIMILARITY_BINS[i + 1]}\n(n={bin_sizes['top'][i]:,})"
                for i in range(len(SIMILARITY_BINS) - 1)
            ],
            rotation=45,
            fontsize=12,
        )
        ax.set_ylim(0, 100)
        ax.tick_params(axis="y", labelsize=12)

        ax.grid(True, linestyle="--", alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(NAME_MAPPING[method], fontsize=14, fontweight="bold")
        if m == 0:
            ax.legend(loc="upper left", fontsize=10, frameon=False, ncol=1)

    confidence_plot_2(
        axs[-2],
        full_datasets,
        annotated_df,
        column_name="lig_prot_chain_iptm_average_rmsd",
        label_text="E",
        methods=methods,
        ylabel="Ligand-Protein iPTM-based Accuracy (%)",
        system_ids=system_ids,
    )
    confidence_plot_2(
        axs[-1],
        full_datasets,
        annotated_df,
        column_name="prot_lig_chain_iptm_average_rmsd",
        label_text="F",
        methods=[c for c in methods if c != "af3" and c != "protenix"],
        ylabel="Protein-Ligand iPTM-based Accuracy (%)",
        system_ids=system_ids,
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")


def confidence_plot_2(
    ax,
    full_datasets,
    annotated_df,
    label_text,
    column_name="lig_prot_chain_iptm_average_rmsd",
    methods=COMMON_SUBSET_METHODS,
    ylabel="iPTM-based Accuracy (%)",
    system_ids=None,
):
    bin_labels = []
    all_data = {
        method: {
            "acc": [],
            "ci_low_acc": [],
            "ci_high_acc": [],
        }
        for method in methods
    }
    min_sample_size = 250

    for method in methods:
        df = full_datasets[method].merge(
            annotated_df,
            on="group_key",
            how="left",
        )
        if system_ids is not None:
            df = df[df["system_id_x"].isin(system_ids)]
        df = df[df["ligand_is_proper_x"] & (df["sucos_shape"].notna())].reset_index(
            drop=True
        )
        df["success"] = (df["lddt_pli"] > LDDT_PLI_THRESHOLD) & (
            df["rmsd"] < RMSD_THRESHOLD
        )
        y_true = df["success"].astype(int)
        y_scores = df[column_name]
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
        optimal_idx_acc = np.argmax(tpr - fpr)
        threshold_acc = thresholds_roc[optimal_idx_acc]
        all_data[method]["threshold_acc"] = threshold_acc

        if method == methods[0]:
            for i in range(len(SIMILARITY_BINS) - 1):
                bin_mask = (df[SIMILARITY_METRIC] >= SIMILARITY_BINS[i]) & (
                    df[SIMILARITY_METRIC] < SIMILARITY_BINS[i + 1]
                )
                bin_data = df[bin_mask].reset_index(drop=True)
                bin_labels.append(f"{SIMILARITY_BINS[i]}-{SIMILARITY_BINS[i + 1]}")

        for i in range(len(SIMILARITY_BINS) - 1):
            bin_mask = (df[SIMILARITY_METRIC] >= SIMILARITY_BINS[i]) & (
                df[SIMILARITY_METRIC] < SIMILARITY_BINS[i + 1]
            )
            bin_data = df[bin_mask].reset_index(drop=True)

            if len(bin_data) > 0:
                accuracies = []
                for _ in range(20):
                    # Get positive and negative samples
                    pos_samples = bin_data[bin_data["success"]]
                    neg_samples = bin_data[~bin_data["success"]]

                    # Randomly sample equal numbers
                    if min_sample_size > 0:
                        if (
                            len(pos_samples) < min_sample_size
                            or len(neg_samples) < min_sample_size
                        ):
                            pos_sampled = pos_samples.sample(
                                n=min_sample_size, replace=True
                            )
                            neg_sampled = neg_samples.sample(
                                n=min_sample_size, replace=True
                            )
                        else:
                            pos_sampled = pos_samples.sample(
                                n=min_sample_size, replace=False
                            )
                            neg_sampled = neg_samples.sample(
                                n=min_sample_size, replace=False
                            )

                        # Combine samples
                        balanced_data = pd.concat([pos_sampled, neg_sampled])

                        bin_y_true = balanced_data["success"].astype(int)
                        bin_y_scores = balanced_data[column_name]

                        # Calculate Accuracy using ROC threshold
                        y_pred_acc = (bin_y_scores >= threshold_acc).astype(int)
                        accuracy = accuracy_score(bin_y_true, y_pred_acc)
                        accuracies.append(accuracy)
                    else:
                        accuracies.append(0)

                # Calculate mean and 95% confidence intervals
                mean_acc = np.mean(accuracies)

                ci_low_acc, ci_high_acc = np.percentile(accuracies, [2.5, 97.5])

                all_data[method]["acc"].append(mean_acc)

                all_data[method]["ci_low_acc"].append(ci_low_acc)
                all_data[method]["ci_high_acc"].append(ci_high_acc)
            else:
                all_data[method]["acc"].append(0)
                all_data[method]["ci_low_acc"].append(0)
                all_data[method]["ci_high_acc"].append(0)

    # Plot metrics with confidence intervals
    x = range(len(bin_labels))
    axes = [(ax, "acc", ylabel)]

    for ax, metric, title in axes:
        for method in methods:
            label = f"{NAME_MAPPING[method]}"
            if metric == "acc":
                label += f" (threshold={all_data[method]['threshold_acc']:.2f})"

            # Convert fractions to percentages
            y_values = [v * 100 for v in all_data[method][metric]]
            ci_low = [v * 100 for v in all_data[method][f"ci_low_{metric}"]]
            ci_high = [v * 100 for v in all_data[method][f"ci_high_{metric}"]]

            ax.plot(
                x, y_values, marker=SHAPES[method], color=COLORS[method], label=label
            )
            ax.fill_between(x, ci_low, ci_high, color=COLORS[method], alpha=0.2)

            ax.set_xticks(range(len(bin_labels)))
            ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=12)
            ax.set_xlabel(
                "Similarity to the training set", fontsize=12, fontweight="bold"
            )
            ax.set_ylabel(title, fontsize=12, fontweight="bold")
            ax.tick_params(axis="y", labelsize=12)
            ax.legend(ncol=2, frameon=False)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(50, 100)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.text(
                0.05,
                1.05,
                label_text,
                transform=ax.transAxes,
                fontsize=14,
                fontweight="bold",
                va="top",
            )


def rmsd_vs_lddt_pli(
    full_datasets,
    filename,
    methods=COMMON_SUBSET_METHODS,
    rmsd_threshold=RMSD_THRESHOLD,
    lddt_pli_threshold=LDDT_PLI_THRESHOLD,
):
    all_lddt_plis = []
    all_rmsds = []
    colors = []
    for method in methods:
        all_lddt_plis.extend(
            full_datasets[method][
                full_datasets[method]["ligand_is_proper"].fillna(False)
            ]["lddt_pli"].dropna()
        )
        all_rmsds.extend(
            full_datasets[method][
                full_datasets[method]["ligand_is_proper"].fillna(False)
            ]["rmsd"].dropna()
        )
        colors.extend(
            [COLORS[method]]
            * len(
                full_datasets[method][
                    full_datasets[method]["ligand_is_proper"].fillna(False)
                ]["lddt_pli"].dropna()
            )
        )
    all_lddt_plis = np.array(all_lddt_plis)
    all_rmsds = np.array(all_rmsds)

    plt.figure(figsize=(6, 6))
    plt.scatter(all_lddt_plis, all_rmsds, c=colors, alpha=0.01, linewidths=0)
    plt.yscale("log")
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
    plt.xlabel("LDDT-PLI")
    plt.ylabel("RMSD")

    plt.axhline(
        y=rmsd_threshold,
        color="darkred",
        linestyle="--",
        alpha=1,
        label=f"RMSD = {rmsd_threshold}Å",
        linewidth=2,
    )
    plt.axvline(
        x=lddt_pli_threshold,
        color="darkgreen",
        linestyle="--",
        alpha=1,
        label=f"LDDT-PLI = {lddt_pli_threshold}",
        linewidth=2,
    )
    plt.legend()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    lddt_pli_fraction = (
        100
        * all_lddt_plis[
            (all_rmsds < rmsd_threshold) & (all_lddt_plis < lddt_pli_threshold)
        ].shape[0]
        / all_lddt_plis.shape[0]
    )
    print(
        f"Percentage of systems with RMSD < {rmsd_threshold}Å and LDDT-PLI < {lddt_pli_threshold}: {lddt_pli_fraction:.2f}"
    )
    rmsd_fraction = (
        100
        * all_lddt_plis[
            (all_rmsds > rmsd_threshold) & (all_lddt_plis > lddt_pli_threshold)
        ].shape[0]
        / all_lddt_plis.shape[0]
    )
    print(
        f"Percentage of systems with RMSD > {rmsd_threshold}Å and LDDT-PLI > {lddt_pli_threshold}: {rmsd_fraction:.2f}"
    )


def example_molecules_per_bin(cluster_df, filename):
    # Get 10 random smiles from each similarity bin
    smiles_by_bin = {}
    for i in range(len(SIMILARITY_BINS) - 1):
        bin_mask = (
            (cluster_df[SIMILARITY_METRIC] >= SIMILARITY_BINS[i])
            & (cluster_df[SIMILARITY_METRIC] <= SIMILARITY_BINS[i + 1])
            & (
                cluster_df["num_training_systems_with_similar_ccds"]
                < NUM_CCDS_THRESHOLD
            )
            & (cluster_df["ligand_is_proper"])
        )
        bin_df = cluster_df[bin_mask]
        if len(bin_df) > 0:
            sample_df = bin_df.sample(n=min(10, len(bin_df)))
            smiles_by_bin[f"{SIMILARITY_BINS[i]}-{SIMILARITY_BINS[i + 1]}"] = dict(
                zip(sample_df["ligand_ccd_code"], sample_df["ligand_smiles"])
            )

    all_smiles = []
    row_labels = []

    for bin_label, bin_smiles in smiles_by_bin.items():
        if bin_smiles:
            row_labels.append(f"Similarity {bin_label}")
            for code, smiles in bin_smiles.items():
                all_smiles.append((code, smiles))

    molecules = [Chem.MolFromSmiles(smiles) for _, smiles in all_smiles]
    labels = [code for code, _ in all_smiles]

    image = DrawMolsZoomed(
        molecules,
        molsPerRow=10,
        subImgSize=(200, 200),
        legends=labels,
        row_labels=row_labels,
    )
    image.save(filename)
    return image


def make_figure_ligand_prevalence(df, filename):
    plt.figure(figsize=(9, 2), dpi=300)
    cutoffs = [
        0,
        1,
        2,
        3,
        5,
        10,
        25,
        50,
        75,
        100,
        500,
        2500,
        10000,
        60000,
    ]
    counts = []
    for cutoff in cutoffs:
        count = (
            df.drop_duplicates("system_id")["num_training_systems_with_similar_ccds"]
            < cutoff
        ).mean() * 100
        counts.append(count)

    plt.plot(range(len(cutoffs)), counts, "o-", color="black")

    for i, (cutoff, count) in enumerate(zip(cutoffs, counts)):
        plt.annotate(
            str(cutoff),
            (i, count),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=6,
        )

    idx_100 = cutoffs.index(100)
    plt.axvline(x=idx_100, color="red", linestyle="--", alpha=0.5)

    plt.xscale("log")
    plt.xticks([], [])
    plt.xlabel("No. training systems with similar ligands")
    plt.ylabel("% of systems")
    plt.savefig(filename, dpi=300, bbox_inches="tight")


def make_figure_prevalent_ligands(df, filename):
    mask = df["num_training_systems_with_similar_ccds"] > NUM_CCDS_THRESHOLD
    smiles_100 = dict(
        zip(
            df[mask]["ligand_ccd_code"],
            df[mask]["ligand_smiles"],
        )
    )
    molecules_100 = [Chem.MolFromSmiles(smiles) for smiles in smiles_100.values()]
    labels_100 = [code for code in smiles_100.keys()]

    image = DrawMolsZoomed(
        molecules_100,
        molsPerRow=16,
        subImgSize=(200, 200),
        legends=labels_100,
    )
    image.save(filename)
    return image
