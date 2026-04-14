from typing import Tuple, Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px

def plot_interaction_overlap(df_melted):
    """
    Plots a grouped bar chart of overlap (y) by interaction_type (x),
    grouped by method (hue).
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_melted,
        x="interaction_type", y="overlap", hue="method",
        palette="Set2"
    )
    # Optional: reference line at y=1
    plt.axhline(1.0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel("Interaction Type")
    plt.ylabel("Overlap")
    plt.xticks(rotation=45, ha="right")
    plt.title("Overlap by Method and Interaction Type")
    plt.legend()
    plt.tight_layout()
    plt.show()

    
def plot_single_approach_posebusters(df: pd.DataFrame, approach_name: str, test_cols: list) -> None:
    """
    Creates a bar plot showing the percentages of poses passing various tests.
    """
    total = len(df)
    results = []
    for col in test_cols:
        pass_count = df[col].sum()
        prop = pass_count / total if total else 0.0
        results.append((col, pass_count, total, prop))
    prop_df = pd.DataFrame(results, columns=["test", "pass_count", "total", "proportion"])
    prop_df["percentage"] = 100.0 * prop_df["proportion"]
    prop_df["test"] = pd.Categorical(prop_df["test"], categories=test_cols, ordered=True)
    prop_df.sort_values("test", inplace=True)
    plt.figure(figsize=(15, 10))
    sns.barplot(data=prop_df, x="test", y="percentage", color="salmon")
    plt.ylim(0, 110)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("")
    plt.ylabel("Passing (%)")
    plt.title(f"PoseBusters Checks: {approach_name}")
    ax = plt.gca()
    for i, row in prop_df.iterrows():
        ax.text(i, row["percentage"] + 1, f"{int(row['pass_count'])}/{int(row['total'])}",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_detailed_heatmap(summary_df: pd.DataFrame, title: str) -> None:
    """
    Plots a heatmap for aggregated detailed interaction counts.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(summary_df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title(title)
    plt.xlabel("Method")
    plt.ylabel("Interaction Type")
    plt.tight_layout()
    plt.show()


def plot_detailed_bar(summary_df: pd.DataFrame, title: str) -> None:
    """
    Plots a grouped bar chart for detailed interaction counts.
    """
    summary_df = summary_df.rename_axis("interaction").reset_index()
    melted = summary_df.melt(id_vars="interaction", var_name="Method", value_name="Average Count")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x="interaction", y="Average Count", hue="Method")
    plt.title(title)
    plt.xlabel("Interaction Type")
    plt.ylabel("Average Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_overall_bar(overall_df: pd.DataFrame, metric: str, title: str) -> None:
    """
    Plots a bar chart for an overall metric from the summary report.
    """
    plt.figure(figsize=(12, 4))
    sns.barplot(data=overall_df, x="method", y=metric)
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("Method")
    plt.tight_layout()
    plt.show()


def plot_cumulative_ecdf(df: pd.DataFrame, x_col: str, hue_col: str, xlim: Tuple[float, float] = (0, 10)) -> None:
    """
    Plots an empirical CDF (ECDF) for x_col, grouping by the hue_col.
    """
    plt.figure(figsize=(8, 6))
    ax = sns.ecdfplot(data=df, x=x_col, hue=hue_col)
    plt.xlabel(f"{x_col}")
    plt.xlim(xlim)
    plt.ylabel(f"Proportion of Examples with {x_col} ≤ x")
    plt.title(f"Cumulative Distribution of {x_col} by {hue_col}")
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, title=hue_col, loc="best")
    plt.show()


def plot_score_vs_rmsd(df: pd.DataFrame, method: str) -> None:
    """
    Creates a scatter plot of score vs RMSD for a given method
    and adds a regression line along with the correlation coefficient.
    """
    method_data = df[(df["method"] == method) & (df["score"].notna())]
    plt.figure(figsize=(10, 6))
    plt.scatter(method_data["score"], method_data["rmsd"], alpha=0.5)
    if not method_data.empty:
        z = np.polyfit(method_data["score"], method_data["rmsd"], 1)
        p = np.poly1d(z)
        plt.plot(method_data["score"], p(method_data["score"]), "r--", alpha=0.8)
        corr = method_data["score"].corr(method_data["rmsd"])
        plt.text(0.05, 0.95, f"r = {corr:.2f}", transform=plt.gca().transAxes, verticalalignment="top")
    plt.title(f"RMSD vs Score for {method}")
    plt.xlabel("Score")
    plt.ylabel("RMSD (Å)")
    plt.tight_layout()
    plt.show()