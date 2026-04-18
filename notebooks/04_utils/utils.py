import os
import glob
import argparse
import tempfile
import os
from pathlib import Path
from typing import Tuple, Dict, List

import pandas as pd
import MDAnalysis as mda
import prolif as plf
from rdkit import DataStructs
from rdkit import Chem
from prolif.molecule import Molecule

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_all_complexes(input_df: pd.DataFrame, methods: List[str]) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Analyzes multiple protein-ligand pairs. Assumes the input DataFrame has columns:
      - complex_id: Unique identifier for each protein-ligand pair.
      - predicted_ligand: Docked ligand SDF file path.
      - true_ligand: Reference (true) ligand SDF file path.
      - protein_pdb: Protein PDB file path.
      - method: Docking method name.
      - rank: Pose rank.

    For each unique complex, the function:
      1. Computes the reference fingerprint.
      2. Initializes a ConformationAnalyzer for the complex.
      3. For each method in the provided list, computes:
           - Overall comparison summary.
           - Detailed missing/extra interaction counts.
           - Optionally, an interaction summary.

    Returns:
        A nested dictionary in the form:
        {
          complex_id: {
              method: {
                  "overall": overall_comparison_df,
                  "detailed": (missing_df, extra_df),
                  "summary": interaction_summary_df
              },
              ...
          },
          ...
        }
    """
    results = {}

    # Group by unique complex. Here we assume 'complex_id' uniquely identifies a pair.
    for complex_id, group in input_df.groupby("protein"):
        print(f"Processing complex {complex_id} ...")
        # Get the first row to extract reference paths.
        first_row = group.iloc[0]
        ref_ligand_path = first_row["true_ligand"]
        protein_path = first_row["protein_pdb"]

        # Compute the reference fingerprint.
        print("compuing fingerprints:")
        ref_fp_df = compute_reference_fingerprint(ref_ligand_path, protein_path)

        # Initialize an analyzer for this complex.
        analyzer = ConformationAnalyzer(group)

        complex_results = {}
        for method in methods:
            try:
                print(f"  Analyzing method: {method}")
                analyzer.analyze_method(method)
                overall = analyzer.compare_with_reference(method, ref_fp_df)
                missing_df, extra_df = analyzer.compare_with_reference_detailed(method, ref_fp_df)
                summary = analyzer.summarize_interactions(method)

                complex_results[method] = {
                    "overall": overall,
                    "detailed": (missing_df, extra_df),
                    "summary": summary
                }
            except Exception as e:
                print(f"Error processing method {method} for complex {complex_id}: {e}")
        results[complex_id] = complex_results
    return results

def aggregate_detailed_results(all_results: dict, detail: str = "missing") -> pd.DataFrame:
    """
    Aggregates detailed interaction counts (missing or extra) across all complexes for each method.

    Parameters:
      all_results: Nested dictionary from analyze_all_complexes, of the form:
        {
          complex_id: {
              method: {
                  "detailed": (missing_df, extra_df),
                  ...
              },
              ...
          },
          ...
        }
      detail: "missing" or "extra" to indicate which detail to aggregate.

    Returns:
      A DataFrame where rows are interaction types and columns are methods. The values
      are the average (across complexes) of the average counts (across poses) for each interaction type.
    """
    # Dictionary to accumulate results per method.
    method_dict = {}
    for complex_id, complex_data in all_results.items():
        for method, result_dict in complex_data.items():
            # result_dict["detailed"] returns a tuple: (missing_df, extra_df)
            if detail == "missing":
                df_detail = result_dict["detailed"][0]
            elif detail == "extra":
                df_detail = result_dict["detailed"][1]
            else:
                raise ValueError("detail must be either 'missing' or 'extra'")

            # For this complex and method, average over poses (rows).
            avg_detail = df_detail.mean(axis=0)  # Series: index=interaction types, value = average count
            if method not in method_dict:
                method_dict[method] = []
            method_dict[method].append(avg_detail)

    # Now aggregate for each method over complexes by averaging the Series.
    method_summary = {}
    for method, series_list in method_dict.items():
        # Concatenate all Series into a DataFrame and average over columns.
        combined = pd.concat(series_list, axis=1)
        method_summary[method] = combined.mean(axis=1)

    # Create a summary DataFrame: rows are interaction types, columns are methods.
    summary_df = pd.DataFrame(method_summary)
    return summary_df

def plot_detailed_heatmap(summary_df: pd.DataFrame, title: str):
    """
    Plots a heatmap of the aggregated detailed interaction counts.

    Parameters:
      summary_df: DataFrame with rows as interaction types and columns as methods.
      title: Title for the plot.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(summary_df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title(title)
    plt.xlabel("Method")
    plt.ylabel("Interaction Type")
    plt.tight_layout()
    plt.show()

def plot_detailed_bar(summary_df: pd.DataFrame, title: str):
    """
    Plots a grouped bar chart for the aggregated detailed interaction counts.

    Parameters:
      summary_df: DataFrame with rows as interaction types and columns as methods.
      title: Title for the plot.
    """
    # Ensure the index is named "interaction"
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

def generate_summary_report(all_results: dict) -> dict:
    """
    Generates an overall summary report that includes both overall and detailed analyses.

    Returns a dictionary containing:
      - overall_df: A DataFrame aggregating overall comparison metrics (from aggregate_overall_results).
      - detailed_missing_df: A DataFrame summarizing detailed missing interactions (average per interaction type by method).
      - detailed_extra_df: A DataFrame summarizing detailed extra interactions.
    """
    overall_df = aggregate_overall_overall(all_results)  # We'll assume you have a function for overall aggregation.
    detailed_missing_df = aggregate_detailed_results(all_results, detail="missing")
    detailed_extra_df = aggregate_detailed_results(all_results, detail="extra")

    report = {
        "overall": overall_df,
        "detailed_missing": detailed_missing_df,
        "detailed_extra": detailed_extra_df,
    }
    return report

# If you already have the overall aggregation from before:
def aggregate_overall_overall(all_results: dict) -> pd.DataFrame:
    """
    Aggregates overall metrics across complexes and methods.
    Returns a DataFrame with overall averages for each complex-method pair,
    then aggregated by method.
    """
    rows = []
    for complex_id, methods_dict in all_results.items():
        for method, result_dict in methods_dict.items():
            overall_df = result_dict["overall"]
            avg_missing = overall_df['num_missing'].mean()
            avg_extra = overall_df['num_extra'].mean()
            avg_common = overall_df['num_common'].mean()
            avg_total_in_pose = overall_df['total_in_pose'].mean()
            total_in_ref = overall_df['total_in_ref'].iloc[0]
            rows.append({
                "complex_id": complex_id,
                "method": method,
                "avg_num_missing": avg_missing,
                "avg_num_extra": avg_extra,
                "avg_num_common": avg_common,
                "total_in_ref": total_in_ref,
                "avg_total_in_pose": avg_total_in_pose
            })
    agg_df = pd.DataFrame(rows)
    # Group only by numeric columns; we don't average the 'complex_id' column.
    numeric_cols = ["avg_num_missing", "avg_num_extra", "avg_num_common", "total_in_ref", "avg_total_in_pose"]
    method_summary = agg_df.groupby("method")[numeric_cols].mean().reset_index()
    return method_summary

def plot_summary_report(report: dict):
    """
    Generates plots for the overall and detailed summary reports.
    """
    overall_df = report["overall"]
    detailed_missing_df = report["detailed_missing"]
    detailed_extra_df = report["detailed_extra"]

    # Plot overall metrics (bar charts)
    plt.figure(figsize=(12, 4))
    sns.barplot(data=overall_df, x="method", y="avg_num_missing")
    plt.title("Average Missing Interactions (Overall)")
    plt.ylabel("Average Missing Count")
    plt.xlabel("Method")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    sns.barplot(data=overall_df, x="method", y="avg_num_extra")
    plt.title("Average Extra Interactions (Overall)")
    plt.ylabel("Average Extra Count")
    plt.xlabel("Method")
    plt.tight_layout()
    plt.show()

    # Plot detailed interactions as heatmaps:
    plot_detailed_heatmap(detailed_missing_df, "Detailed Missing Interactions (Average per Interaction Type)")
    plot_detailed_heatmap(detailed_extra_df, "Detailed Extra Interactions (Average per Interaction Type)")

    # Optionally, you can also use bar charts:
    plot_detailed_bar(detailed_missing_df, "Detailed Missing Interactions (Average per Interaction Type)")
    plot_detailed_bar(detailed_extra_df, "Detailed Extra Interactions (Average per Interaction Type)")