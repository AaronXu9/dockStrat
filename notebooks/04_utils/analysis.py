import os
import tempfile
from typing import Tuple, Dict, List

import pandas as pd
import numpy as np
import MDAnalysis as mda
import prolif as plf
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from prolif.molecule import Molecule
import json

def compare_interaction_fingerprints(ref_bool: pd.Series, pred_bool: pd.Series):
    """
    Given two boolean Series (with a MultiIndex, e.g. (ligand, protein, interaction))
    representing the reference and predicted interaction fingerprints, align them over the
    union of their indices and compute:
      - overall_overlap: (common interactions)/(total interactions in reference)
      - Per-interaction type statistics: for each interaction type,
        count common interactions, count in reference, count in predicted, and overlap ratio.
    
    Returns:
      - detail_results: A dictionary mapping each interaction type to a dictionary with keys:
          "common", "ref_count", "pred_count", "overlap_ratio"
      - overall_overlap: The overall overlap ratio.
    """
    # Align both Series to the union of their indices
    all_idx = ref_bool.index.union(pred_bool.index)
    ref_aligned = ref_bool.reindex(all_idx, fill_value=False)
    pred_aligned = pred_bool.reindex(all_idx, fill_value=False)
    
    common_total = (ref_aligned & pred_aligned).sum()
    ref_total = ref_aligned.sum()
    overall_overlap = common_total / ref_total if ref_total > 0 else np.nan

    detail_results = {}
    # Group by interaction type (assumes the MultiIndex has a level named "interaction")
    for interaction, group in ref_aligned.groupby(level="interaction"):
    # Suppose 'ref_aligned' is a DataFrame with multi-level columns.
        ref_aligned.index = ref_aligned.index.set_names(["ligand", "protein", "interaction"])
        pred_group = pred_aligned.reindex(group.index, fill_value=False)
        common = (group & pred_group).sum()
        ref_count = group.sum()
        pred_count = pred_group.sum()
        ratio = common / ref_count if ref_count > 0 else np.nan
        detail_results[interaction] = {
            "common": common,
            "ref_count": ref_count,
            "pred_count": pred_count,
            "overlap_ratio": ratio
        }
    return detail_results, overall_overlap


def comprehensive_analysis(df: pd.DataFrame, methods: List[str]) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    enriched_rows = []
    omitted = []

    # Caches for reference and predicted fingerprints.
    ref_cache = {}
    pred_cache = {}

    for idx, row in df.iterrows():
        true_ligand = row["true_ligand"]
        predicted_ligand = row["predicted_ligand"]
        protein_path = row["protein_pdb"]
        print(f'Computing fingerprints for {row["protein"]}')
        # Use a tuple key based on true ligand and protein.
        ref_key = (true_ligand, protein_path)
        if ref_key in ref_cache:
            ref_fp_df = ref_cache[ref_key]
        else:
            try:
                ref_fp_df = compute_reference_fingerprint(true_ligand, protein_path)
                ref_cache[ref_key] = ref_fp_df
            except Exception as e:
                print(f"Error computing reference fingerprint for row {idx}: {e}")
                continue

        # Similarly, cache the predicted fingerprint.
        pred_key = (predicted_ligand, protein_path)
        if pred_key in pred_cache:
            pred_fp_df = pred_cache[pred_key]
        else:
            try:
                pred_fp_df = compute_reference_fingerprint(predicted_ligand, protein_path)
                pred_cache[pred_key] = pred_fp_df
            except Exception as e:
                print(f"Error computing predicted fingerprint for row {idx}: {e}")
                continue

        # Convert to booleans: an interaction is present if count > 0.
        ref_bool = (ref_fp_df.iloc[0] > 0)
        pred_bool = (pred_fp_df.iloc[0] > 0)

        # Compare fingerprints using the union of their keys.
        detail_results, overall_overlap = compare_interaction_fingerprints(ref_bool, pred_bool)
        ref_total = ref_bool.sum()
        omit_protein = (ref_total == 0)
        if omit_protein:
            omitted.append({
                "protein": row.get("protein", "Unknown"),
                "protein_pdb": protein_path,
                "true_ligand": true_ligand
            })
        
        # Update the row with new information.
        new_row = row.to_dict()
        new_row["overall_overlap"] = overall_overlap
        ref_dict = { "_".join(map(str, k)) if isinstance(k, tuple) else str(k): v 
                for k, v in ref_bool.to_dict().items() }
        new_row["ref_interactions"] = json.dumps(ref_dict)
        new_row["omit_protein"] = omit_protein
        for interaction, stats in detail_results.items():
            col_name = f"overlap_{interaction}"
            new_row[col_name] = stats["overlap_ratio"]
        enriched_rows.append(new_row)

    enriched_df = pd.DataFrame(enriched_rows)

    if omitted:
        print("The following proteins have no interactions in the reference:")
        print(pd.DataFrame(omitted)[["protein", "protein_pdb", "true_ligand"]])

    return enriched_df


def compute_reference_fingerprint(ref_ligand_path: str, protein_path: str) -> pd.DataFrame:
    """
    Generates a ProLIF fingerprint for a single reference pose.
    Returns a DataFrame with a single row (Pose = 'reference').
    """
    fp_calc = plf.Fingerprint() # You can set count=True if needed

    # Load the reference ligand
    try:
        protein_name = protein_path.split("/")[-1].split(".")[0]
        # print(protein_name)
        # Load the reference ligand
        supplier = Chem.SDMolSupplier(ref_ligand_path)
        ligands = [Molecule.from_rdkit(mol) for mol in supplier if mol is not None]
        ref_mol = ligands[0]  # Take the first ligand (if multiple poses exist)
        ref_mol.SetProp("_Name", "reference")  # label the pose

        # Load the protein
        rdkit_prot = Chem.MolFromPDBFile(protein_path, removeHs=False)
        protein = plf.Molecule(rdkit_prot)

        # Run ProLIF
        fp_calc.run_from_iterable([ref_mol], protein)

        # Convert to DataFrame
        ref_fp_df = fp_calc.to_dataframe(index_col="Pose")
    except Exception as e:
        print(f"{protein_name}: Error computing reference fingerprint: {str(e)}")
        return None
    return ref_fp_df


class ConformationAnalyzer:
    """
    Analyzes protein-ligand interactions for multiple conformations using ProLIF.
    This class handles loading structures and generating interaction fingerprints.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame that contains paths to the predicted ligand,
        protein PDB, method, and rank.
        """
        self.df = df
        self.fp_calculator = plf.Fingerprint()
        self.results = {}  # Method -> fingerprint DataFrame mapping

    def combine_conformations(self, method_name: str) -> Tuple[str, str]:
        """
        Combines multiple conformations for one method into a single SDF file.
        Returns the temporary ligand SDF file path and the protein PDB path.
        """
        method_data = self.df[self.df['method'] == method_name].sort_values('rank')
        temp_sdf = tempfile.NamedTemporaryFile(suffix='.sdf', delete=False)
        writer = Chem.SDWriter(temp_sdf.name)
        for _, row in method_data.iterrows():
            mol_supplier = Chem.SDMolSupplier(str(row['predicted_ligand']))
            if mol_supplier:
                mol = mol_supplier[0]
                if mol is not None:
                    mol.SetProp('_Name', f"Pose_{row['rank']}")
                    writer.write(mol)
        writer.close()
        protein_path = str(method_data.iloc[0]['protein_pdb'])
        return temp_sdf.name, protein_path

    def analyze_method(self, method_name: str) -> pd.DataFrame:
        """
        Analyzes all conformations for a specified method and returns the fingerprint DataFrame.
        """
        try:
            ligand_file, protein_path = self.combine_conformations(method_name)
            ligands = plf.sdf_supplier(ligand_file)
            rdkit_prot = Chem.MolFromPDBFile(protein_path, removeHs=False)
            protein = plf.Molecule(rdkit_prot)
            self.fp_calculator.run_from_iterable(ligands, protein)
            fp_df = self.fp_calculator.to_dataframe(index_col="Pose")
            self.results[method_name] = fp_df
            os.unlink(ligand_file)
            return fp_df
        except Exception as e:
            print(f"Error analyzing method {method_name}: {str(e)}")
            if 'ligand_file' in locals():
                os.unlink(ligand_file)
            raise

    def analyze_all_methods(self) -> Dict[str, pd.DataFrame]:
        """
        Runs analysis for all unique methods in self.df.
        Returns a dictionary mapping each method to its fingerprint DataFrame.
        """
        methods = self.df['method'].unique()
        for method in methods:
            self.analyze_method(method)
        return self.results

    def summarize_interactions(self, method_name: str) -> pd.DataFrame:
        """
        Summarizes how often each interaction appears across all poses for a given method.
        """
        if method_name not in self.results:
            self.analyze_method(method_name)
        fp_df = self.results[method_name]
        summary = pd.DataFrame({
            'occurrence_rate': fp_df.mean(),
            'always_present': fp_df.all(),
            'never_present': ~fp_df.any(),
            'variable': fp_df.any() & ~fp_df.all()
        })
        return summary

    def compare_with_reference(self, method_name: str, ref_fp_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare the fingerprint of each pose from a method to a single-row reference fingerprint.
        Returns a summary DataFrame for each pose containing counts for missing, extra, and common interactions.
        """
        if method_name not in self.results:
            self.analyze_method(method_name)
        method_fp = self.results[method_name]
        # Get the reference fingerprint (assume single row)
        ref_index = ref_fp_df.index[0]
        ref_fp = ref_fp_df.loc[ref_index]

        comparison_rows = []
        for pose_label, row in method_fp.iterrows():
            # Compute the union of indices from both the reference and the pose.
            all_idx = ref_fp.index.union(row.index)
            # Reindex both series using the union, filling missing keys with False.
            ref_aligned = ref_fp.reindex(all_idx, fill_value=False)
            row_aligned = row.reindex(all_idx, fill_value=False)

            # Now compute masks:
            missing_mask = ref_aligned & ~row_aligned   # interactions expected (in reference) but missing in pose.
            extra_mask   = ~ref_aligned & row_aligned    # interactions not expected (absent in reference) but present in pose.
            common_mask  = ref_aligned & row_aligned       # interactions that are present in both.

            comparison_info = {
                'pose': pose_label,
                'num_missing': missing_mask.sum(),
                'num_extra': extra_mask.sum(),
                'num_common': common_mask.sum(),
                'total_in_ref': ref_aligned.sum(),
                'total_in_pose': row_aligned.sum(),
            }
            comparison_rows.append(comparison_info)
        return pd.DataFrame(comparison_rows)
    
    def compare_with_reference_detailed(self, method_name: str, ref_fp_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns two DataFrames that detail, for each pose, the counts of interactions:
        - missing_df: interactions present in the reference but missing in the pose.
        - extra_df: interactions present in the pose but not in the reference.
        
        This function aligns the reference and pose fingerprints over the union of their indices
        so that any extra interactions (i.e. keys in the pose that are missing in the reference)
        are not dropped.
        """
        if method_name not in self.results:
            self.analyze_method(method_name)
        method_fp = self.results[method_name]
        ref_index = ref_fp_df.index[0]
        ref_fp = ref_fp_df.loc[ref_index]
        
        # Get the set of interaction types that appear in the method fingerprint.
        interaction_types = method_fp.columns.get_level_values("interaction").unique()
        
        # Initialize empty DataFrames for detailed results.
        missing_df = pd.DataFrame(0, index=method_fp.index, columns=interaction_types)
        extra_df = pd.DataFrame(0, index=method_fp.index, columns=interaction_types)
        
        for pose_label, pose_fp in method_fp.iterrows():
            # Compute union of indices from reference and pose.
            all_idx = ref_fp.index.union(pose_fp.index)
            # Reindex both series to cover all keys, filling missing with False.
            ref_aligned = ref_fp.reindex(all_idx, fill_value=False)
            pose_aligned = pose_fp.reindex(all_idx, fill_value=False)
            
            # Compute boolean masks:
            missing_mask = ref_aligned & ~pose_aligned   # Present in reference but missing in pose.
            extra_mask   = ~ref_aligned & pose_aligned    # Present in pose but not in reference.
            
            # Group by the "interaction" level (aggregating across residues).
            missing_counts = missing_mask.astype(int).groupby(level="interaction").sum()
            extra_counts = extra_mask.astype(int).groupby(level="interaction").sum()
            
            # Fill the pre-initialized DataFrames. If an interaction type from the union is missing in the initial columns,
            # it will be ignored; you can choose to update missing_df and extra_df to include new columns if desired.
            for i_type, i_count in missing_counts.items():
                if i_type in missing_df.columns:
                    missing_df.at[pose_label, i_type] = i_count
            for i_type, i_count in extra_counts.items():
                if i_type in extra_df.columns:
                    extra_df.at[pose_label, i_type] = i_count
                    
        return missing_df, extra_df


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
    for complex_id, group in input_df.groupby("protein"):
        print(f"Processing complex {complex_id} ...")
        first_row = group.iloc[0]
        ref_ligand_path = first_row["true_ligand"]
        protein_path = first_row["protein_pdb"]
        ref_fp_df = compute_reference_fingerprint(ref_ligand_path, protein_path)
        analyzer = ConformationAnalyzer(group)
        complex_results = {}
        for method in methods:
            try:
                print(f" Analyzing method: {method}")
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
                print(f" Error processing method {method} for complex {complex_id}: {e}")
        results[complex_id] = complex_results
    return results


def aggregate_detailed_results(all_results: dict, detail: str = "missing") -> pd.DataFrame:
    """
    Aggregates detailed interaction counts (missing or extra) across all complexes for each method.
    Returns a DataFrame with rows as interaction types and columns as methods.
    """
    method_dict = {}
    for complex_id, complex_data in all_results.items():
        for method, result_dict in complex_data.items():
            df_detail = result_dict["detailed"][0] if detail == "missing" else result_dict["detailed"][1]
            avg_detail = df_detail.mean(axis=0)
            method_dict.setdefault(method, []).append(avg_detail)
    method_summary = {m: pd.concat(series_list, axis=1).mean(axis=1) for m, series_list in method_dict.items()}
    return pd.DataFrame(method_summary)


def aggregate_overall_overall(all_results: dict) -> pd.DataFrame:
    """
    Aggregates overall comparison metrics across complexes and methods.
    Returns a DataFrame with overall averages per method.
    """
    rows = []
    for complex_id, methods_dict in all_results.items():
        for method, result_dict in methods_dict.items():
            overall_df = result_dict["overall"]
            rows.append({
                "complex_id": complex_id,
                "method": method,
                "avg_num_missing": overall_df['num_missing'].mean(),
                "avg_num_extra": overall_df['num_extra'].mean(),
                "avg_num_common": overall_df['num_common'].mean(),
                "total_in_ref": overall_df['total_in_ref'].iloc[0],
                "avg_total_in_pose": overall_df['total_in_pose'].mean()
            })
    agg_df = pd.DataFrame(rows)
    numeric_cols = ["avg_num_missing", "avg_num_extra", "avg_num_common", "total_in_ref", "avg_total_in_pose"]
    method_summary = agg_df.groupby("method")[numeric_cols].mean().reset_index()
    return method_summary


def generate_summary_report(all_results: dict) -> dict:
    """
    Generates an overall summary report including:
      - Overall comparison metrics.
      - Detailed missing interactions.
      - Detailed extra interactions.
    Returns a dictionary with these three DataFrames.
    """
    overall_df = aggregate_overall_overall(all_results)
    detailed_missing_df = aggregate_detailed_results(all_results, detail="missing")
    detailed_extra_df = aggregate_detailed_results(all_results, detail="extra")
    return {
        "overall": overall_df,
        "detailed_missing": detailed_missing_df,
        "detailed_extra": detailed_extra_df,
    }