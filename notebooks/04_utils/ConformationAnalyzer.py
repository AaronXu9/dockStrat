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
from rdkit.Chem import rdMolTransforms

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

class ConformationAnalyzer:
    """
    Analyzes protein-ligand interactions for multiple conformations using ProLIF.
    This class handles loading structures and generating interaction fingerprints
    for multiple poses of the same protein-ligand system.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame containing paths to structures.

        Parameters:
            df: DataFrame with columns for predicted_ligand, protein_pdb, method, and rank
        """
        self.df = df
        self.fp_calculator = plf.Fingerprint()
        self.results = {}  # Stores method -> fingerprint DataFrame

    def combine_conformations(self, method_name: str) -> tuple[str, str]:
        """
        Combines multiple conformations into a single SDF file for a given method.
        Returns the combined ligand SDF path and protein PDB path.
        """
        method_data = self.df[self.df['method'] == method_name].sort_values('rank')

        temp_sdf = tempfile.NamedTemporaryFile(suffix='.sdf', delete=False)
        writer = Chem.SDWriter(temp_sdf.name)

        for _, row in method_data.iterrows():
            mol = Chem.SDMolSupplier(str(row['predicted_ligand']))[0]
            if mol is not None:
                mol.SetProp('_Name', f"Pose_{row['rank']}")
                writer.write(mol)
        writer.close()

        protein_path = str(method_data.iloc[0]['protein_pdb'])
        return temp_sdf.name, protein_path

    def analyze_method(self, method_name: str) -> pd.DataFrame:
        """
        Analyzes protein-ligand interactions for all conformations of a method
        and returns a ProLIF fingerprint DataFrame.
        """
        try:
            ligand_file, protein_path = self.combine_conformations(method_name)

            ligands = plf.sdf_supplier(ligand_file)
            rdkit_prot = Chem.MolFromPDBFile(protein_path, removeHs=False)
            protein = plf.Molecule(rdkit_prot)

            self.fp_calculator.run_from_iterable(ligands, protein)
            fp_df = self.fp_calculator.to_dataframe(index_col="Pose")
            self.results[method_name] = fp_df

            # Cleanup temp file
            os.unlink(ligand_file)

            return fp_df
        except Exception as e:
            print(f"Error analyzing method {method_name}: {str(e)}")
            if 'ligand_file' in locals():
                os.unlink(ligand_file)
            raise

    def analyze_all_methods(self) -> dict:
        """
        Runs analyze_method() for all unique methods in self.df.
        Returns a dict of {method: fingerprint_df}.
        """
        methods = self.df['method'].unique()
        for method in methods:
            self.analyze_method(method)
        return self.results

    def summarize_interactions(self, method_name: str) -> pd.DataFrame:
        """
        Summarizes how often each interaction appears across all poses of a method.
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
        Compare the fingerprint of each pose (for the given method) to a single-row
        reference fingerprint (e.g., a crystal pose).

        Returns a summary DataFrame that shows, for each pose:
            - number of 'missing' interactions (in ref but not in pose)
            - number of 'extra' interactions (in pose but not in ref)
            - number of 'common' interactions (in both ref and pose)
            - total interactions in reference
            - total interactions in pose
        """
        # Ensure we have the method fingerprint
        if method_name not in self.results:
            self.analyze_method(method_name)
        method_fp = self.results[method_name]

        # ref_fp_df should have exactly one row (index = "reference", or something similar)
        # If you have multiple rows in the reference, select one, e.g.:
        # ref_fp = ref_fp_df.loc["reference"]
        # But typically you'll have a single row:
        ref_index = ref_fp_df.index[0]
        # print(ref_index)
        ref_fp = ref_fp_df.loc[ref_index]  # This is a Series with the same multi-index columns
        # print(ref_fp)
        # Prepare a list to hold comparison stats
        comparison_rows = []

        for pose_label, row in method_fp.iterrows():
            # row is a boolean Series for that pose
            # ref_fp is a boolean Series for the reference

            # missing: true in reference, false in pose
            missing_mask = ref_fp & ~row
            # print(missing_mask)
            # extra: false in reference, true in pose
            extra_mask = ~ref_fp & row
            # common: true in both
            common_mask = ref_fp & row

            comparison_info = {
                'pose': pose_label,
                'num_missing': missing_mask.sum(),
                'num_extra': extra_mask.sum(),
                'num_common': common_mask.sum(),
                'total_in_ref': ref_fp.sum(),
                'total_in_pose': row.sum(),
            }
            comparison_rows.append(comparison_info)

        comparison_df = pd.DataFrame(comparison_rows)
        return comparison_df

    def compare_with_reference_detailed(self, method_name: str, ref_fp_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compare the interaction fingerprint of each pose (for a given method)
        to a single-row reference fingerprint (e.g., a crystal pose),
        returning two DataFrames that show how many interactions of each type
        are 'missing' or 'extra' for each pose.

        Returns:
            missing_df: A DataFrame indexed by pose label, with columns for each
                        interaction type, containing how many times that interaction
                        is missing compared to the reference.
            extra_df:   Same structure, showing how many times that interaction
                        is extra in the pose (i.e., not present in the reference).
        """
        # 1) Ensure we have the method's fingerprint
        if method_name not in self.results:
            self.analyze_method(method_name)
        method_fp = self.results[method_name]  # A boolean DataFrame

        # 2) The reference fingerprint should have exactly one row (e.g., index "reference")
        ref_index = ref_fp_df.index[0]
        ref_fp = ref_fp_df.loc[ref_index]  # boolean Series for the reference

        # 3) Gather all interaction types from the multi-index columns
        #    Typically the columns look like (ligand, protein, interaction).
        #    We'll group by the "interaction" level.
        interaction_types = method_fp.columns.get_level_values("interaction").unique()

        # 4) Prepare empty DataFrames to store counts
        #    Rows = each pose, Columns = each interaction type
        missing_df = pd.DataFrame(0, index=method_fp.index, columns=interaction_types)
        extra_df = pd.DataFrame(0, index=method_fp.index, columns=interaction_types)

        # 5) Compare each pose with the reference row
        for pose_label, pose_fp in method_fp.iterrows():
            # Boolean masks
            missing_mask = ref_fp & ~pose_fp  # true in reference, false in pose
            extra_mask   = ~ref_fp & pose_fp  # false in reference, true in pose

            # Convert True/False to 1/0, then sum by interaction type
            missing_counts = missing_mask.astype(int).groupby(level="interaction").sum()
            extra_counts   = extra_mask.astype(int).groupby(level="interaction").sum()

            # Fill the row in missing_df and extra_df
            for i_type, i_count in missing_counts.items():
                missing_df.at[pose_label, i_type] = i_count
            for i_type, i_count in extra_counts.items():
                extra_df.at[pose_label, i_type] = i_count

        return missing_df, extra_df