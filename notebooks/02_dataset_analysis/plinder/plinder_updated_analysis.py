import pandas as pd
import numpy as np
import ast
from typing import List, Dict
import os
import matplotlib.pyplot as plt
import seaborn as sns
from plinder_analysis_utils import DockingAnalysisBase, PoseBustersAnalysis, PropertyAnalysis

PLINDER_TEST_COLUMNS = [
    "system_id", "ligand_smiles",
    # binary 
    "ligand_is_covalent", "ligand_is_ion", "ligand_is_cofactor", "ligand_is_artifact",
    # discrete
    "system_num_protein_chains",
    "ligand_num_rot_bonds", "ligand_num_hbd", "ligand_num_hba", "ligand_num_rings",
    # continuous
    "entry_resolution", "entry_validation_molprobity", 
    "system_num_pocket_residues", "system_num_interactions",
    "ligand_molecular_weight", "ligand_crippen_clogp", 
    "ligand_num_interacting_residues", "ligand_num_neighboring_residues", "ligand_num_interactions",
]
# Create category mapping for visualization
CATEGORY_MAPPING = {
    "ligand_is_covalent": "binary",
    "ligand_is_ion": "binary",
    "ligand_is_cofactor": "binary",
    "ligand_is_artifact": "binary",
    "system_num_protein_chains": "discrete",
    "ligand_num_rot_bonds": "continuous",    
    "ligand_num_hbd": "continuous",
    "ligand_num_hba": "continuous",
    "ligand_num_rings": "discrete",
    "entry_resolution": "continuous",
    "entry_validation_molprobity": "continuous",
    "system_num_pocket_residues": "continuous",
    "system_num_interactions": "continuous",
    "ligand_molecular_weight": "continuous",
    "ligand_crippen_clogp": "continuous",
    "ligand_num_interacting_residues": "continuous",
    "ligand_num_neighboring_residues": "continuous",
    "ligand_num_interactions": "continuous",
    "ligand_is_artifact": "binary"     
}

MIXED_EFFECT_VARS = [
    "protein", "rmsd","method",
    # "system_id", "ligand_smiles",
    # binary 
    # "ligand_is_covalent", "ligand_is_ion", "ligand_is_cofactor", "ligand_is_artifact",
    # discrete
    # "system_num_protein_chains",
    "ligand_num_rot_bonds", "ligand_num_hbd", "ligand_num_hba", "ligand_num_rings",
    # continuous
    "entry_resolution", "entry_validation_molprobity", 
    # "system_num_pocket_residues", 
    "system_num_interactions",
    "ligand_molecular_weight", "ligand_crippen_clogp", 
    "ligand_num_interacting_residues", 
    "ligand_num_neighboring_residues", 
    # "ligand_num_interactions",
]

def universal_analysis(df_combined, prop, prop_type):
    property_analysis = PropertyAnalysis(df_combined)
    # First, analyze a property universally
    universal_result = property_analysis.analyze_property_universal(
        methods=methods,
        property_name=prop,
        property_type=prop_type,
        plot=False  # Don't generate plots yet
    )

    # Then analyze the same property for each individual method
    method_results = {}
    methods = ["icm", "gnina", "surfdock", "diffdock_pocket_only", "chai-1"]
    for method in methods:
        method_results[method] = property_analysis.analyze_property_single_method(
            method=method,
            property_name=prop,
            property_type=prop_type,
            plot=True  # Don't generate plots yet
        )

    # Now plot the p-value comparison
    fig = property_analysis.plot_property_pvalue_comparison(
        property_name=prop,
        property_type=prop_type,
        universal_result=universal_result,
        method_results=method_results,
        log_scale=True  # Use log scale for better visibility
    )

    fig.show()

def comparative_analysis(df_combined, method_1_group: List[str], method_2_group: List[str], prop: str, prop_type: str):
    property_analysis = PropertyAnalysis(df_combined)
    property_analysis.analyze_property_group_comparative(
        group1_methods=method_1_group,
        group2_methods=method_2_group,
        property_name=prop,
        property_type=prop_type,
        plot=True
    )
    print("Comparative analysis complete.") 

def comparative_mixed_effect(df_combined, property: str):
    df_mixed = df_combined[MIXED_EFFECT_VARS]
    property_analysis = PropertyAnalysis(df_combined)

    # Create a Method_Type column based on the classification
    df_mixed['Method_Type'] = df_mixed['method'].apply(
        lambda x: 'ML' if x in ['chai-1', 'diffdock_pocket_only', 'surfdock'] else 'Physics'
    )
    df_mixed = (
        df_mixed[[
            "rmsd",
            property,
            "Method_Type",
            "protein"       # or "system_id", whichever you’re grouping by
        ]]
        .dropna()
        .reset_index(drop=True)    # <<-- important!
    )

    # property_analysis.comparative_mixed_effect(
    #     df_mixed=df_mixed,
    #     property=property
    # )
    results = property_analysis.analyze_continous_property_mixed_effects(
        df_combined[df_combined["method"].isin(["chai-1", "diffdock_pocket_only", "surfdock"])],
        df_combined[df_combined["method"].isin(["icm", "gnina", "vina"])],
        property_name="ligand_num_interacting_residues", 
        group1_label="ML",
        group2_label="Physics",
        best_rmsd=True,
    )
    print(results['model_full'].summary())
    print(results['model_noint'].summary())


if __name__ == "__main__":
    # Load the Processed DataFrame
    df_combined = pd.read_csv('./notebooks/plinder_test_merged.csv')

    property_analysis = PropertyAnalysis(df_combined)

    # # 4. Look at universal patterns across all methods
    # for prop in PLINDER_TEST_COLUMNS[6:]:
    #     # 2. Explore impact of properties on a single method
    #     prop_type = CATEGORY_MAPPING[prop]
    #     prop_analysis = property_analysis.analyze_property_universal(
    #         ["icm", "gnina", "surfdock", "diffdock_pocket_only", "chai-1"], prop, prop_type,
    #     )

    # for prop in PLINDER_TEST_COLUMNS[2:]:
    #     prop_type = CATEGORY_MAPPING[prop]
    #     prop_analysis = property_analysis.analyze_property_single_method(
    #         "surfdock", prop, prop_type,
    #     )
    
    df_combined = pd.read_csv("./notebooks/plinder_set_0_annotated.csv")
    # First analyze multiple properties
    
    # Comparative 
    # methods = ["surfdock", "gnina", "chai-1", "diffdock_pocket_only", "icm", "vina"]
    # methods = ["surfdock", "gnina"]
    # comparative_analysis(df_combined, ["surfdock"], ["gnina"], "ligand_num_interacting_residues", "continuous")

    # comparative_mixed_effect(df_combined, "ligand_num_interacting_residues")

    # Complementary 
    # comp = property_analysis.complementary_success_analysis(
    #     method1="diffdock_pocket_only",
    #     method2="vina",
    #     rmsd_threshold=2.0
    # )

    # print("McNemar p =", comp['mcnemar_p'])
    # print("#DiffDock-only successes:", len(comp['list_m1succ_m2fail']))
    # print("#Vina-only successes   :", len(comp['list_m1fail_m2succ']))

    # # stratified_analysis = property_analysis.stratified_success_analysis(
    # stratified_analysis = property_analysis.stratified_mcnemar_analysis(
    #     methods_group1="diffdock_pocket_only",
    #     methods_group2="vina",
    #     rmsd_threshold=2.0,
    #     property_name="ligand_num_interacting_residues"
    # )

    # print("Stratified McNemar p =", stratified_analysis['cmh'])

    # stratified_analysis = property_analysis.stratified_mcnemar_analysis(
    #     methods_group1=["diffdock_pocket_only","surfdock", "chai-1"],
    #     methods_group2=["vina", "gnina", "icm"],
    #     rmsd_threshold=2.0,
    #     property_name="ligand_num_interacting_residues",
    # )
    
    # print("Stratified McNemar p =", stratified_analysis['cmh'])

    # # pb analysis
    analysis = PoseBustersAnalysis(df_combined)
    # # Analyze a single method with default RMSD threshold of 2.0Å
    # single_method_results = analysis.analyze_single_method(
    #     method="diffdock_pocket_only",
    #     rmsd_threshold=2.0,
    #     imbalance_threshold=0.05,
    #     plot=True  # Generate visualization plots
    # )

    # # Access the results
    # success_rate = single_method_results['overall_success_rate']
    # significant_metrics = single_method_results['significant_metrics']

    # PB Mixed_effect Analysis
    # Compare ML-based vs physics-based methods
    result = analysis.mixed_effect_analysis(
        filter_name='minimum_distance_to_protein',
        method_groups={
            'ML-based': ['diffdock_pocket_only', 'chai-1', 'surfdock'], 
            'Physics-based': ['vina', 'gnina', 'icm']
        },
        rmsd_threshold=2.0,
        outcome_type='rmsd',
        plot=True
    )

    # Access model and results
    print(result)
    # print("Model:", result['model'])
    # print("Interaction p-value:", result['p_interaction'])