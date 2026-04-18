#!/usr/bin/env python3
"""
Robust RMSD Calculator for Protein-Ligand Complexes

This module provides functions to compute RMSD between predicted and reference
protein-ligand complexes after proper structural alignment. It handles common
issues like substructure mismatches and different atom counts.

Author: Generated for PoseBench project
Date: June 26, 2025
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def superimpose_structures(mobile_coords: np.ndarray, target_coords: np.ndarray) -> tuple:
    """
    Superimpose mobile coordinates onto target coordinates using Kabsch algorithm.
    
    Args:
        mobile_coords: Nx3 array of mobile structure coordinates
        target_coords: Nx3 array of target structure coordinates
        
    Returns:
        tuple: (rotation_matrix, translation_vector, rmsd_after_superposition)
    """
    # Center both coordinate sets
    mobile_center = np.mean(mobile_coords, axis=0)
    target_center = np.mean(target_coords, axis=0)
    
    mobile_centered = mobile_coords - mobile_center
    target_centered = target_coords - target_center
    
    # Compute covariance matrix
    H = mobile_centered.T @ target_centered
    
    # SVD decomposition
    U, _, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = target_center - R @ mobile_center
    
    # Apply transformation and compute RMSD
    mobile_transformed = (R @ mobile_coords.T).T + t
    rmsd = np.sqrt(np.mean(np.sum((mobile_transformed - target_coords)**2, axis=1)))
    
    return R, t, rmsd


def apply_transformation_to_ligand(ligand_sdf_path: str, rotation_matrix: np.ndarray, 
                                 translation_vector: np.ndarray, output_path: str) -> None:
    """
    Apply rotation and translation transformation to ligand coordinates.
    
    Args:
        ligand_sdf_path: Path to input ligand SDF file
        rotation_matrix: 3x3 rotation matrix
        translation_vector: 3x1 translation vector
        output_path: Path to save transformed ligand
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolTransforms
    
    # Load ligand
    supplier = Chem.SDMolSupplier(ligand_sdf_path, removeHs=False)
    mol = supplier[0]
    
    if mol is None:
        raise ValueError(f"Could not load molecule from {ligand_sdf_path}")
    
    # Get conformer
    conf = mol.GetConformer()
    
    # Apply transformation to each atom
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        pos_array = np.array([pos.x, pos.y, pos.z])
        
        # Apply rotation and translation
        new_pos = rotation_matrix @ pos_array + translation_vector
        
        # Update atom position
        conf.SetAtomPosition(i, (float(new_pos[0]), float(new_pos[1]), float(new_pos[2])))
    
    # Write transformed molecule
    writer = Chem.SDWriter(output_path)
    writer.write(mol)
    writer.close()

def compute_rmsd_after_alignment(
    pred_complex_pdb: str, 
    ref_protein_pdb: str, 
    ref_ligand_sdf: str, 
    output_dir: str = "/tmp", 
    complex_id: str = "test",
    cleanup_temp_files: bool = True
) -> Dict:
    """
    Compute RMSD between predicted and reference ligands after proper structural superposition.
    
    This method:
    1. Extracts protein and ligands from predicted complex
    2. Superimposes predicted protein to reference protein using C-alpha atoms (Kabsch algorithm)
    3. Applies same transformation matrix to predicted ligand coordinates
    4. Computes RMSD between superimposed predicted ligand and reference ligand
    
    The key improvement is ensuring both structures are in the same coordinate system
    before RMSD calculation through proper 3D superposition.
    
    Args:
        pred_complex_pdb: Path to predicted complex PDB file
        ref_protein_pdb: Path to reference protein PDB file  
        ref_ligand_sdf: Path to reference ligand SDF file
        output_dir: Directory for temporary files
        complex_id: Identifier for this complex
        cleanup_temp_files: Whether to remove temporary files after computation
        
    Returns:
        dict: Contains protein_rmsd, ligand_rmsd, ligand_rmsd_kabsch, success status, and error info
    """
    # Import required modules (done inside function to avoid import errors if not available)
    try:
        from prody import parsePDB
        from rdkit import Chem
        from rdkit.Chem import rdMolAlign
        from dockstrat.utils.data_utils import extract_protein_and_ligands_with_prody
        from dockstrat.analysis.complex_alignment import save_aligned_complex
    except ImportError as e:
        return {
            'protein_rmsd': None,
            'ligand_rmsd': None, 
            'ligand_rmsd_kabsch': None,
            'success': False,
            'error': f'Import error: {str(e)}'
        }
    
    results = {
        'protein_rmsd': None,
        'ligand_rmsd': None, 
        'ligand_rmsd_kabsch': None,
        'success': False,
        'error': None
    }
    
    # Create list to track temporary files for cleanup
    temp_files = []
    
    try:
        # Create unique temp files for this complex
        pred_protein_path = os.path.join(output_dir, f"{complex_id}_pred_protein.pdb")
        pred_ligand_path = os.path.join(output_dir, f"{complex_id}_pred_ligand.sdf")
        superimposed_ligand_path = os.path.join(output_dir, f"{complex_id}_pred_ligand_superimposed.sdf")
        
        temp_files.extend([pred_protein_path, pred_ligand_path, superimposed_ligand_path])
        
        # Step 1: Extract protein and ligands from predicted complex
        logger.info(f"Extracting protein and ligands from {pred_complex_pdb}")
        extract_protein_and_ligands_with_prody(
            input_pdb_file=pred_complex_pdb,
            protein_output_pdb_file=pred_protein_path,
            ligands_output_sdf_file=pred_ligand_path,
        )
        
        # Verify extraction worked
        if not os.path.exists(pred_protein_path) or not os.path.exists(pred_ligand_path):
            raise FileNotFoundError("Failed to extract protein or ligand from predicted complex")
        
        # Step 2: Load protein structures and extract C-alpha coordinates
        logger.info(f"Loading protein structures for superposition")
        pred_prot = parsePDB(pred_protein_path)
        ref_prot = parsePDB(ref_protein_pdb)
        
        # Select C-alpha atoms for superposition
        pred_ca = pred_prot.select("name CA")
        ref_ca = ref_prot.select("name CA")
        
        if pred_ca is None or ref_ca is None:
            raise ValueError("Could not select C-alpha atoms from proteins")
        
        # Handle different sequence lengths
        if len(pred_ca) != len(ref_ca):
            logger.warning(f"Different number of CA atoms - pred: {len(pred_ca)}, ref: {len(ref_ca)}")
            min_len = min(len(pred_ca), len(ref_ca))
            pred_ca_coords = pred_ca.getCoords()[:min_len]
            ref_ca_coords = ref_ca.getCoords()[:min_len]
        else:
            pred_ca_coords = pred_ca.getCoords()
            ref_ca_coords = ref_ca.getCoords()
        
        # Step 3: Superimpose predicted protein onto reference protein
        logger.info(f"Performing 3D superposition using Kabsch algorithm")
        rotation_matrix, translation_vector, protein_rmsd = superimpose_structures(
            pred_ca_coords, ref_ca_coords
        )
        
        results['protein_rmsd'] = protein_rmsd
        logger.info(f"Protein superposition RMSD: {protein_rmsd:.3f} Å")
        
        # Step 4: Apply the same transformation to the predicted ligand
        logger.info(f"Applying transformation to predicted ligand")
        apply_transformation_to_ligand(
            pred_ligand_path, rotation_matrix, translation_vector, superimposed_ligand_path
        )
        
        # Step 5: Compute ligand RMSD after superposition
        pred_lig = Chem.SDMolSupplier(superimposed_ligand_path, removeHs=False)[0]
        ref_lig = Chem.SDMolSupplier(ref_ligand_sdf, removeHs=False)[0]
        
        if pred_lig is None or ref_lig is None:
            raise ValueError("Could not load ligand molecules")
        
        # Method 1: Direct coordinate comparison (after superposition)
        try:
            pred_conf = pred_lig.GetConformer()
            ref_conf = ref_lig.GetConformer()
            
            # Handle case where molecules have different number of atoms
            if pred_lig.GetNumAtoms() != ref_lig.GetNumAtoms():
                logger.warning(f"Different number of atoms - pred: {pred_lig.GetNumAtoms()}, ref: {ref_lig.GetNumAtoms()}")
                # For direct comparison, we need same number of atoms
                ligand_rmsd = None
            else:
                pred_coords = np.array(pred_conf.GetPositions())
                ref_coords = np.array(ref_conf.GetPositions())
                ligand_rmsd = np.sqrt(np.mean(np.sum((pred_coords - ref_coords)**2, axis=1)))
                
            results['ligand_rmsd'] = ligand_rmsd
            
        except Exception as e:
            logger.warning(f"Direct ligand RMSD failed: {e}")
            results['ligand_rmsd'] = None
        
        # Method 2: Kabsch alignment for ligands (handles substructure mismatches)
        try:
            ligand_rmsd_kabsch = rdMolAlign.GetBestRMS(pred_lig, ref_lig)
            results['ligand_rmsd_kabsch'] = ligand_rmsd_kabsch
        except Exception as e:
            logger.warning(f"Kabsch ligand RMSD failed: {e}")
            # Try with different options
            try:
                ligand_rmsd_kabsch = rdMolAlign.CalcRMS(pred_lig, ref_lig)
                results['ligand_rmsd_kabsch'] = ligand_rmsd_kabsch
            except Exception as e2:
                logger.warning(f"Alternative ligand RMSD also failed: {e2}")
                results['ligand_rmsd_kabsch'] = None
        
        results['success'] = True
        logger.info(f"Successfully computed RMSD for {complex_id}")
        
    except Exception as e:
        results['error'] = str(e)
        logger.error(f"Error in RMSD computation for {complex_id}: {e}")
    
    finally:
        # Clean up temporary files
        if cleanup_temp_files:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        logger.warning(f"Could not remove temp file {temp_file}: {e}")
    
    return results


def batch_compute_rmsd(
    pred_dir: Union[str, Path],
    data_dir: Union[str, Path],
    output_csv: Optional[str] = None,
    output_dir: str = "/tmp",
    pattern_prefix: str = "boltz_results_"
) -> pd.DataFrame:
    """
    Batch process multiple predicted complexes to compute RMSD.
    
    Args:
        pred_dir: Directory containing prediction subdirectories
        data_dir: Directory containing reference data
        output_csv: Path to save results CSV (optional)
        output_dir: Directory for temporary files
        pattern_prefix: Prefix pattern for prediction directories
        
    Returns:
        pd.DataFrame: Results dataframe with RMSD calculations
    """
    pred_dir = Path(pred_dir)
    data_dir = Path(data_dir)
    
    # Get all prediction directories
    pred_subdirs = [d for d in pred_dir.iterdir() if d.is_dir() and d.name.startswith(pattern_prefix)]
    
    logger.info(f"Found {len(pred_subdirs)} prediction directories")
    
    # Store results
    all_results = []
    
    for i, subdir in enumerate(pred_subdirs, 1):
        # Extract complex ID from directory name
        complex_id = subdir.name.replace(pattern_prefix, "")
        
        logger.info(f"Processing {i}/{len(pred_subdirs)}: {complex_id}")
        
        # Define paths
        pred_complex_pdb = subdir / "predictions" / complex_id / f"{complex_id}_model_0.pdb"
        ref_protein_pdb = data_dir / complex_id / f"{complex_id}_protein.pdb"
        ref_ligand_sdf = data_dir / complex_id / f"{complex_id}_ligand.sdf"
        
        # Check if all required files exist
        if not all(path.exists() for path in [pred_complex_pdb, ref_protein_pdb, ref_ligand_sdf]):
            logger.warning(f"Skipping {complex_id}: missing files")
            all_results.append({
                'complex_id': complex_id,
                'protein_rmsd': None,
                'ligand_rmsd': None,
                'ligand_rmsd_kabsch': None,
                'success': False,
                'error': 'Missing files'
            })
            continue
        
        # Compute RMSD using the new method
        results = compute_rmsd_after_alignment(
            pred_complex_pdb=str(pred_complex_pdb),
            ref_protein_pdb=str(ref_protein_pdb),
            ref_ligand_sdf=str(ref_ligand_sdf),
            output_dir=output_dir,
            complex_id=complex_id
        )
        
        # Add complex ID to results
        results['complex_id'] = complex_id
        all_results.append(results)
        
        # Print summary for this complex
        if results['success']:
            protein_rmsd = results['protein_rmsd']
            ligand_rmsd_kabsch = results['ligand_rmsd_kabsch']
            logger.info(f"  ✓ Protein RMSD: {protein_rmsd:.3f} Å")
            if ligand_rmsd_kabsch is not None:
                logger.info(f"  ✓ Ligand RMSD: {ligand_rmsd_kabsch:.3f} Å")
            else:
                logger.warning("  ⚠ Ligand RMSD: Could not compute")
        else:
            logger.error(f"  ✗ Failed: {results['error']}")

    # Create DataFrame with results
    df_results = pd.DataFrame(all_results)
    
    # Print summary statistics
    print_summary_statistics(df_results)
    
    # Save to CSV if requested
    if output_csv:
        save_results_to_csv(df_results, output_csv)
    
    return df_results


def print_summary_statistics(df_results: pd.DataFrame) -> None:
    """Print summary statistics for the RMSD results."""
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Total complexes processed: {len(df_results)}")
    print(f"Successful calculations: {df_results['success'].sum()}")
    print(f"Failed calculations: {(~df_results['success']).sum()}")

    # Show some statistics for successful calculations
    successful_df = df_results[df_results['success']]
    if len(successful_df) > 0:
        print(f"\nProtein RMSD statistics:")
        print(f"  Mean: {successful_df['protein_rmsd'].mean():.3f} Å")
        print(f"  Median: {successful_df['protein_rmsd'].median():.3f} Å")
        print(f"  Min: {successful_df['protein_rmsd'].min():.3f} Å")
        print(f"  Max: {successful_df['protein_rmsd'].max():.3f} Å")
        
        # Ligand RMSD statistics (only for cases where it was computed)
        ligand_rmsd_data = successful_df['ligand_rmsd_kabsch'].dropna()
        if len(ligand_rmsd_data) > 0:
            print(f"\nLigand RMSD statistics:")
            print(f"  Mean: {ligand_rmsd_data.mean():.3f} Å")
            print(f"  Median: {ligand_rmsd_data.median():.3f} Å")
            print(f"  Min: {ligand_rmsd_data.min():.3f} Å")
            print(f"  Max: {ligand_rmsd_data.max():.3f} Å")
            print(f"  Computed for {len(ligand_rmsd_data)}/{len(successful_df)} complexes")


def save_results_to_csv(df_results: pd.DataFrame, output_csv: str) -> None:
    """Save results to CSV files."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Create a clean DataFrame for saving
    save_df = df_results[['complex_id', 'protein_rmsd', 'ligand_rmsd', 'ligand_rmsd_kabsch', 'success', 'error']].copy()
    
    # Save to CSV
    save_df.to_csv(output_csv, index=False)
    logger.info(f"Results saved to: {output_csv}")
    
    # Also save a summary of only successful results
    successful_save_df = save_df[save_df['success']].drop(['success', 'error'], axis=1)
    successful_csv = output_csv.replace('.csv', '_successful.csv')
    successful_save_df.to_csv(successful_csv, index=False)
    logger.info(f"Successful results saved to: {successful_csv}")
    
    print(f"\nFiles created:")
    print(f"  - {output_csv} (all results, {len(save_df)} rows)")
    print(f"  - {successful_csv} (successful only, {len(successful_save_df)} rows)")


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute robust RMSD for protein-ligand complexes")
    parser.add_argument("--pred_dir", type=str, required=True, 
                       help="Directory containing prediction results")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing reference data")
    parser.add_argument("--output_csv", type=str, default="rmsd_results.csv",
                       help="Output CSV file path")
    parser.add_argument("--output_dir", type=str, default="/tmp",
                       help="Directory for temporary files")
    parser.add_argument("--pattern_prefix", type=str, default="boltz_results_",
                       help="Prefix pattern for prediction directories")
    parser.add_argument("--single_complex", type=str, default=None,
                       help="Process only a single complex (provide complex_id)")
    
    args = parser.parse_args()
    
    if args.single_complex:
        # Process single complex
        complex_id = args.single_complex
        pred_complex_pdb = os.path.join(args.pred_dir, f"{args.pattern_prefix}{complex_id}", 
                                       "predictions", complex_id, f"{complex_id}_model_0.pdb")
        ref_protein_pdb = os.path.join(args.data_dir, complex_id, f"{complex_id}_protein.pdb")
        ref_ligand_sdf = os.path.join(args.data_dir, complex_id, f"{complex_id}_ligand.sdf")
        
        results = compute_rmsd_after_alignment(
            pred_complex_pdb=pred_complex_pdb,
            ref_protein_pdb=ref_protein_pdb,
            ref_ligand_sdf=ref_ligand_sdf,
            output_dir=args.output_dir,
            complex_id=complex_id
        )
        
        print(f"\nResults for {complex_id}:")
        print(f"Success: {results['success']}")
        if results['success']:
            print(f"Protein RMSD: {results['protein_rmsd']:.3f} Å")
            if results['ligand_rmsd_kabsch'] is not None:
                print(f"Ligand RMSD: {results['ligand_rmsd_kabsch']:.3f} Å")
        else:
            print(f"Error: {results['error']}")
    else:
        # Batch process
        df_results = batch_compute_rmsd(
            pred_dir=args.pred_dir,
            data_dir=args.data_dir,
            output_csv=args.output_csv,
            output_dir=args.output_dir,
            pattern_prefix=args.pattern_prefix
        )
        
        print(f"\nProcessed {len(df_results)} complexes")


if __name__ == "__main__":
    main()
