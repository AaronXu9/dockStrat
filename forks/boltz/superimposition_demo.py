#!/usr/bin/env python3
"""
Demonstration of 3D Structural Superimposition for RMSD Calculation

This script demonstrates the importance and implementation of 3D structural 
superimposition before computing RMSD between predicted and reference 
protein-ligand complexes.

Usage:
    python superimposition_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from robust_rmsd_calculator import superimpose_structures, compute_rmsd_after_alignment


def demonstrate_superimposition_concept():
    """
    Demonstrate the concept of superimposition with synthetic data
    """
    print("=" * 60)
    print("3D STRUCTURAL SUPERIMPOSITION DEMONSTRATION")
    print("=" * 60)
    
    # Create synthetic protein backbone (C-alpha positions)
    n_residues = 50
    t = np.linspace(0, 4*np.pi, n_residues)
    
    # Reference structure (spiral)
    ref_coords = np.column_stack([
        5 * np.cos(t),
        5 * np.sin(t),
        t
    ])
    
    # Create a "predicted" structure by applying rotation and translation
    # This simulates a structure in a different coordinate system
    angle = np.pi/6  # 30 degrees
    rotation_true = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    translation_true = np.array([10, 5, -3])
    
    # Add some noise to simulate prediction errors
    noise = np.random.normal(0, 0.2, ref_coords.shape)
    pred_coords = (rotation_true @ ref_coords.T).T + translation_true + noise
    
    print(f"Created synthetic protein with {n_residues} C-alpha atoms")
    print(f"Applied rotation and translation to simulate coordinate system difference")
    
    # Calculate centers before superimposition
    ref_center = np.mean(ref_coords, axis=0)
    pred_center_before = np.mean(pred_coords, axis=0)
    dist_before = np.linalg.norm(pred_center_before - ref_center)
    
    print(f"\nBefore superimposition:")
    print(f"Reference center: [{ref_center[0]:.2f}, {ref_center[1]:.2f}, {ref_center[2]:.2f}]")
    print(f"Predicted center:  [{pred_center_before[0]:.2f}, {pred_center_before[1]:.2f}, {pred_center_before[2]:.2f}]")
    print(f"Distance between centers: {dist_before:.2f} Å")
    
    # Perform superimposition
    print(f"\nPerforming Kabsch superimposition...")
    rotation_matrix, translation_vector, rmsd_super = superimpose_structures(
        pred_coords, ref_coords
    )
    
    # Apply transformation
    pred_aligned = (rotation_matrix @ pred_coords.T).T + translation_vector
    pred_center_after = np.mean(pred_aligned, axis=0)
    dist_after = np.linalg.norm(pred_center_after - ref_center)
    
    print(f"\nAfter superimposition:")
    print(f"Reference center: [{ref_center[0]:.2f}, {ref_center[1]:.2f}, {ref_center[2]:.2f}]")
    print(f"Predicted center:  [{pred_center_after[0]:.2f}, {pred_center_after[1]:.2f}, {pred_center_after[2]:.2f}]")
    print(f"Distance between centers: {dist_after:.2f} Å")
    print(f"RMSD after superimposition: {rmsd_super:.3f} Å")
    
    # Calculate RMSD without superimposition for comparison
    rmsd_no_super = np.sqrt(np.mean(np.sum((pred_coords - ref_coords)**2, axis=1)))
    print(f"\nRMSD without superimposition: {rmsd_no_super:.3f} Å")
    print(f"RMSD with superimposition:    {rmsd_super:.3f} Å")
    print(f"Improvement: {rmsd_no_super - rmsd_super:.3f} Å ({(rmsd_no_super-rmsd_super)/rmsd_no_super*100:.1f}%)")
    
    # Visualization
    fig = plt.figure(figsize=(15, 5))
    
    # Before superimposition
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(ref_coords[:, 0], ref_coords[:, 1], ref_coords[:, 2], 
             'b-o', label='Reference', markersize=3, alpha=0.7)
    ax1.plot(pred_coords[:, 0], pred_coords[:, 1], pred_coords[:, 2], 
             'r-o', label='Predicted', markersize=3, alpha=0.7)
    ax1.set_title('Before Superimposition')
    ax1.legend()
    ax1.set_xlabel('X (Å)')
    ax1.set_ylabel('Y (Å)')
    try:
        ax1.set_zlabel('Z (Å)')
    except:
        pass  # Some matplotlib versions don't support set_zlabel
    
    # After superimposition
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot(ref_coords[:, 0], ref_coords[:, 1], ref_coords[:, 2], 
             'b-o', label='Reference', markersize=3, alpha=0.7)
    ax2.plot(pred_aligned[:, 0], pred_aligned[:, 1], pred_aligned[:, 2], 
             'r-o', label='Predicted (aligned)', markersize=3, alpha=0.7)
    ax2.set_title('After Superimposition')
    ax2.legend()
    ax2.set_xlabel('X (Å)')
    ax2.set_ylabel('Y (Å)')
    try:
        ax2.set_zlabel('Z (Å)')
    except:
        pass  # Some matplotlib versions don't support set_zlabel
    
    # Per-residue RMSD
    ax3 = fig.add_subplot(133)
    residue_rmsd = np.sqrt(np.sum((pred_aligned - ref_coords)**2, axis=1))
    ax3.plot(residue_rmsd, 'g-', linewidth=2)
    ax3.set_title('Per-Residue RMSD\n(After Superimposition)')
    ax3.set_xlabel('Residue Index')
    ax3.set_ylabel('RMSD (Å)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=np.mean(residue_rmsd), color='r', linestyle='--', 
                label=f'Mean: {np.mean(residue_rmsd):.3f} Å')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('superimposition_demo.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'superimposition_demo.png'")
    plt.show()
    
    return rotation_matrix, translation_vector, rmsd_super, rmsd_no_super


def explain_why_superimposition_matters():
    """
    Explain the importance of superimposition in structural analysis
    """
    print("\n" + "=" * 60)
    print("WHY SUPERIMPOSITION IS CRITICAL FOR RMSD CALCULATION")
    print("=" * 60)
    
    explanations = [
        "1. COORDINATE SYSTEM INDEPENDENCE:",
        "   - Predicted and reference structures often have different origins",
        "   - Rotation and translation differences don't reflect prediction quality",
        "   - Superimposition removes these arbitrary differences",
        "",
        "2. MEANINGFUL STRUCTURAL COMPARISON:",
        "   - RMSD should measure shape/conformation differences, not orientation",
        "   - Protein backbone alignment ensures ligand comparison is meaningful",
        "   - Same transformation applied to ligand maintains relative geometry",
        "",
        "3. STANDARDIZED BENCHMARKING:",
        "   - Enables fair comparison between different prediction methods",
        "   - Removes bias from different coordinate system conventions",
        "   - Provides reproducible and interpretable metrics",
        "",
        "4. BIOLOGICAL RELEVANCE:",
        "   - Binding affinity depends on relative protein-ligand geometry",
        "   - Absolute coordinates in space are not biologically meaningful",
        "   - Superimposition focuses on biologically relevant features"
    ]
    
    for line in explanations:
        print(line)


def demonstrate_real_example():
    """
    Demonstrate with real PDB structures if available
    """
    print("\n" + "=" * 60)
    print("REAL STRUCTURE EXAMPLE")
    print("=" * 60)
    
    # This would demonstrate with actual Boltz results
    example_paths = {
        'pred_complex': '/home/aoxu/projects/PoseBench/forks/boltz/boltz_results_7PK0_BYC/7PK0_BYC_holo_aligned_esmfold.pdb',
        'ref_protein': '/home/aoxu/projects/PoseBench/data/test_cases/7PK0_BYC/7PK0_protein.pdb',
        'ref_ligand': '/home/aoxu/projects/PoseBench/data/test_cases/7PK0_BYC/7PK0_BYC_ligand.sdf'
    }
    
    # Check if files exist
    import os
    if all(os.path.exists(path) for path in example_paths.values()):
        print("Found real structure example - computing RMSD with superimposition...")
        
        results = compute_rmsd_after_alignment(
            pred_complex_pdb=example_paths['pred_complex'],
            ref_protein_pdb=example_paths['ref_protein'],
            ref_ligand_sdf=example_paths['ref_ligand'],
            complex_id="7PK0_BYC_demo",
            cleanup_temp_files=True
        )
        
        print(f"\nReal structure results:")
        print(f"Protein RMSD: {results.get('protein_rmsd', 'N/A'):.3f} Å")
        print(f"Ligand RMSD (direct): {results.get('ligand_rmsd', 'N/A'):.3f} Å")
        print(f"Ligand RMSD (Kabsch): {results.get('ligand_rmsd_kabsch', 'N/A'):.3f} Å")
        print(f"Success: {results.get('success', False)}")
        
        if results.get('error'):
            print(f"Error: {results['error']}")
            
    else:
        print("Real structure files not found - skipping real example")
        print("Example paths checked:")
        for name, path in example_paths.items():
            exists = "✅" if os.path.exists(path) else "❌"
            print(f"  {exists} {name}: {path}")


if __name__ == "__main__":
    # Run demonstrations
    try:
        # Synthetic demonstration
        demonstrate_superimposition_concept()
        
        # Explain importance
        explain_why_superimposition_matters()
        
        # Real example if available
        demonstrate_real_example()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("Key takeaways:")
        print("- Superimposition is essential for meaningful RMSD comparison")
        print("- Kabsch algorithm provides optimal structural alignment")
        print("- Protein backbone alignment guides ligand comparison")
        print("- Results are more biologically relevant and interpretable")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all required packages are installed:")
        print("- numpy, matplotlib, rdkit, prody")
    except Exception as e:
        print(f"Error in demonstration: {e}")
