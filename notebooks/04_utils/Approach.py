import os
import re
from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
from rdkit import Chem
from posebusters.posebusters import PoseBusters


class DockingApproach(ABC):
    @abstractmethod
    def get_name(self) -> str:
        """A short identifier for this method (e.g. 'icm', 'diffdock', 'chai')."""
        pass

    @abstractmethod
    def list_top_n_files(self, protein_dir: str, top_n: int) -> List[str]:
        """
        Return up to top_n SDF file paths for that protein_dir, 
        in descending or ascending 'rank' order, as appropriate.
        """
        pass

    def parse_score(self, sdf_path: str) -> float:
        """
        If this approach has a numeric score to parse, override this method.
        If there's no numeric score, return None or float('nan').
        """
        return float('nan')  # default: no score
    
class ICMApproach(DockingApproach):
    def get_name(self) -> str:
        return "icm"

    def list_top_n_files(self, protein_dir: str, top_n: int) -> List[str]:
        """
        Assume ICM wrote 'rank1.sdf', 'rank2.sdf', etc.
        We'll list all files matching 'rankX.sdf', sort by X, return top_n.
        """
        all_files = os.listdir(protein_dir)
        sdf_files = [f for f in all_files if f.startswith("rank") and f.endswith(".sdf")]
        
        def extract_rank(fname: str) -> int:
            # "rank(\d+).sdf"
            match = re.match(r"rank(\d+)\.sdf", fname)
            if match:
                return int(match.group(1))
            return 999999
        
        sdf_files.sort(key=extract_rank)
        return [os.path.join(protein_dir, f) for f in sdf_files[:top_n]]

    def parse_score(self, sdf_path: str) -> float:
        """
        ICM stores the docking score in the SDF property 'Score'.
        We'll read the single pose from the file and extract that property.
        """
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        for mol in suppl:
            if mol is not None and "Score" in mol.GetPropNames():
                return float(mol.GetProp("Score"))
        return float('nan')

class ICMRTCNNApproach(DockingApproach):
    def get_name(self) -> str:
        return "icm_rtcnn"

    def list_top_n_files(self, protein_dir: str, top_n: int) -> List[str]:
        """
        Assume ICM wrote 'rank1.sdf', 'rank2.sdf', etc.
        We'll list all files matching 'rankX.sdf', sort by X, return top_n.
        """
        all_files = os.listdir(protein_dir)
        sdf_files = [f for f in all_files if f.startswith("RTCNN_rank") and f.endswith(".sdf")]
        
        def extract_rank(fname: str) -> int:
            # "rank(\d+).sdf"
            match = re.match(r"RTCNN_rank(\d+)\.sdf", fname)
            if match:
                return int(match.group(1))
            return 999999
        
        sdf_files.sort(key=extract_rank)
        return [os.path.join(protein_dir, f) for f in sdf_files[:top_n]]

    def parse_score(self, sdf_path: str) -> float:
        """
        ICM stores the docking score in the SDF property 'Score'.
        We'll read the single pose from the file and extract that property.
        """
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        for mol in suppl:
            if mol is not None and "RTCNN" in mol.GetPropNames():
                return float(mol.GetProp("RTCNN"))
        return float('nan')

class DiffDockApproach(DockingApproach):
    def get_name(self) -> str:
        return "diffdock"

    def list_top_n_files(self, protein_dir: str, top_n: int) -> List[str]:
        """
        DiffDock outputs might be named 'rank1.sdf' or 
        'rank2_confidence-0.16.sdf', etc.
        We'll parse 'rank(\d+)' to get the rank, sort, and return top_n.
        """
        all_files = os.listdir(protein_dir)
        sdf_files = [
            f for f in all_files 
            if f.startswith("rank") and f.endswith(".sdf")
        ]

        def extract_rank(fname: str) -> int:
            match = re.search(r"rank(\d+)", fname)
            if match:
                return int(match.group(1))
            return 999999
        
        sdf_files.sort(key=extract_rank)
        return [os.path.join(protein_dir, f) for f in sdf_files[:top_n]]

    def parse_score(self, sdf_path: str) -> float:
        """
        DiffDock doesn't store 'Score' in the SDF properties.
        Instead, there's a confidence value in the filename
        like 'rank2_confidence-0.16.sdf'.
        We'll parse that out. If not present, return NaN.
        """
        fname = os.path.basename(sdf_path)
        # Look for something like _confidence-0.16
        match = re.search(r"_confidence-([\d\.]+)", fname)
        if match:
            conf_str = match.group(1)
            # Remove trailing '.' if present, e.g. "1.40." => "1.40"
            conf_str = conf_str.rstrip('.')
            try:
                return float(conf_str)
            except ValueError:
                print(f"[WARNING] Could not parse confidence '{conf_str}' from {fname}, returning NaN.")
                return float('nan')
        return float('nan')
    
class DiffDockPocketApproach(DockingApproach):
    def get_name(self) -> str:
        return "diffdock_pocket_only"

    def list_top_n_files(self, protein_dir: str, top_n: int) -> List[str]:
        """
        DiffDock doesn't store 'Score' in the SDF properties.
        Instead, there's a confidence value in the filename
        like 'rank2_confidence-0.16.sdf'.
        We'll parse that out. If not present, return NaN.
        """
        all_files = os.listdir(protein_dir)
        sdf_files = [
            f for f in all_files 
            if f.startswith("rank") and f.endswith(".sdf")
        ]

        def extract_rank(fname: str) -> int:
            match = re.search(r"rank(\d+)", fname)
            if match:
                return int(match.group(1))
            return 999999
        
        sdf_files.sort(key=extract_rank)
        return [os.path.join(protein_dir, f) for f in sdf_files[:top_n]]

    def parse_score(self, sdf_path: str) -> float:
        """
        DiffDock doesn't store 'Score' in the SDF properties.
        Instead, there's a confidence value in the filename
        like 'rank2_confidence-0.16.sdf'.
        We'll parse that out. If not present, return NaN.
        """
        fname = os.path.basename(sdf_path)
        # Look for something like _confidence-0.16
        match = re.search(r"_confidence-([\d\.]+)", fname)
        if match:
            conf_str = match.group(1)
            # Remove trailing '.' if present, e.g. "1.40." => "1.40"
            conf_str = conf_str.rstrip('.')
            try:
                return float(conf_str)
            except ValueError:
                print(f"[WARNING] Could not parse confidence '{conf_str}' from {fname}, returning NaN.")
                return float('nan')
        return float('nan')
    
class ChaiApproach(DockingApproach):
    def get_name(self) -> str:
        return "chai-1"

    def list_top_n_files(self, protein_dir: str, top_n: int) -> List[str]:
        """
        Example filenames: pred.model_idx_0_ligand_aligned.sdf,
                          pred.model_idx_1_ligand_aligned.sdf, ...
        We'll parse the model_idx to define an order. 
        """
        all_files = os.listdir(protein_dir)
        sdf_files = [
            f for f in all_files
            if f.startswith("pred.model_idx_") and f.endswith("_ligand_aligned.sdf")
        ]

        def extract_model_idx(fname: str) -> int:
            m = re.search(r"model_idx_(\d+)", fname)
            if m:
                return int(m.group(1))
            return 999999

        sdf_files.sort(key=extract_model_idx)
        return [os.path.join(protein_dir, f) for f in sdf_files[:top_n]]

    # No score or confidence stored
    def parse_score(self, sdf_path: str) -> float:
        return float('nan')
    
class VinaApproach(DockingApproach):
    def get_name(self) -> str:
        return "vina"

    def list_top_n_files(self, protein_dir: str, top_n: int) -> List[str]:
        """
        Vina outputs are named like '5SAK_ZRY_pose3_score-7.97.sdf'
        We'll parse 'pose(\d+)' to get the rank, sort, and return top_n.
        """
        all_files = os.listdir(protein_dir)
        sdf_files = [f for f in all_files if '_pose' in f and f.endswith('.sdf')]

        def extract_pose_num(fname: str) -> int:
            match = re.search(r'pose(\d+)', fname)
            if match:
                return int(match.group(1))
            return 999999
        
        sdf_files.sort(key=extract_pose_num)
        return [os.path.join(protein_dir, f) for f in sdf_files[:top_n]]


    def parse_score(self, sdf_path: str) -> float:
        """
        Score is in filename like '5SAK_ZRY_pose3_score-7.97.sdf'
        """
        fname = os.path.basename(sdf_path)
        match = re.search(r'score-([\-\d\.]+)', fname)
        if match:
            score_str = match.group(1)
            score_str = score_str.rstrip('.')
            try:
                return float(score_str)
            except ValueError:
                print(f"[WARNING] Could not parse score '{score_str}' from {fname}")
                return float('nan')
        return float('nan')
    
class GninaApproach(DockingApproach):
    def get_name(self) -> str:
        return "gnina"
    
    def list_top_n_files(self, protein_dir: str, top_n: int) -> List[str]:
        """
        GNINA outputs are named like 'rank1_score-7.97.sdf'
        We'll parse 'rank(\d+)' to get the rank, sort, and return top_n.
        """
        all_files = os.listdir(protein_dir)
        sdf_files = [f for f in all_files if f.startswith('rank') and f.endswith('.sdf')]
        
        def extract_rank_num(fname: str) -> int:
            match = re.search(r'rank(\d+)', fname)
            if match:
                return int(match.group(1))
            return 999999
        
        sdf_files.sort(key=extract_rank_num)
        return [os.path.join(protein_dir, f) for f in sdf_files[:top_n]]
    
    def parse_score(self, sdf_path: str) -> float:
        """
        Score is in filename like 'rank1_score-7.97.sdf'
        """
        fname = os.path.basename(sdf_path)
        match = re.search(r'score([\-\d\.]+)', fname)
        if match:
            score_str = match.group(1)
            score_str = score_str.rstrip('.')
            try:
                return float(score_str)
            except ValueError:
                print(f"[WARNING] Could not parse score '{score_str}' from {fname}")
                return float('nan')
        return float('nan')
    
class SurfDockApproach(DockingApproach):
    def get_name(self) -> str:
        return "surfdock"

    def list_top_n_files(self, protein_dir: str, top_n: int) -> List[str]:
        """
        Rename the directory from something like:
          5SAK_ZRY_protein_8A_5SAK_ZRY_ligand
        to:
          5SAK_ZRY
        Then locate any SDF files and return the top_n entries (sorted if needed).
        """
        # Example directory rename (adjust as necessary)
        new_dir = re.sub(r"_protein_.*_ligand$", "", protein_dir)
        if os.path.exists(protein_dir) and not os.path.exists(new_dir):
            os.rename(protein_dir, new_dir)
        
        all_files = os.listdir(new_dir)
        sdf_files = [f for f in all_files if f.endswith(".sdf")]

        # Sort by rank if needed, for example:
        # parse something like ..._rank_11_... 
        def extract_rank(fname: str) -> int:
            match = re.search(r"rank_(\d+)", fname)
            if match:
                return int(match.group(1))
            return 999999

        sdf_files.sort(key=extract_rank)
        return [os.path.join(new_dir, f) for f in sdf_files[:top_n]]

    def parse_score(self, sdf_path: str) -> float:
        """
        Extract RMSD and confidence from filenames like:
          5SAK_5SAK_ZRY_A_404_5SAK_ZRY_ligand.sdf_file_inner_idx_0_sample_idx_19_rank_11_rmsd_0.38968_confidence_280.8584.sdf
        Parse both RMSD and confidence, return confidence as the 'score'.
        """
        fname = os.path.basename(sdf_path)
        
        # Parse RMSD
        match_rmsd = re.search(r"rmsd_([\-\d\.]+)", fname)
        if match_rmsd:
            try:
                rmsd = float(match_rmsd.group(1))
            except ValueError:
                print(f"[WARNING] Could not parse RMSD from {fname}")
                rmsd = float('nan')
        else:
            rmsd = float('nan')
        
        # Parse confidence
        match_conf = re.search(r"confidence_([\-\d\.]+)", fname)
        if match_conf:
            try:
                confidence = float(match_conf.group(1).rstrip('.'))
            except ValueError:
                print(f"[WARNING] Could not parse confidence from {fname}")
                confidence = float('nan')
        else:
            confidence = float('nan')
        
        print(f"Parsed from {fname}: RMSD={rmsd}, confidence={confidence}")
        # Return confidence as the 'score' for consistency
        return confidence
 
class BoltzApproach(DockingApproach):
    """BoltzApproach for molecular structure prediction with robust extraction and alignment"""
    
    def get_name(self) -> str:
        return "boltz"
    
    def list_top_n_files(self, protein_dir: str, top_n: int) -> List[str]:
        """
        Boltz outputs are in predictions/<protein_name>/<protein_name>_model_X.pdb
        We'll extract ligand structures and apply robust alignment.
        """
        # Navigate to the predictions subdirectory
        protein_name = os.path.basename(protein_dir).replace("boltz_results_", "")
        pred_dir = os.path.join(protein_dir, "predictions", protein_name)
        
        if not os.path.exists(pred_dir):
            print(f"[WARNING] Predictions directory not found: {pred_dir}")
            return []
        
        # Find all PDB model files
        all_files = os.listdir(pred_dir)
        pdb_files = [f for f in all_files if f.endswith('.pdb') and '_model_' in f]
        
        def extract_model_num(fname: str) -> int:
            match = re.search(r'_model_(\d+)\.pdb', fname)
            if match:
                return int(match.group(1))
            return 999999
        
        # Sort by model number
        pdb_files.sort(key=extract_model_num)
        
        # Extract ligand structures and apply robust alignment
        aligned_paths = []
        for pdb_file in pdb_files[:top_n]:
            pdb_path = os.path.join(pred_dir, pdb_file)
            sdf_path = self._robust_ligand_extraction(pdb_path, protein_name)
            if sdf_path:
                aligned_paths.append(sdf_path)
        
        return aligned_paths
    
    def _robust_ligand_extraction(self, pdb_path: str, protein_name: str) -> str:
        """
        Robust ligand extraction and alignment mimicking complex_alignment.py approach.
        """
        try:
            # Import required modules
            from Bio.PDB.PDBParser import PDBParser
            from Bio.PDB.PDBExceptions import PDBConstructionWarning
            from scipy.spatial.transform import Rotation
            import warnings
            import tempfile
            
            # Parse the predicted complex structure
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=PDBConstructionWarning)
                parser = PDBParser()
                predicted_structure = parser.get_structure("predicted", pdb_path)
                predicted_model = predicted_structure[0]
            
            # Extract ligand using robust HETATM parsing
            predicted_ligand_sdf = self._extract_ligand_to_sdf(pdb_path)
            if not predicted_ligand_sdf:
                print(f"[WARNING] Failed to extract ligand from {pdb_path}")
                return predicted_ligand_sdf or ""
            
            # Load reference structures
            ref_protein_path = f"/Users/aoxu/projects/DrugDiscovery/PoseBench/data/plinder_set/{protein_name}/{protein_name}_protein.pdb"
            ref_ligand_path = f"/Users/aoxu/projects/DrugDiscovery/PoseBench/data/plinder_set/{protein_name}/{protein_name}_ligand.sdf"
            
            if not os.path.exists(ref_protein_path) or not os.path.exists(ref_ligand_path):
                print(f"[WARNING] Reference structures not found for {protein_name}")
                return predicted_ligand_sdf
            
            # Apply robust complex alignment
            aligned_sdf_path = self._apply_robust_complex_alignment(
                pdb_path, predicted_ligand_sdf, ref_protein_path, ref_ligand_path, protein_name
            )
            
            return aligned_sdf_path
            
        except Exception as e:
            print(f"[ERROR] Robust ligand extraction failed for {pdb_path}: {str(e)}")
            return ""
    
    def _extract_ligand_to_sdf(self, pdb_path: str) -> str:
        """Extract ligand from PDB file using robust HETATM parsing."""
        try:
            import tempfile
            
            # Extract HETATM records
            ligand_lines = []
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith('HETATM'):
                        ligand_lines.append(line)
            
            if not ligand_lines:
                print(f"[WARNING] No HETATM records found in {pdb_path}")
                return ""
            
            # Create temporary ligand PDB file
            temp_pdb = tempfile.mktemp(suffix='_ligand.pdb')
            with open(temp_pdb, 'w') as f:
                for line in ligand_lines:
                    f.write(line)
                f.write('END\n')
            
            # Use robust molecule reading (mimicking read_molecule from complex_alignment.py)
            mol = self._read_molecule_robust(temp_pdb)
            if mol is None:
                print(f"[WARNING] Could not parse ligand molecule from {pdb_path}")
                os.remove(temp_pdb)
                return ""
            
            # Save to SDF
            sdf_path = pdb_path.replace('.pdb', '_ligand.sdf')
            writer = Chem.SDWriter(sdf_path)
            writer.write(mol)
            writer.close()
            
            # Clean up
            os.remove(temp_pdb)
            return sdf_path
            
        except Exception as e:
            print(f"[ERROR] Failed to extract ligand from {pdb_path}: {str(e)}")
            return ""
    
    def _read_molecule_robust(self, molecule_file: str):
        """Robust molecule reading mimicking the complex_alignment.py approach."""
        try:
            from rdkit import Chem
            
            # Try different parsing approaches
            if molecule_file.endswith('.pdb'):
                # First try using direct RDKit approach
                try:
                    mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
                    if mol is not None:
                        return mol
                except:
                    pass
                
                # If that fails, try PDB block approach
                try:
                    with open(molecule_file) as f:
                        pdb_data = f.readlines()
                    pdb_block = ""
                    for line in pdb_data:
                        if line.startswith("HETATM"):
                            pdb_block += line
                    mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
                    return mol
                except:
                    pass
            
            elif molecule_file.endswith('.sdf'):
                supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
                for mol in supplier:
                    if mol is not None:
                        return mol
                        
        except Exception as e:
            print(f"[WARNING] RDKit was unable to read the molecule {molecule_file} due to: {e}")
            
        return None
    
    def _apply_robust_complex_alignment(self, predicted_pdb: str, predicted_ligand_sdf: str, 
                                      ref_protein_pdb: str, ref_ligand_sdf: str, protein_name: str) -> str:
        """
        Apply robust complex alignment using the approach from complex_alignment.py.
        """
        try:
            # Import necessary modules
            from Bio.PDB.PDBParser import PDBParser
            from Bio.PDB.PDBExceptions import PDBConstructionWarning
            from rdkit.Chem import rdGeometry
            from scipy.spatial.transform import Rotation
            from scipy.optimize import Bounds, minimize
            import warnings
            
            # Load structures with robust parsing
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=PDBConstructionWarning)
                parser = PDBParser()
                
                # Parse predicted and reference proteins
                predicted_structure = parser.get_structure("predicted", predicted_pdb)
                predicted_rec = predicted_structure[0]
                
                reference_structure = parser.get_structure("reference", ref_protein_pdb)
                reference_rec = reference_structure[0]
            
            # Load ligands
            predicted_ligand = self._read_molecule_robust(predicted_ligand_sdf)
            reference_ligand = self._read_molecule_robust(ref_ligand_sdf)
            
            if predicted_ligand is None or reference_ligand is None:
                print(f"[WARNING] Could not load ligands for {protein_name}")
                return predicted_ligand_sdf
            
            # Extract CA coordinates using robust structure extraction
            try:
                predicted_calpha_coords = self._extract_calpha_coordinates(predicted_rec, reference_ligand)
                reference_calpha_coords = self._extract_calpha_coordinates(reference_rec, reference_ligand)
            except Exception as e:
                print(f"[WARNING] Failed to extract CA coordinates for {protein_name}: {e}")
                return predicted_ligand_sdf
            
            if predicted_calpha_coords is None or reference_calpha_coords is None:
                print(f"[WARNING] Could not extract CA coordinates for {protein_name}")
                return predicted_ligand_sdf
            
            if reference_calpha_coords.shape != predicted_calpha_coords.shape:
                print(f"[WARNING] CA coordinate shape mismatch for {protein_name}: {reference_calpha_coords.shape} vs {predicted_calpha_coords.shape}")
                return predicted_ligand_sdf
            
            # Get reference ligand coordinates
            reference_ligand_coords = reference_ligand.GetConformer().GetPositions()
            predicted_ligand_conf = predicted_ligand.GetConformer()
            
            # Perform optimization-based alignment (mimicking complex_alignment.py)
            res = minimize(
                self._align_prediction_objective,
                [0.1],
                bounds=Bounds([0.0], [1.0]),
                args=(reference_calpha_coords, predicted_calpha_coords, reference_ligand_coords),
                tol=1e-8,
            )
            
            smoothing_factor = res.x[0]
            rotation, reference_centroid, predicted_centroid = self._compute_alignment_transformation(
                smoothing_factor, reference_calpha_coords, predicted_calpha_coords, reference_ligand_coords
            )
            
            if rotation is None:
                print(f"[WARNING] Could not compute alignment transformation for {protein_name}")
                return predicted_ligand_sdf
            
            # Apply transformation to predicted ligand
            predicted_ligand_coords = predicted_ligand_conf.GetPositions()
            predicted_ligand_aligned = (
                rotation.apply(predicted_ligand_coords - predicted_centroid) + reference_centroid
            )
            
            # Update ligand coordinates
            for i in range(predicted_ligand.GetNumAtoms()):
                x, y, z = predicted_ligand_aligned[i]
                predicted_ligand_conf.SetAtomPosition(i, rdGeometry.Point3D(x, y, z))
            
            # Save aligned ligand
            aligned_sdf_path = predicted_ligand_sdf.replace('.sdf', '_complex_aligned.sdf')
            with Chem.SDWriter(aligned_sdf_path) as writer:
                writer.write(predicted_ligand)
            
            print(f"[INFO] Applied robust complex alignment for {protein_name}")
            return aligned_sdf_path
            
        except Exception as e:
            print(f"[WARNING] Complex alignment failed for {protein_name}: {e}")
            return predicted_ligand_sdf
    
    def _extract_calpha_coordinates(self, structure, reference_ligand):
        """Extract CA coordinates from BioPython structure."""
        try:
            from scipy import spatial
            import numpy as np
            
            if reference_ligand is not None:
                ref_ligand_coords = reference_ligand.GetConformer().GetPositions()
            else:
                ref_ligand_coords = None
            
            calpha_coords = []
            for chain in structure:
                for residue in chain:
                    # Filter out water and hetero residues
                    if residue.get_resname() == "HOH" or len(residue.get_id()[0]) > 1:
                        continue
                    
                    # Look for CA atom
                    if 'CA' in residue:
                        ca_atom = residue['CA']
                        calpha_coords.append(list(ca_atom.get_vector()))
            
            if not calpha_coords:
                return None
                
            return np.array(calpha_coords)
            
        except Exception as e:
            print(f"[WARNING] Failed to extract CA coordinates: {e}")
            return None
    
    def _align_prediction_objective(self, smoothing_factor, ref_calpha_coords, pred_calpha_coords, ref_ligand_coords):
        """Objective function for alignment optimization."""
        try:
            from scipy import spatial
            import numpy as np
            from scipy.spatial.transform import Rotation
            
            # Compute weighted centroids
            ref_dists = None
            if ref_ligand_coords is not None:
                ref_dists = spatial.distance.cdist(ref_calpha_coords, ref_ligand_coords)
                weights = np.exp(-1 * smoothing_factor[0] * np.amin(ref_dists, axis=1))
                ref_centroid = np.sum(np.expand_dims(weights, axis=1) * ref_calpha_coords, axis=0) / np.sum(weights)
                pred_centroid = np.sum(np.expand_dims(weights, axis=1) * pred_calpha_coords, axis=0) / np.sum(weights)
            else:
                weights = None
                ref_centroid = np.mean(ref_calpha_coords, axis=0)
                pred_centroid = np.mean(pred_calpha_coords, axis=0)
            
            # Center coordinates
            centered_ref = ref_calpha_coords - ref_centroid
            centered_pred = pred_calpha_coords - pred_centroid
            
            # Compute rotation
            rotation, rmsd = Rotation.align_vectors(centered_ref, centered_pred, weights)
            
            # Compute objective
            if ref_ligand_coords is not None and ref_dists is not None:
                centered_ref_ligand = ref_ligand_coords - ref_centroid
                aligned_pred_calpha = rotation.apply(centered_pred)
                aligned_pred_ref_dists = spatial.distance.cdist(aligned_pred_calpha, centered_ref_ligand)
                inv_r_rmse = np.sqrt(np.mean(((1 / ref_dists) - (1 / aligned_pred_ref_dists)) ** 2))
                return inv_r_rmse
            else:
                return 0.0
                
        except Exception as e:
            return float('inf')
    
    def _compute_alignment_transformation(self, smoothing_factor, ref_calpha_coords, pred_calpha_coords, ref_ligand_coords):
        """Compute the alignment transformation matrices."""
        try:
            from scipy import spatial
            import numpy as np
            from scipy.spatial.transform import Rotation
            
            # Compute weighted centroids
            if ref_ligand_coords is not None:
                ref_dists = spatial.distance.cdist(ref_calpha_coords, ref_ligand_coords)
                weights = np.exp(-1 * smoothing_factor * np.amin(ref_dists, axis=1))
                ref_centroid = np.sum(np.expand_dims(weights, axis=1) * ref_calpha_coords, axis=0) / np.sum(weights)
                pred_centroid = np.sum(np.expand_dims(weights, axis=1) * pred_calpha_coords, axis=0) / np.sum(weights)
            else:
                weights = None
                ref_centroid = np.mean(ref_calpha_coords, axis=0)
                pred_centroid = np.mean(pred_calpha_coords, axis=0)
            
            # Center coordinates
            centered_ref = ref_calpha_coords - ref_centroid
            centered_pred = pred_calpha_coords - pred_centroid
            
            # Compute rotation
            rotation, rmsd = Rotation.align_vectors(centered_ref, centered_pred, weights)
            
            return rotation, ref_centroid, pred_centroid
            
        except Exception as e:
            print(f"[ERROR] Failed to compute alignment transformation: {e}")
            return None, None, None
    
    def parse_score(self, sdf_path: str) -> float:
        """Parse confidence score from the corresponding JSON file."""
        try:
            # Get the base path and model number from SDF path
            pred_dir = os.path.dirname(sdf_path)
            
            # Extract model number from the SDF path
            match = re.search(r'_model_(\d+)_ligand', sdf_path)
            if not match:
                print(f"[WARNING] Could not extract model number from {sdf_path}")
                return float('nan')
            
            model_num = match.group(1)
            protein_name = os.path.basename(pred_dir)
            
            # Look for confidence JSON file
            confidence_file = f"confidence_{protein_name}_model_{model_num}.json"
            confidence_path = os.path.join(pred_dir, confidence_file)
            
            if os.path.exists(confidence_path):
                import json
                with open(confidence_path, 'r') as f:
                    data = json.load(f)
                    confidence_score = data.get('confidence_score', float('nan'))
                    print(f"[INFO] Found confidence score {confidence_score} for model {model_num}")
                    return confidence_score
            
            print(f"[WARNING] No score files found for {sdf_path}")
            return float('nan')
            
        except Exception as e:
            print(f"[ERROR] Failed to parse score from {sdf_path}: {str(e)}")
            return float('nan')

class UniDock2Approach(DockingApproach):
    def get_name(self) -> str:
        return "unidock2"

    def list_top_n_files(self, protein_dir: str, top_n: int) -> List[str]:
        """
        Uni-Dock2 outputs are named like '5SAK_ZRY_pose3_score-7.97.sdf'
        We'll parse 'pose(\d+)' to get the rank, sort, and return top_n.
        """
        all_files = os.listdir(protein_dir)
        sdf_files = [f for f in all_files if '_pose' in f and f.endswith('.sdf')]

        def extract_rank_num(fname: str) -> int:
            match = re.search(r'rank(\d+)', fname)
            if match:
                return int(match.group(1))
            return 999999

        sdf_files.sort(key=extract_rank_num)
        return [os.path.join(protein_dir, f) for f in sdf_files[:top_n]]


    def parse_score(self, sdf_path: str, metric: Optional[str] = "vina_binding_free_energy") -> float:
        """
        get score from sdf file
        """
        try:
            with open(sdf_path, 'r') as f:
                content = f.read()
                match = re.search(r'<{}>\s*\(\d+\)\s*\n([-\d.]+)'.format(metric), content)
                if match:
                    return float(match.group(1))
        except Exception as e:
            print(f"[ERROR] Failed to parse score from {sdf_path}: {str(e)}")
        return float('nan')