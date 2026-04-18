import os
import yaml
import argparse
import glob


"""# Process all runsNposes data (recommended)
python prepare_input.py --batch-process

# Process specific directory
python prepare_input.py --input /path/to/fasta/dir --output /path/to/yaml/dir --runsNposes-dir /path/to/runsNposes/data

# Process single file
python prepare_input.py --input file.fasta --output file.yaml --runsNposes-dir /path/to/runsNposes/data

"""
def extract_smiles_from_sdf(sdf_file):
    """Extract SMILES string from SDF file using RDKit"""
    try:
        from rdkit import Chem
        
        # Use getattr to avoid linting issues with RDKit
        supplier_func = getattr(Chem, 'SDMolSupplier', None)
        mol_to_smiles_func = getattr(Chem, 'MolToSmiles', None)
        
        if supplier_func is None or mol_to_smiles_func is None:
            print(f"RDKit functions not available")
            return None
        
        supplier = supplier_func(sdf_file)
        mol = next(iter(supplier), None)  # Get first molecule
        
        if mol is not None:
            smiles = mol_to_smiles_func(mol)
            return smiles
        else:
            print(f"Could not read molecule from {sdf_file}")
            return None
            
    except ImportError:
        print("RDKit not available. Please install: conda install -c conda-forge rdkit")
        return None
    except Exception as e:
        print(f"Error extracting SMILES from {sdf_file}: {e}")
        return None

def find_ligand_sdf(system_name, runsNposes_dir):
    """Find the ligand SDF file for a given system"""
    # For runsNposes format, SDF files are in data/runsNposes/ground_truth/[system_name]/ligand_files/
    # The runsNposes_dir could be pointing to different locations, so we need to find the right path
    
    # First try the direct path if runsNposes_dir points to the ground_truth directory
    ground_truth_path = os.path.join(runsNposes_dir, "ground_truth", system_name, "ligand_files")
    if not os.path.exists(ground_truth_path):
        # Try if runsNposes_dir already includes ground_truth
        ground_truth_path = os.path.join(runsNposes_dir, system_name, "ligand_files")
    
    if not os.path.exists(ground_truth_path):
        # Try going up from runsNposes_dir to find data/runsNposes/ground_truth
        base_path = runsNposes_dir
        while base_path and base_path != '/':
            test_path = os.path.join(base_path, "data", "runsNposes", "ground_truth", system_name, "ligand_files")
            if os.path.exists(test_path):
                ground_truth_path = test_path
                break
            base_path = os.path.dirname(base_path)
    
    # Look for SDF files in the ligand_files directory
    if os.path.exists(ground_truth_path):
        sdf_files = glob.glob(os.path.join(ground_truth_path, "*.sdf"))
        if sdf_files:
            # Return the first SDF file found
            return sdf_files[0]
    
    # Fallback: try other possible locations for compatibility
    possible_paths = [
        os.path.join(runsNposes_dir, system_name, f"{system_name}_ligand.sdf"),
        os.path.join(runsNposes_dir, system_name, "ligand_files", f"{system_name}_ligand.sdf"),
        os.path.join(runsNposes_dir, system_name, "ligand_files", "ligand.sdf"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def convert_fasta_format(input_file, output_file=None):
    """
    Convert FASTA files by:
    1. Adding chain letters (A for protein, B for ligand)
    2. Changing "ligand" to "smiles" in the header
    3. Removing strings after the "|" symbol
    """
    if output_file is None:
        # Use the same filename if no output file is specified
        output_file = input_file
    with open(input_file, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    
    for line in lines:
        line = line.strip()
        if line.startswith(">protein|"):
            # Replace with >A|protein|
            new_lines.append(">A|protein|")
        elif line.startswith(">ligand|"):
            # Replace with >B|smiles|
            new_lines.append(">B|smiles|")
        else:
            new_lines.append(line)
    
    with open(output_file, "w") as f:
        for i, line in enumerate(new_lines):
            f.write(line + "\n")
    
    print(f"Converted: {input_file} -> {output_file}")


def convert_fasta_to_yaml(fasta_file, output_file=None, runsNposes_dir=None):
    """
    Convert FASTA file to YAML format for Boltz.
    
    Args:
        fasta_file: Path to the FASTA file
        output_file: Output YAML file path (optional)
        runsNposes_dir: Directory to search for SDF files if SMILES not found in FASTA
    """
    if output_file is None:
        output_file = os.path.splitext(fasta_file)[0] + ".yaml"
    
    # Extract system name from filename for SDF searching
    system_name = os.path.basename(fasta_file).replace('.fasta', '')
    
    protein_sequence = ""
    smiles_string = ""
    current_section = None
    protein_lines = []
    smiles_lines = []
    
    # Parse FASTA file
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Check what type of sequence this is
                if '|protein|' in line:
                    # Plinder format: >A|protein|
                    current_section = 'protein'
                    protein_lines = []  # Reset protein
                elif '|smiles|' in line:
                    # Plinder format: >B|smiles|
                    current_section = 'smiles'
                    smiles_lines = []  # Reset smiles
                else:
                    # runsNposes format: >1.A, >1.B, etc.
                    # Treat all sequences as protein since runsNposes FASTA doesn't contain SMILES
                    current_section = 'protein'
                    protein_lines = []  # Reset protein
            elif current_section == 'protein':
                protein_lines.append(line)
            elif current_section == 'smiles':
                smiles_lines.append(line)
    
    # Combine sequences
    protein_sequence = ''.join(protein_lines)
    smiles_string = ''.join(smiles_lines)
    
    # If no SMILES found in FASTA (runsNposes case), try to find SDF file
    if not smiles_string and runsNposes_dir:
        sdf_file = find_ligand_sdf(system_name, runsNposes_dir)
        if sdf_file:
            smiles_string = extract_smiles_from_sdf(sdf_file)
            if smiles_string:
                print(f"Extracted SMILES from {sdf_file}: {smiles_string}")
            else:
                print(f"Failed to extract SMILES from {sdf_file}")
        else:
            print(f"No SDF file found for system {system_name}")
    
    # Create YAML structure following plinder format
    yaml_data = {
        'properties': [
            {
                'affinity': {
                    'binder': 'B'
                }
            }
        ],
        'sequences': [
            {
                'protein': {
                    'id': 'A',
                    'sequence': protein_sequence
                }
            }
        ],
        'version': 1
    }
    
    # Add ligand section if SMILES is available
    if smiles_string:
        yaml_data['sequences'].append({
            'ligand': {
                'id': 'B',
                'smiles': smiles_string
            }
        })
    
    # Write YAML file
    with open(output_file, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"Converted {fasta_file} -> {output_file}")
    if smiles_string:
        print(f"  SMILES: {smiles_string}")
    else:
        print(f"  Warning: No SMILES found for {system_name}")


def batch_convert(input_dir, output_dir=None, runsNposes_dir=None):
    """Convert all FASTA files in a directory"""
    if output_dir is None:
        output_dir = input_dir
        
    os.makedirs(output_dir, exist_ok=True)
    
    fasta_files = glob.glob(os.path.join(input_dir, '*.fasta'))
    
    for fasta_file in fasta_files:
        base_name = os.path.basename(fasta_file)
        name_without_ext = os.path.splitext(base_name)[0]
        yaml_file = os.path.join(output_dir, f"{name_without_ext}.yaml")
        convert_fasta_to_yaml(fasta_file, yaml_file, runsNposes_dir)


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description="Convert FASTA files to YAML format for Boltz")
    parser.add_argument("--input", "-i", type=str, help="Input FASTA file or directory")
    parser.add_argument("--output", "-o", type=str, help="Output YAML file or directory")
    parser.add_argument("--runsNposes-dir", type=str, 
                       help="Directory containing runsNposes data with SDF files")
    parser.add_argument("--batch-process", action="store_true", 
                       help="Process runsNposes data from archive to YAML")
    
    args = parser.parse_args()
    
    if args.batch_process:
        # Default batch processing for runsNposes
        archive_dir = "/home/aoxu/projects/PoseBench/data/runsNposes_archive/zenodo_downloads/ground_truth/"
        fasta_output_dir = "/home/aoxu/projects/PoseBench/forks/boltz/input/runsNposes/fasta"
        yaml_output_dir = "/home/aoxu/projects/PoseBench/forks/boltz/input/runsNposes/yaml"
        runsNposes_data_dir = "/home/aoxu/projects/PoseBench/data/runsNposes"
        
        # Create output directories
        os.makedirs(fasta_output_dir, exist_ok=True)
        os.makedirs(yaml_output_dir, exist_ok=True)
        
        print("Step 1: Converting FASTA format...")
        # Convert FASTA format first
        for root, dirs, files in os.walk(archive_dir):
            for dir in dirs:
                file_path = os.path.join(root, dir, "sequences.fasta")
                if os.path.exists(file_path):
                    output_file = os.path.join(fasta_output_dir, f"{dir}.fasta")
                    convert_fasta_format(file_path, output_file)
        
        print("\nStep 2: Converting FASTA to YAML with SMILES extraction...")
        # Convert FASTA to YAML with SMILES extraction
        batch_convert(fasta_output_dir, yaml_output_dir, runsNposes_data_dir)
        
    elif args.input:
        if os.path.isdir(args.input):
            batch_convert(args.input, args.output, args.runsNposes_dir)
        else:
            convert_fasta_to_yaml(args.input, args.output, args.runsNposes_dir)
    else:
        print("Please specify --input or use --batch-process")

if __name__ == "__main__":
    main()