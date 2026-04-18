# FILE: run_automation.py
# VERSION 4: Final version for processing 5 predictions for each unique reference system.
LENGTH_CUTOFF = 21  # ← change here if you later confirm 23

import os
import glob
import subprocess
import tempfile
import re
from typing import Tuple, Dict, Optional

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Split "foo_model_42"  →  {"complex": "foo", "model": 42}
# ──────────────────────────────────────────────────────────────────────────────
def parse_complex_id(cid: str) -> Dict[str, str | int]:
    """
    Break a complex identifier of the form '<complex>_model_<int>' into pieces.

    Parameters
    ----------
    cid : str
        Example: '6au0__1__1.A__1.B_model_0'

    Returns
    -------
    dict
        {'complex': '6au0__1__1.A__1.B', 'model': 0}
    """
    try:
        complex_part, model_part = cid.rsplit('_model_', 1)
        return {"complex": complex_part, "model": int(model_part)}
    except ValueError as exc:
        raise ValueError(f"Unexpected complex-ID format: {cid!r}") from exc


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Scrape the LAST   "<cid>,r_prot_rmsd,<float>"  line from stdout
# ──────────────────────────────────────────────────────────────────────────────
def extract_icm_rmsd(stdout: str) -> Tuple[str, int, float]:
    """
    Return **complex name**, **model id**, **protein RMSD** from an ICM log.

    Parameters
    ----------
    stdout : str
        Raw text captured from the ICM run.

    Returns
    -------
    Tuple[str, int, float]
        complex_name, model_id, rmsd

    Raises
    ------
    ValueError
        If no rmsd line is found or ID cannot be parsed.
    """
    # regex:  start-of-line, CID, ',r_prot_rmsd,', number
    pat = re.compile(r'^([^,\s]+),\s*r_prot_rmsd,\s*([0-9]*\.?[0-9]+)', re.M)

    *_, last = pat.findall(stdout) or [()]
    if not last:
        raise ValueError("No r_prot_rmsd entry found.")

    cid, rmsd_str = last
    parts = parse_complex_id(cid)              # {'complex': ..., 'model': ...}
    return parts["complex"], parts["model"], float(rmsd_str)

def get_loading_tag(stdout: str, which: str = "last") -> str:
    """
    Extract the '<index>_<chain>' tag that appears right after
        'parrayToMol lig_table.mol [ I_out ] ""'
        'Loading molecules  .icm/<index>_<chain>...'

    Examples of the target substring:
        .icm/1_B>
        .icm/3_F>

    Parameters
    ----------
    stdout : str
        Full text captured from the ICM subprocess.
    which  : {'first', 'last'}, default 'last'
        Some runs emit several Loading-molecules lines.  Choose which one
        to return.

    Returns
    -------
    str
        The tag, e.g. '3_F'.

    Raises
    ------
    ValueError
        If the pattern cannot be found.
    """
    # ▸  Look for “Loading molecules  .icm/<id>” where <id>=digits + '_' + letter
    pattern = re.compile(r'Loading molecules\s+\.icm/([0-9]+_[A-Za-z])>', re.M)

    hits = pattern.findall(stdout)
    if not hits:
        raise ValueError("No 'Loading molecules  .icm/<tag>' line found.")

    if which not in {"first", "last"}:
        raise ValueError("Argument 'which' must be 'first' or 'last'.")

    return hits[0] if which == "first" else hits[-1]

def icm_loading_tag(complex_id: str, cutoff: int = 21) -> str:
    """
    Predict the tag ICM prints after 'Loading molecules  .icm/<tag>'.

    Empirical rule
    --------------
    1.  Strip a trailing  _model_<n>  (if present).
    2.  Grab the final double-underscore segment,  e.g. '1.F_1.G'.
    3.  If that segment contains multiple ligand tokens (underscores):
          • If the *whole* complex_id (after step-1) is **longer than
            `cutoff` characters**, choose the **first** ligand token.
          • Otherwise choose the **last** ligand token.
        (If there is only one token, use it regardless of length.)
    4.  Dot 'idx.lig'  →  'idx_lig'.

    Parameters
    ----------
    complex_id : str
    cutoff     : int, default 21
        Length threshold that flips 'first-vs-last' choice.

    Returns
    -------
    str
        Tag such as '2_F', '1_FA', '1_G', ...
    """
    # 1️⃣  remove optional '_model_<digits>'
    bare = re.sub(r'_model_\d+$', '', complex_id)

    # 2️⃣  final '__' segment
    tail = bare.rsplit('__', 1)[-1]          # e.g. '1.F_1.G'

    # 3️⃣  split into ligand tokens like ['1.F', '1.G']
    tokens = tail.split('_')

    # choose first/last depending on length (if >1 token)
    if len(tokens) > 1:
        chosen = tokens[0] if len(bare) > cutoff else tokens[-1]
    else:
        chosen = tokens[0]

    # 4️⃣  turn '1.F' → '1_F'
    return chosen.replace('.', '_')

def icm_loading_tag_lenaware(complex_id: str,
                             cutoff: int = LENGTH_CUTOFF) -> str:
    """
    Guess the ICM ligand tag using the empirical length rule.

    Parameters
    ----------
    complex_id : str
        e.g. '1g4s__2__1.B__1.F_1.G'  or  '2oz2__1__2.A_2.B__2.F_2.L'.
    cutoff : int, default 21
        Length threshold at which ICM *sometimes* chooses the 2nd ligand.

    Returns
    -------
    str
        Ligand tag such as '1_F', '1_G', '2_F', '1_FA'.

    Notes
    -----
    •  Still ignores an optional trailing `_model_<n>`.
    •  If the length rule guesses the *wrong* token (there are corner cases!)
       parse the real tag from stdout instead; see earlier reply.
    """
    # 0) drop trailing _model_<n>
    cid_no_model = re.sub(r'_model_\d+$', '', complex_id)

    # 1) take last `__` segment
    last_chunk = cid_no_model.rsplit('__', 1)[-1]      # '1.F_1.G', '2.F'

    tokens: List[str] = last_chunk.split('_')           # ['1.F', '1.G']
    first_tok = tokens[0]                               # '1.F'

    # choose second token only if length ≤ cutoff AND we truly have >1 token
    if len(cid_no_model) <= cutoff and len(tokens) > 1:
        chosen = tokens[1]                              # '1.G'
    else:
        chosen = first_tok                              # default path

    return chosen.replace('.', '_')                     # '1_G' or '1_F'

def create_temporary_icm_script(template_path, system_id, icm_lig_name):
    """
    Create a temporary ICM script for a specific system by replacing ICMLIGNAME
    with the actual ligand name.
    
    Returns a NamedTemporaryFile object that can be used as a context manager.
    """
    # Read the template file
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    # Replace ICMLIGNAME with the actual ligand name
    modified_content = template_content.replace('ICMLIGNAME', icm_lig_name)
    
    # Create a temporary file with .icm extension
    temp_file = tempfile.NamedTemporaryFile(
        mode='w+', 
        suffix='.icm', 
        prefix=f'rmsd_{system_id}_',
        delete=False  # We'll manage deletion ourselves for better control
    )
    
    # Write the modified script
    temp_file.write(modified_content)
    temp_file.flush()  # Ensure content is written to disk
    
    return temp_file

def icm_loading_tag_from_stdout(stdout: str, which: str = "last") -> str:
    """
    Return the exact tag that appears in

        Loading molecules  .icm/<tag>

    Parameters
    ----------
    stdout : str
        Full text captured from the ICM subprocess.
    which  : {'first', 'last'}, default 'last'
        Choose the first or last occurrence.

    Returns
    -------
    str
        Tag such as '3_F', '1_FA', '1_G', …

    Raises
    ------
    ValueError
        If no tag is found.
    """
    pat = re.compile(r'Loading molecules\s+\.icm/([0-9]+_[A-Za-z0-9]+)>')
    hits = pat.findall(stdout)
    if not hits:
        raise ValueError("No 'Loading molecules  .icm/<tag>' line found.")
    return hits[0] if which == "first" else hits[-1]

def main():
    # --- Configuration ---
    
    # 1. Path to the reference data (where each system has its own folder)
    # reference_data_root = "/home/aoxu/projects/PoseBench/data/plinder_set/"
    reference_data_root = "/home/aoxu/projects/PoseBench/data/runsNposes"
     
    # 2. ROOT path for all your prediction folders
    predictions_root = "/home/aoxu/projects/PoseBench/forks/boltz/inference/runsNposes_0/"
    
    # 3. Path to your ICM script template
   # Get the directory where this Python script is located.
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Create a full, absolute path to the ICM script.
    # This assumes 'calculate_rmsd.icm' is in the SAME directory as this Python script.
    icm_script_path = os.path.join(script_dir, "calculate_rmsd.icm")

    # 4. Where to save the final results
    output_csv_path = "icm_runsNpose_rmsd_results.csv"
    ligand_names = {}  # Dictionary to collect ligand names by system
    results = []  # List to collect results for later writing

    # --- Find all unique systems to process using the reference ligands as anchors ---
    ligand_files = glob.glob(os.path.join(reference_data_root, "*/*_ligand.sdf"), recursive=True)
    
    print(f"Found {len(ligand_files)} unique systems. Will process 5 models for each.")

    # --- Prepare the output file ---
    with open(output_csv_path, "w") as f:
        f.write("protein,idx,rmsd\n") # Write the header

    # --- Outer Loop: Iterate through each unique system ---
    for i, ref_lig_path in enumerate(sorted(ligand_files)):
        
        # Construct paths for the REFERENCE files for this system
        base_path_ref = ref_lig_path.replace("_ligand.sdf", "")
        system_name = os.path.basename(base_path_ref)
        ref_prot_path = base_path_ref + "_protein.pdb"
        icm_lig_name = icm_loading_tag_lenaware(system_name)

        # if system_name < "1g4s__2__1.B__1.F_1.G_1.H":
        #     continue

        print(f"\nProcessing System {i+1}/{len(ligand_files)}: {system_name}")

        # create temporary ICM script for this system
        temp_script = create_temporary_icm_script(
            icm_script_path, 
            system_name, 
            icm_lig_name
        )
        temp_script_path = temp_script.name
        temp_script.close()  # Close the file so it can be used by ICM

        # --- Inner Loop: Iterate through each predicted model (0 to 4) for this system ---
        for model_idx in range(5):
            
            # Create a unique name for this specific prediction
            prediction_name = f"{system_name}_model_{model_idx}"
            
            # Construct the full path to the PREDICTED pdb file
            pred_path = os.path.join(
                predictions_root,
                f"boltz_results_{system_name}",
                "predictions",
                system_name,
                f"{prediction_name}.pdb"
            )

            print(f"  -> Model {model_idx}...")

            # Check that all files for this specific run exist
            if not all(os.path.exists(p) for p in [ref_prot_path, pred_path, ref_lig_path]):
                print(f"    -> SKIPPING: Missing one or more files.")
                with open(output_csv_path, "a") as f:
                    f.write(f"{prediction_name},MISSING_FILE,MISSING_FILE\n")
                continue

            # Build and run the ICM command for this specific prediction
            command = [
                "/home/aoxu/icm-3.9-4/icm64",
                "-s",
                temp_script_path,  # Path to the temporary ICM script
                "--",
                ref_prot_path,      # S_ARGV[1]
                pred_path,          # S_ARGV[2]
                ref_lig_path,       # S_ARGV[3]
                prediction_name     # S_ARGV[4]
            ]
            
            try:
                # Run the command and capture the output
                result = subprocess.run(command, capture_output=True, text=True, timeout=300)
                complex_name, model_id, prot_rmsd = extract_icm_rmsd(result.stdout)
                # instead of writing directly to CSV, collect results in a list
                results.append({
                    "protein": complex_name,
                    "idx":    model_id,
                    "rmsd": prot_rmsd
                })
                if prot_rmsd == 0:
                    # lig_name = get_loading_tag(result.stdout, which="last")
                    lig_name = icm_loading_tag_from_stdout(result.stdout, )
                    ligand_names[system_name] = lig_name
                    print("FAILED: ligand RMSD is 0, using the last loading tag:", lig_name, icm_lig_name)
                    break

            except subprocess.CalledProcessError as e:
                print(f"    -> ERROR processing {prediction_name}: {e.stderr.strip()}")
                with open(output_csv_path, "a") as f:
                    f.write(f"{prediction_name},ICM_ERROR,ICM_ERROR\n")
            except subprocess.TimeoutExpired:
                print(f"    -> TIMEOUT processing {prediction_name}.")
                with open(output_csv_path, "a") as f:
                    f.write(f"{prediction_name},TIMEOUT,TIMEOUT\n")
        # Clean up the temporary file
        if temp_script is not None:
            try:
                temp_script_path = temp_script.name
                if os.path.exists(temp_script_path):
                    os.unlink(temp_script_path)
                    print(f"  -> Deleted temporary script: {temp_script_path}")
            except Exception as e:
                print(f"  -> WARNING: Could not delete temporary script: {e}")
                temp_script.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, mode='a', header=False, index=False)
    print(f"\nAutomation complete. Results saved to {output_csv_path}")
    
    # Ligand names are collected but not used in this script.
    with open("ligand_names.txt", "w") as f:
        for system_name, lig_name in ligand_names.items():
            f.write(f"{system_name}, {lig_name}\n")


def main_supplement():
    """
    Supplementary function to run the main automation script.
    This is useful for testing or running in different contexts.
    """
    root_dir = os.path.dirname(os.path.realpath(__file__))
    ligand_names = {}  # Dictionary to collect ligand names by system
    with open(f"{root_dir}/ligand_names.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(", ")
            if len(parts) == 2:
                system_name, lig_name = parts
                ligand_names[system_name] = lig_name

    # --- Configuration ---
    
    # 1. Path to the reference data (where each system has its own folder)
    # reference_data_root = "/home/aoxu/projects/PoseBench/data/plinder_set/"
    reference_data_root = "/home/aoxu/projects/PoseBench/data/runsNposes"
    
    # 2. ROOT path for all your prediction folders
    # predictions_root = "/home/aoxu/projects/PoseBench/forks/boltz/inference/plinder_set_0/"
    predictions_root = "/home/aoxu/projects/PoseBench/forks/boltz/inference/runsNposes_0/"
    
    # 3. Path to your ICM script template
   # Get the directory where this Python script is located.
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Create a full, absolute path to the ICM script.
    # This assumes 'calculate_rmsd.icm' is in the SAME directory as this Python script.
    icm_script_path = os.path.join(script_dir, "calculate_rmsd.icm")

    # 4. Where to save the final results
    output_csv_path = os.path.join(root_dir, "icm_runsNposes_rmsd_results.csv")
    results = []  # List to collect results for later writing
    # rmsd_df = pd.read_csv(f"{root_dir}/icm_runsNposes_rmsd_results_supplement.csv", index_col=0)

    # --- Find all unique systems to process using the reference ligands as anchors ---
    ligand_files = glob.glob(os.path.join(reference_data_root, "*/*_ligand.sdf"), recursive=True)
    
    print(f"Found {len(ligand_files)} unique systems. Will process 5 models for each.")

    # --- Prepare the output file ---
    with open(output_csv_path, "w") as f:
        f.write("protein,idx,rmsd\n") # Write the header

    # --- Outer Loop: Iterate through each unique system ---
    for i, ref_lig_path in enumerate(sorted(ligand_files)):
        
        # Construct paths for the REFERENCE files for this system
        base_path_ref = ref_lig_path.replace("_ligand.sdf", "")
        system_name = os.path.basename(base_path_ref)
        ref_prot_path = base_path_ref + "_protein.pdb"
        
        if system_name not in ligand_names:
            continue

        icm_lig_name = ligand_names[system_name]
        print(f"\nProcessing System {i+1}/{len(ligand_files)}: {system_name}")

        # create temporary ICM script for this system
        temp_script = create_temporary_icm_script(
            icm_script_path, 
            system_name, 
            icm_lig_name
        )
        temp_script_path = temp_script.name
        temp_script.close()  # Close the file so it can be used by ICM

        # --- Inner Loop: Iterate through each predicted model (0 to 4) for this system ---
        for model_idx in range(5):
            
            # Create a unique name for this specific prediction
            prediction_name = f"{system_name}_model_{model_idx}"
            
            # Construct the full path to the PREDICTED pdb file
            pred_path = os.path.join(
                predictions_root,
                f"boltz_results_{system_name}",
                "predictions",
                system_name,
                f"{prediction_name}.pdb"
            )

            print(f"  -> Model {model_idx}...")

            # Check that all files for this specific run exist
            if not all(os.path.exists(p) for p in [ref_prot_path, pred_path, ref_lig_path]):
                print(f"    -> SKIPPING: Missing one or more files.")
                with open(output_csv_path, "a") as f:
                    f.write(f"{prediction_name},MISSING_FILE,MISSING_FILE\n")
                continue

            # Build and run the ICM command for this specific prediction
            command = [
                "/home/aoxu/icm-3.9-4/icm64",
                "-s",
                temp_script_path,  # Path to the temporary ICM script
                "--",
                ref_prot_path,      # S_ARGV[1]
                pred_path,          # S_ARGV[2]
                ref_lig_path,       # S_ARGV[3]
                prediction_name     # S_ARGV[4]
            ]
            
            try:
                # Run the command and capture the output
                result = subprocess.run(command, capture_output=True, text=True, timeout=300)
                complex_name, model_id, prot_rmsd = extract_icm_rmsd(result.stdout)
                # instead of writing directly to CSV, collect results in a list
                results.append({
                    "protein": complex_name,
                    "idx":    model_id,
                    "rmsd": prot_rmsd
                })
                if prot_rmsd == 0:
                    # lig_name = get_loading_tag(result.stdout, which="last")
                    lig_name = icm_loading_tag_from_stdout(result.stdout, )
                    ligand_names[system_name] = lig_name
                    print("FAILED: ligand RMSD is 0, using the last loading tag:", lig_name, icm_lig_name)
                    break

            except subprocess.CalledProcessError as e:
                print(f"    -> ERROR processing {prediction_name}: {e.stderr.strip()}")
                with open(output_csv_path, "a") as f:
                    f.write(f"{prediction_name},ICM_ERROR,ICM_ERROR\n")
            except subprocess.TimeoutExpired:
                print(f"    -> TIMEOUT processing {prediction_name}.")
                with open(output_csv_path, "a") as f:
                    f.write(f"{prediction_name},TIMEOUT,TIMEOUT\n")
        # Clean up the temporary file
        if temp_script is not None:
            try:
                temp_script_path = temp_script.name
                if os.path.exists(temp_script_path):
                    os.unlink(temp_script_path)
                    print(f"  -> Deleted temporary script: {temp_script_path}")
            except Exception as e:
                print(f"  -> WARNING: Could not delete temporary script: {e}")
                temp_script.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, mode='a', header=False, index=False)
    print(f"\nAutomation complete. Results saved to {output_csv_path}")
    
    # Ligand names are collected but not used in this script.
    with open("ligand_names.txt", "w") as f:
        for system_name, lig_name in ligand_names.items():
            f.write(f"{system_name}, {lig_name}\n")

def merge(): 
    """
    Merge the results from the main and supplementary runs into a single CSV file.
    """
    root_dir = os.path.dirname(os.path.realpath(__file__))
    main_results = pd.read_csv(f"{root_dir}/icm_rmsd_results.csv")
    supplement_results = pd.read_csv(f"{root_dir}/icm_rmsd_results_supplement.csv")

    # Combine the two DataFrames
    combined_results = pd.concat([main_results, supplement_results], ignore_index=True)
    # remove duplicates where rmsd is 0
    combined_results['rmsd'] = pd.to_numeric(combined_results['rmsd'], errors='coerce')
    combined_results = combined_results[combined_results['rmsd'] != 0.0]
    combined_results = combined_results[combined_results['idx'] != 'MISSING_FILE']
    # Save the combined results to a new CSV file
    combined_results.to_csv(f"{root_dir}/icm_rmsd_combined_results.csv", index=False)
    # compute the proportions of systems with rmsd < 2 for each unique protein with minimum rmsd
    proportion = (combined_results.groupby('protein')['rmsd'].min() < 2).sum() / combined_results['protein'].nunique()
    print(f"Proportion of systems with minimum RMSD < 2: {proportion:.2%}")

    print("Combined results saved to icm_rmsd_combined_results.csv")

if __name__ == "__main__":
    main()
    # main_supplement()
    # merge()