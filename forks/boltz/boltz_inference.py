import os
import glob
import subprocess
import argparse
import time
import logging
from pathlib import Path

# ─── configure logging ────────────────────────────────────────────────────────
# this will append to runtime.log in the current working dir
logging.basicConfig(
    filename="./logs/plinder_set_0_runtime.log",
    filemode="a",
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)

def run_boltz_inference(input_dir="input", output_dir="inference/plinder_set_0", samples=5):
    os.makedirs(output_dir, exist_ok=True)
    yaml_files = glob.glob(os.path.join(input_dir, "*.yaml"))
    if not yaml_files:
        print(f"No yaml files found in {input_dir}")
        return
    print(f"Found {len(yaml_files)} yaml files to process")

    for i, yaml_file in enumerate(yaml_files, start=1):
        basename = os.path.basename(yaml_file)
        file_output_dir = output_dir
        os.makedirs(file_output_dir, exist_ok=True)
        print(f"[{i}/{len(yaml_files)}] Processing {basename}")

        cmd = [
            "boltz", "predict", yaml_file,
            "--out_dir", file_output_dir,
            "--use_msa_server",
            "--output_format", "pdb",
            "--diffusion_samples", str(samples)
        ]

        # ─── time the call ──────────────────────────────────────────────────────
        start = time.time()
        try:
            subprocess.run(cmd, check=True)
            elapsed = time.time() - start
            msg = f"SUCCESS {basename} in {elapsed:.2f}s"
            print(msg)
            logging.info(msg)
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start
            msg = f"FAIL    {basename} in {elapsed:.2f}s | {e}"
            print(msg)
            logging.error(msg)


def run_boltz_inference_fasta(input_dir="input", output_dir="inference/plinder_set_0", samples=5):
    os.makedirs(output_dir, exist_ok=True)
    fasta_files = glob.glob(os.path.join(input_dir, "*.fasta"))
    if not fasta_files:
        print(f"No fasta files found in {input_dir}")
        return
    print(f"Found {len(fasta_files)} fasta files to process")

    for i, fasta_file in enumerate(fasta_files, start=1):
        basename = os.path.basename(fasta_file)
        name_without_ext = os.path.splitext(basename)[0]
        file_output_dir = os.path.join(output_dir, name_without_ext)
        os.makedirs(file_output_dir, exist_ok=True)
        print(f"[{i}/{len(fasta_files)}] Processing {basename}")

        cmd = [
            "boltz", "predict", fasta_file,
            "--out_dir", file_output_dir,
            "--use_msa_server",
            "--output_format", "pdb",
            "--diffusion_samples", str(samples)
        ]

        start = time.time()
        try:
            subprocess.run(cmd, check=True)
            elapsed = time.time() - start
            msg = f"SUCCESS {basename} in {elapsed:.2f}s"
            print(msg)
            logging.info(msg)
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start
            msg = f"FAIL    {basename} in {elapsed:.2f}s | {e}"
            print(msg)
            logging.error(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Boltz predictions on fasta files")
    parser.add_argument("--input_dir", default="input", help="Directory containing input fasta files")
    parser.add_argument("--output_dir", default="inference/plinder_set_0", help="Base directory for outputs")
    parser.add_argument("--samples", type=int, default=5, help="Number of diffusion samples to generate")
    
    args = parser.parse_args()
    
    run_boltz_inference(args.input_dir, args.output_dir, args.samples)