import argparse
from pathlib import Path
import random

import torch

from chai_lab.chai1 import run_inference

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on protein structures")
    parser.add_argument("--fasta_file", type=Path, help="Path to fasta_file", required=True)
    parser.add_argument("-o", "--output_dir", type=Path, help="Directory for output files", required=True)
    parser.add_argument("--num_trunk_recycles", type=int, default=10, help="Number of trunk recycles")
    parser.add_argument("--num_diffn_timesteps", type=int, default=200, help="Number of diffusion timesteps")
    parser.add_argument("--num_seeds", type=int, default=1, help="Number of random seeds to use")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--use_esm_embeddings", action="store_true", help="Use ESM embeddings")
    parser.add_argument("--msa_directory", type=Path, help="Directory to msa files")
    return parser.parse_args()

def main():
    args = parse_args()

    fasta_path = args.fasta_file

    output_dir = args.output_dir

    for i in range(args.num_seeds):
        seed = random.randint(0, 10000)

        seed_output_dir = output_dir / f"seed_{seed}"
        seed_output_dir.mkdir(parents=True, exist_ok=True)

        output_paths = run_inference(
            fasta_file=fasta_path,
            output_dir=seed_output_dir,
            num_trunk_recycles=args.num_trunk_recycles,
            num_diffn_timesteps=args.num_diffn_timesteps,
            seed=seed,
            device=torch.device(args.device),
            use_esm_embeddings=True,
            use_msa_server=False,
            msa_directory=args.msa_directory,
        )

        print(f"Run {i+1}/{args.num_seeds} - Seed: {seed}")
        print(f"Output files saved to: {output_paths}")

if __name__ == "__main__":
    main()
