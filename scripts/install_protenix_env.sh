#!/usr/bin/env bash
# Install Protenix for CogLigandBench.
#
# Creates a conda env at /mnt/katritch_lab2/aoxu/envs/protenix,
# symlinks it from CogLigandBench/envs/protenix, and pip-installs
# the protenix package. Models auto-download to ~/.protenix on first run.
#
# Requires: conda, CUDA GPU.
# Usage: bash scripts/install_protenix_env.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PARENT="/mnt/katritch_lab2/aoxu/envs"
ENV_PREFIX="${ENV_PARENT}/protenix"
ENV_LINK="${PROJECT_ROOT}/envs/protenix"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# [1/5] Create parent directory
log "[1/5] Ensuring env parent directory exists: ${ENV_PARENT}"
mkdir -p "${ENV_PARENT}"

# [2/5] Create conda env with Python 3.11
if [[ -d "${ENV_PREFIX}" ]]; then
    log "[2/5] Conda env already exists at ${ENV_PREFIX} — skipping creation."
else
    log "[2/5] Creating conda env at ${ENV_PREFIX} with Python 3.11 ..."
    conda create -y -p "${ENV_PREFIX}" python=3.11
fi

# [3/5] Symlink from project envs/ directory
log "[3/5] Symlinking ${ENV_LINK} -> ${ENV_PREFIX}"
mkdir -p "$(dirname "${ENV_LINK}")"
ln -sfn "${ENV_PREFIX}" "${ENV_LINK}"

# [4/5] Install protenix
log "[4/5] Installing protenix ..."
conda run -p "${ENV_PREFIX}" pip install protenix

# [5/5] Sanity check
log "[5/5] Verifying installation ..."
conda run -p "${ENV_PREFIX}" protenix --help
log ""
log "=== Protenix environment ready ==="
log "  Env:     ${ENV_PREFIX}"
log "  Symlink: ${ENV_LINK}"
log "  Binary:  ${ENV_PREFIX}/bin/protenix"
log ""
log "Models will auto-download to ~/.protenix on first run."
log "Test with: conda run -p ${ENV_PREFIX} protenix pred -i examples/input.json -o ./output -n protenix-v2"
