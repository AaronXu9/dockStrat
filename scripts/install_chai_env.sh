#!/usr/bin/env bash
# Install Chai-1 (chai-lab) for CogLigandBench.
#
# Creates a conda env at /mnt/katritch_lab2/aoxu/envs/chai, symlinks it from
# CogLigandBench/envs/chai, and pip-installs the `chai_lab` package.
#
# Idempotent: safe to re-run.
# Requires: conda, CUDA GPU (>=24 GB VRAM recommended).
# Usage: bash scripts/install_chai_env.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PARENT="/mnt/katritch_lab2/aoxu/envs"
ENV_PREFIX="${ENV_PARENT}/chai"
ENV_LINK="${PROJECT_ROOT}/envs/chai"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "[1/5] Ensuring env parent directory exists: ${ENV_PARENT}"
mkdir -p "${ENV_PARENT}"

log "[2/5] Creating conda env at ${ENV_PREFIX}"
if [ -x "${ENV_PREFIX}/bin/python" ]; then
    log "      env already exists; skipping create"
else
    conda create -y -p "${ENV_PREFIX}" python=3.11
fi

log "[3/5] Symlinking ${ENV_LINK} -> ${ENV_PREFIX}"
mkdir -p "${PROJECT_ROOT}/envs"
ln -sfn "${ENV_PREFIX}" "${ENV_LINK}"

log "[4/5] pip-installing chai_lab into ${ENV_PREFIX}"
if "${ENV_PREFIX}/bin/python" -c "from chai_lab.chai1 import run_inference" >/dev/null 2>&1; then
    log "      chai_lab already importable; skipping"
else
    "${ENV_PREFIX}/bin/python" -m pip install --upgrade pip
    "${ENV_PREFIX}/bin/python" -m pip install chai_lab
fi

log "[5/5] Sanity check"
"${ENV_PREFIX}/bin/python" -c "from chai_lab.chai1 import run_inference; print('chai_lab import OK')"

log ""
log "=== Chai-1 environment ready ==="
log "  Env:     ${ENV_PREFIX}"
log "  Symlink: ${ENV_LINK}"
log "  Python:  ${ENV_PREFIX}/bin/python"
log ""
log "Used by: cogligand_config/model/chai_inference.yaml (python_exec_path)"
log "Weights auto-download on first run."
