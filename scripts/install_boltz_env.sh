#!/usr/bin/env bash
# Install Boltz (boltz-1 / boltz-2) for CogLigandBench.
#
# Creates a conda env at /mnt/katritch_lab2/aoxu/envs/boltz, symlinks it from
# CogLigandBench/envs/boltz, and pip-installs the `boltz` package. The same
# env serves both boltz1 and boltz2 (the model variant is selected at
# inference time via the YAML config).
#
# Idempotent: safe to re-run.
# Requires: conda, CUDA GPU.
# Usage: bash scripts/install_boltz_env.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PARENT="/mnt/katritch_lab2/aoxu/envs"
ENV_PREFIX="${ENV_PARENT}/boltz"
ENV_LINK="${PROJECT_ROOT}/envs/boltz"

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

log "[4/5] pip-installing boltz into ${ENV_PREFIX}"
if "${ENV_PREFIX}/bin/python" -c "import boltz" >/dev/null 2>&1; then
    log "      boltz already importable; skipping"
else
    "${ENV_PREFIX}/bin/python" -m pip install --upgrade pip
    "${ENV_PREFIX}/bin/python" -m pip install boltz
fi

log "[5/5] Sanity check"
"${ENV_PREFIX}/bin/boltz" --help >/dev/null 2>&1 && log "      boltz CLI works"

log ""
log "=== Boltz environment ready ==="
log "  Env:     ${ENV_PREFIX}"
log "  Symlink: ${ENV_LINK}"
log "  Binary:  ${ENV_PREFIX}/bin/boltz"
log ""
log "Used by:"
log "  cogligand_config/model/boltz1_inference.yaml"
log "  cogligand_config/model/boltz2_inference.yaml"
log ""
log "Models auto-download on first inference run."
