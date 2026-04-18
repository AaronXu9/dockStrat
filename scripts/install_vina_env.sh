#!/usr/bin/env bash
# Install AutoDock Vina + Open Babel for CogLigandBench.
#
# Creates a conda env at /mnt/katritch_lab2/aoxu/envs/vina with the
# `vina` and `obabel` binaries on $PATH, symlinks it from
# CogLigandBench/envs/vina.
#
# Idempotent: safe to re-run.
# Usage: bash scripts/install_vina_env.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PARENT="/mnt/katritch_lab2/aoxu/envs"
ENV_PREFIX="${ENV_PARENT}/vina"
ENV_LINK="${PROJECT_ROOT}/envs/vina"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "[1/4] Ensuring env parent directory exists: ${ENV_PARENT}"
mkdir -p "${ENV_PARENT}"

log "[2/4] Creating conda env at ${ENV_PREFIX} with vina + openbabel"
if [ -x "${ENV_PREFIX}/bin/vina" ] && [ -x "${ENV_PREFIX}/bin/obabel" ]; then
    log "      env already has vina + obabel; skipping create"
else
    conda create -y -p "${ENV_PREFIX}" -c conda-forge \
        python=3.11 vina openbabel
fi

log "[3/4] Symlinking ${ENV_LINK} -> ${ENV_PREFIX}"
mkdir -p "${PROJECT_ROOT}/envs"
ln -sfn "${ENV_PREFIX}" "${ENV_LINK}"

log "[4/4] Sanity check"
"${ENV_PREFIX}/bin/vina" --version
"${ENV_PREFIX}/bin/obabel" -V

log ""
log "=== Vina environment ready ==="
log "  Env:     ${ENV_PREFIX}"
log "  Symlink: ${ENV_LINK}"
log "  Vina:    ${ENV_PREFIX}/bin/vina"
log "  obabel:  ${ENV_PREFIX}/bin/obabel"
log ""
log "Activate with: conda activate ${ENV_PREFIX}"
log "Or prepend ${ENV_PREFIX}/bin to your PATH before running dock_engine('vina', ...)."
