#!/usr/bin/env bash
# Install SurfDock for CogLigandBench.
#
# Creates a conda env at /mnt/katritch_lab2/aoxu/envs/surfdock with the
# SurfDock dependencies plus MSMS, APBS, and pdb2pqr binaries needed by
# the surface preprocessing pipeline. Symlinks envs/surfdock to it.
#
# Note: SurfDock is complex. The full reference install lives at
# /home/aoxu/projects/SurfDock (set as `surfdock_dir` in the YAML config),
# which contains MSMS/APBS binaries (~231 MB) and an editable ESM install.
# This script creates a fresh env using environments/surfdock_environment.yaml
# (if present) and verifies prerequisites.
#
# Idempotent: safe to re-run.
# Requires: conda, CUDA GPU.
# Usage: bash scripts/install_surfdock_env.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PARENT="/mnt/katritch_lab2/aoxu/envs"
ENV_PREFIX="${ENV_PARENT}/surfdock"
ENV_LINK="${PROJECT_ROOT}/envs/surfdock"
ENV_YAML="${PROJECT_ROOT}/environments/surfdock_environment.yaml"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "[1/5] Ensuring env parent directory exists: ${ENV_PARENT}"
mkdir -p "${ENV_PARENT}"

log "[2/5] Creating conda env at ${ENV_PREFIX}"
if [ -x "${ENV_PREFIX}/bin/python" ]; then
    log "      env already exists; skipping create"
elif [ -f "${ENV_YAML}" ]; then
    conda env create -p "${ENV_PREFIX}" -f "${ENV_YAML}"
else
    log "      ${ENV_YAML} not found; creating bare Python 3.10 env"
    log "      You will need to install SurfDock dependencies manually."
    conda create -y -p "${ENV_PREFIX}" python=3.10
fi

log "[3/5] Symlinking ${ENV_LINK} -> ${ENV_PREFIX}"
mkdir -p "${PROJECT_ROOT}/envs"
ln -sfn "${ENV_PREFIX}" "${ENV_LINK}"

log "[4/5] Verifying surface tools (MSMS / APBS / pdb2pqr)"
SURFDOCK_DIR="${SURFDOCK_DIR:-/home/aoxu/projects/SurfDock}"
if [ -d "${SURFDOCK_DIR}/comp_surface/tools" ]; then
    log "      surface tools found at ${SURFDOCK_DIR}/comp_surface/tools"
else
    log "      WARNING: surface tools not found at ${SURFDOCK_DIR}/comp_surface/tools."
    log "      Either set SURFDOCK_DIR env var or copy/symlink the tools manually."
    log "      See cogligand_config/model/surfdock_inference.yaml: surfdock_dir field."
fi

log "[5/5] Verifying precomputed arrays"
PRECOMPUTED="${PRECOMPUTED_ARRAYS:-/home/aoxu/projects/precomputed/precomputed_arrays}"
if [ -d "${PRECOMPUTED}" ]; then
    log "      precomputed arrays found at ${PRECOMPUTED}"
else
    log "      WARNING: precomputed arrays not found at ${PRECOMPUTED}."
    log "      SurfDock requires this at import time (utils/torus.py, utils/so3.py)."
fi

log ""
log "=== SurfDock environment ready ==="
log "  Env:     ${ENV_PREFIX}"
log "  Symlink: ${ENV_LINK}"
log "  surfdock_dir: ${SURFDOCK_DIR}"
log "  precomputed_arrays: ${PRECOMPUTED}"
log ""
log "Used by: cogligand_config/model/surfdock_inference.yaml"
