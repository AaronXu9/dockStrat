#!/usr/bin/env bash
# Install UniDock2 for CogLigandBench.
#
# Creates a conda env at /mnt/katritch_lab2/aoxu/envs/unidock2 with the
# `unidock2` CLI binary, symlinks it from CogLigandBench/envs/unidock2.
#
# Idempotent: safe to re-run.
# Requires: conda, CUDA GPU.
# Usage: bash scripts/install_unidock2_env.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PARENT="/mnt/katritch_lab2/aoxu/envs"
ENV_PREFIX="${ENV_PARENT}/unidock2"
ENV_LINK="${PROJECT_ROOT}/envs/unidock2"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "[1/4] Ensuring env parent directory exists: ${ENV_PARENT}"
mkdir -p "${ENV_PARENT}"

log "[2/4] Creating conda env at ${ENV_PREFIX} with unidock"
if [ -x "${ENV_PREFIX}/bin/unidock2" ] || [ -x "${ENV_PREFIX}/bin/unidock" ]; then
    log "      unidock already present; skipping create"
else
    conda create -y -p "${ENV_PREFIX}" -c conda-forge python=3.11 unidock
fi

log "[3/4] Symlinking ${ENV_LINK} -> ${ENV_PREFIX}"
mkdir -p "${PROJECT_ROOT}/envs"
ln -sfn "${ENV_PREFIX}" "${ENV_LINK}"

log "[4/4] Sanity check"
if [ -x "${ENV_PREFIX}/bin/unidock2" ]; then
    "${ENV_PREFIX}/bin/unidock2" --help 2>&1 | head -5 || true
elif [ -x "${ENV_PREFIX}/bin/unidock" ]; then
    "${ENV_PREFIX}/bin/unidock" --help 2>&1 | head -5 || true
else
    log "WARNING: neither unidock2 nor unidock binary found in ${ENV_PREFIX}/bin/"
fi

log ""
log "=== UniDock2 environment ready ==="
log "  Env:     ${ENV_PREFIX}"
log "  Symlink: ${ENV_LINK}"
log ""
log "Used by: cogligand_config/model/unidock2_inference.yaml"
log "Run via: conda run -p ${ENV_PREFIX} unidock2 docking -cf <config>"
