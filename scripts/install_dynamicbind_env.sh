#!/usr/bin/env bash
# Install DynamicBind for CogLigandBench.
#
# DynamicBind ships with its own bundled conda env at
# forks/DynamicBind/DynamicBind/. This script verifies the bundled env is
# present (cloning the upstream repo if not) and symlinks envs/dynamicbind
# to it.
#
# Idempotent: safe to re-run.
# Requires: GPU.
# Usage: bash scripts/install_dynamicbind_env.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_DIR="${PROJECT_ROOT}/forks/DynamicBind"
DB_ENV="${DB_DIR}/DynamicBind"
DB_PYTHON="${DB_ENV}/bin/python3"
ENV_LINK="${PROJECT_ROOT}/envs/dynamicbind"
DB_URL="https://github.com/luwei0917/DynamicBind"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "[1/4] Ensuring forks/DynamicBind/ exists"
mkdir -p "${DB_DIR}"

log "[2/4] Verifying DynamicBind upstream at ${DB_DIR}"
if [ -f "${DB_DIR}/run_single_protein_inference.py" ]; then
    log "      DynamicBind source already present"
else
    log "      cloning from ${DB_URL}"
    git clone "${DB_URL}" "${DB_DIR}"
fi

log "[3/4] Verifying bundled Python env at ${DB_ENV}"
if [ -x "${DB_PYTHON}" ]; then
    log "      bundled env present at ${DB_ENV}"
else
    log ""
    log "ERROR: bundled DynamicBind env not found at ${DB_ENV}."
    log "DynamicBind ships a portable conda env from upstream. Either:"
    log "  (1) Download the bundled env from the DynamicBind release page and extract"
    log "      it to ${DB_ENV}, or"
    log "  (2) Create a fresh env from forks/DynamicBind/environment.yml:"
    log "      conda env create -p ${DB_ENV} -f ${DB_DIR}/environment.yml"
    exit 1
fi

log "[4/4] Symlinking ${ENV_LINK} -> ${DB_ENV}"
mkdir -p "${PROJECT_ROOT}/envs"
ln -sfn "${DB_ENV}" "${ENV_LINK}"

log ""
log "=== DynamicBind environment ready ==="
log "  Env:     ${DB_ENV}"
log "  Symlink: ${ENV_LINK}"
log "  Python:  ${DB_PYTHON}"
log "  Script:  ${DB_DIR}/run_single_protein_inference.py"
log ""
log "Used by: cogligand_config/model/dynamicbind_inference.yaml"
