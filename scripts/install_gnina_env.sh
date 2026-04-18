#!/usr/bin/env bash
# Install GNINA for CogLigandBench.
#
# GNINA ships as a single static binary. This script verifies the binary at
# forks/GNINA/gnina is present (downloading the latest release if not), then
# symlinks envs/gnina -> forks/GNINA/ so the binary lives at envs/gnina/gnina.
#
# Idempotent: safe to re-run.
# Requires: GPU recommended.
# Usage: bash scripts/install_gnina_env.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GNINA_DIR="${PROJECT_ROOT}/forks/GNINA"
GNINA_BIN="${GNINA_DIR}/gnina"
ENV_LINK="${PROJECT_ROOT}/envs/gnina"
GNINA_URL="https://github.com/gnina/gnina/releases/latest/download/gnina"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "[1/3] Ensuring forks/GNINA/ exists"
mkdir -p "${GNINA_DIR}"

log "[2/3] Verifying GNINA binary at ${GNINA_BIN}"
if [ -x "${GNINA_BIN}" ]; then
    log "      gnina binary present; skipping download"
else
    log "      downloading from ${GNINA_URL}"
    curl -L -o "${GNINA_BIN}" "${GNINA_URL}"
    chmod +x "${GNINA_BIN}"
fi

log "[3/3] Symlinking ${ENV_LINK} -> ${GNINA_DIR}"
mkdir -p "${PROJECT_ROOT}/envs"
ln -sfn "${GNINA_DIR}" "${ENV_LINK}"

log ""
"${GNINA_BIN}" --version 2>/dev/null | head -5 || log "Note: gnina --version may need GPU runtime libs"
log ""
log "=== GNINA environment ready ==="
log "  Binary:  ${GNINA_BIN}"
log "  Symlink: ${ENV_LINK} (-> ${GNINA_DIR})"
log ""
log "Used by: cogligand_config/model/gnina_inference.yaml"
