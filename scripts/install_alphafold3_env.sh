#!/usr/bin/env bash
# Install AlphaFold3 for CogLigandBench.
#
# Creates a dedicated conda env at /mnt/katritch_lab2/aoxu/envs/alphafold3,
# symlinks it from CogLigandBench/envs/alphafold3, clones the AlphaFold3
# upstream repo into forks/alphafold3/alphafold3, pip-installs it into the
# env, and decompresses af3.bin.zst → forks/alphafold3/models/af3.bin.
#
# Idempotent: safe to re-run. Each step skips when the target already exists.
#
# Usage: bash scripts/install_alphafold3_env.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PARENT="/mnt/katritch_lab2/aoxu/envs"
ENV_PREFIX="${ENV_PARENT}/alphafold3"
ENV_LINK="${PROJECT_ROOT}/envs/alphafold3"
AF3_SRC="${PROJECT_ROOT}/forks/alphafold3/alphafold3"
AF3_MODELS="${PROJECT_ROOT}/forks/alphafold3/models"
WEIGHTS_ZST="${PROJECT_ROOT}/af3.bin.zst"
WEIGHTS_BIN="${AF3_MODELS}/af3.bin"

echo "[1/8] Ensuring env parent directory exists at ${ENV_PARENT}"
mkdir -p "${ENV_PARENT}"

echo "[2/8] Creating conda env at ${ENV_PREFIX}"
if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
    conda create -y -p "${ENV_PREFIX}" \
        -c bioconda -c conda-forge \
        python=3.12 hmmer
else
    echo "      env already exists; skipping create"
fi

echo "[3/8] Symlinking ${ENV_LINK} → ${ENV_PREFIX}"
mkdir -p "${PROJECT_ROOT}/envs"
ln -sfn "${ENV_PREFIX}" "${ENV_LINK}"

echo "[4/8] Cloning AlphaFold3 source to ${AF3_SRC}"
if [ ! -d "${AF3_SRC}/.git" ]; then
    git clone https://github.com/google-deepmind/alphafold3 "${AF3_SRC}"
else
    echo "      AF3 source already cloned; skipping"
fi

echo "[5/8] pip-installing AlphaFold3 into ${ENV_PREFIX}"
if ! "${ENV_PREFIX}/bin/python" -c "import alphafold3" >/dev/null 2>&1; then
    # Install boost-cpp + zlib (C++ build deps for cifpp/pybind11)
    conda install -y -p "${ENV_PREFIX}" -c conda-forge boost-cpp zlib
    "${ENV_PREFIX}/bin/python" -m pip install --upgrade pip
    if [ -f "${AF3_SRC}/dev-requirements.txt" ]; then
        "${ENV_PREFIX}/bin/python" -m pip install -r "${AF3_SRC}/dev-requirements.txt"
    fi
    # CMAKE_PREFIX_PATH tells CMake where to find conda env's zlib/boost headers
    CMAKE_PREFIX_PATH="${ENV_PREFIX}" "${ENV_PREFIX}/bin/python" -m pip install -e "${AF3_SRC}"
else
    echo "      alphafold3 already importable; skipping pip install"
fi

echo "[6/8] Building intermediate data (CCD pickle files)"
CCD_PICKLE="${AF3_SRC}/src/alphafold3/constants/converters/ccd.pickle"
if [ ! -f "${CCD_PICKLE}" ]; then
    "${ENV_PREFIX}/bin/python" -c "from alphafold3.build_data import build_data; build_data()"
else
    echo "      CCD pickle already exists; skipping build_data"
fi

echo "[7/8] Decompressing weights → ${WEIGHTS_BIN}"
mkdir -p "${AF3_MODELS}"
if [ ! -f "${WEIGHTS_BIN}" ]; then
    if [ ! -f "${WEIGHTS_ZST}" ]; then
        echo "ERROR: ${WEIGHTS_ZST} not found. Place af3.bin.zst at the project root and re-run." >&2
        exit 1
    fi
    if command -v zstd >/dev/null 2>&1; then
        zstd -d "${WEIGHTS_ZST}" -o "${WEIGHTS_BIN}"
    else
        echo "      zstd CLI not found; using Python zstandard fallback"
        "${ENV_PREFIX}/bin/python" -m pip install --quiet zstandard
        "${ENV_PREFIX}/bin/python" - <<PY
import zstandard, pathlib
src = pathlib.Path("${WEIGHTS_ZST}")
dst = pathlib.Path("${WEIGHTS_BIN}")
with src.open("rb") as fh_in, dst.open("wb") as fh_out:
    zstandard.ZstdDecompressor().copy_stream(fh_in, fh_out)
PY
    fi
else
    echo "      weights already present; skipping"
fi

echo "[8/8] Sanity check: importing alphafold3 from the env"
"${ENV_PREFIX}/bin/python" -c "import alphafold3; print('alphafold3 import OK')"

echo
echo "Done. AF3 is installed at:"
echo "  env:     ${ENV_PREFIX}  (symlinked from ${ENV_LINK})"
echo "  source:  ${AF3_SRC}"
echo "  weights: ${WEIGHTS_BIN}"
echo
echo "Next: run '${PROJECT_ROOT}/envs/alphafold3/bin/python ${AF3_SRC}/run_alphafold.py --help' to see the CLI surface (Task 10)."
