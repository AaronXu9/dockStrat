#!/usr/bin/env bash
# prepare_zenodo_cofolding.sh
#
# Companion script to prepare_zenodo_deposit.sh.
# Extends an existing zenodo_staging/ directory with co-folding model data:
# AF3, Boltz, Chai, Protenix.
#
# Usage:
#   bash scripts/prepare_zenodo_cofolding.sh              # copy only
#   bash scripts/prepare_zenodo_cofolding.sh --compress   # copy + compress
#   bash scripts/prepare_zenodo_cofolding.sh --skip-copy --compress
#
# Does NOT modify any existing docking archives — only adds cofolding_*.tar.gz
# and regenerates MANIFEST.md, SHA256SUMS.txt, README.md.

set -euo pipefail
ulimit -n 8192

# ── Configuration ──────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGING="${REPO_ROOT}/zenodo_staging"

COFOLD_CSV_SRC="${REPO_ROOT}/data/runsNposes_archive/zenodo_downloads"
COFOLD_POSES_SRC="/mnt/katritch_lab2/aoxu/data/runsNposes/top5_poses_tweaked"

METHODS=(af3 boltz chai protenix)

DO_COMPRESS=false
SKIP_COPY=false
for arg in "$@"; do
    case "$arg" in
        --compress)  DO_COMPRESS=true ;;
        --skip-copy) SKIP_COPY=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Phase A: Preflight checks ──────────────────────────────────────────
log "Phase A: Preflight checks"

[[ -d "$STAGING" ]] || {
    echo "ERROR: $STAGING missing. Run prepare_zenodo_deposit.sh first."
    exit 1
}

# Existing docking archives must be present
for f in results_tables.tar.gz docking_poses_vina.tar.gz input_icm_docking_maps.tar.gz; do
    [[ -f "$STAGING/$f" ]] || {
        echo "ERROR: missing $STAGING/$f — docking deposit incomplete."
        exit 1
    }
done

# All source CSVs and pose dirs must exist
for m in "${METHODS[@]}"; do
    [[ -f "$COFOLD_CSV_SRC/predictions/$m.csv" ]] || { echo "ERROR: missing predictions/$m.csv"; exit 1; }
    [[ -f "$COFOLD_CSV_SRC/posebusters_results/$m.csv" ]] || { echo "ERROR: missing posebusters_results/$m.csv"; exit 1; }
    [[ -d "$COFOLD_POSES_SRC/$m" ]] || { echo "ERROR: missing $COFOLD_POSES_SRC/$m"; exit 1; }
done

avail_gb=$(df -BG "$STAGING" | awk 'NR==2 {gsub("G",""); print $4}')
if (( avail_gb < 15 )); then
    echo "WARNING: only ${avail_gb} GB free — recommended >=15 GB"
fi

log "  Preflight OK. Staging: $STAGING"

# ── Phase B: Directory skeleton ────────────────────────────────────────
if ! $SKIP_COPY; then
    log "Phase B: Creating co-folding directory structure"
    mkdir -p "$STAGING/cofolding_results_tables/predictions"
    mkdir -p "$STAGING/cofolding_results_tables/posebusters_results"
    for m in "${METHODS[@]}"; do
        mkdir -p "$STAGING/cofolding_poses/$m"
    done
fi

# ── Phase C: Copy co-folding CSVs ──────────────────────────────────────
if ! $SKIP_COPY; then
    log "Phase C: Copying co-folding CSV tables"
    for m in "${METHODS[@]}"; do
        cp "$COFOLD_CSV_SRC/predictions/$m.csv" \
           "$STAGING/cofolding_results_tables/predictions/"
        cp "$COFOLD_CSV_SRC/posebusters_results/$m.csv" \
           "$STAGING/cofolding_results_tables/posebusters_results/"
        log "  Copied $m.csv: $(wc -l < "$STAGING/cofolding_results_tables/predictions/$m.csv") prediction rows"
    done
fi

# ── Phase D: Copy top-5 poses ──────────────────────────────────────────
if ! $SKIP_COPY; then
    log "Phase D: Copying top-5 poses (~10.7 GB via rsync)"
    # --no-o --no-g: skip owner/group preservation (NFS rejects chgrp)
    # Tolerate exit 23 (partial transfer / vanished source files) from cross-FS copies
    for m in "${METHODS[@]}"; do
        log "  rsync $m ..."
        set +e
        rsync -rlptD --no-o --no-g \
            --include='*/' \
            --include='ranking_scores.csv' \
            --include='seed-*_sample-*.cif' \
            --exclude='*' \
            "$COFOLD_POSES_SRC/$m/" "$STAGING/cofolding_poses/$m/"
        rc=$?
        set -e
        if [[ $rc -ne 0 && $rc -ne 23 && $rc -ne 24 ]]; then
            echo "ERROR: rsync failed for $m with exit code $rc"
            exit $rc
        fi

        nsys=$(find "$STAGING/cofolding_poses/$m" -mindepth 1 -maxdepth 1 -type d | wc -l)
        ncif=$(find "$STAGING/cofolding_poses/$m" -name '*.cif' | wc -l)
        nrank=$(find "$STAGING/cofolding_poses/$m" -name 'ranking_scores.csv' | wc -l)
        log "    $m: $nsys systems, $ncif CIFs, $nrank ranking_scores.csv"
    done
fi

# ── Phase E: Refresh README.md ─────────────────────────────────────────
log "Phase E: Refreshing README.md from scripts/zenodo_README.md"
if [[ -f "$REPO_ROOT/scripts/zenodo_README.md" ]]; then
    cp "$REPO_ROOT/scripts/zenodo_README.md" "$STAGING/README.md"
else
    log "  WARNING: scripts/zenodo_README.md not found — skipping README refresh"
fi

# ── Phase F: Regenerate MANIFEST.md ────────────────────────────────────
log "Phase F: Regenerating MANIFEST.md"

cat > "$STAGING/MANIFEST.md" << HEADER
# CogLigandBench Zenodo Deposit — File Manifest

Generated on: $(date '+%Y-%m-%d %H:%M:%S')

## Co-folding staging directories

| Directory | Files | Size |
|-----------|-------|------|
HEADER

for subdir in \
    cofolding_results_tables/predictions \
    cofolding_results_tables/posebusters_results \
    cofolding_poses/af3 \
    cofolding_poses/boltz \
    cofolding_poses/chai \
    cofolding_poses/protenix
do
    dir="$STAGING/$subdir"
    if [[ -d "$dir" ]]; then
        nfiles=$(find "$dir" -type f | wc -l)
        size=$(du -sh "$dir" | cut -f1)
        echo "| $subdir | $nfiles | $size |" >> "$STAGING/MANIFEST.md"
    fi
done

{
    echo ""
    echo "## All compressed archives (for Zenodo upload)"
    echo ""
    echo "| Archive | Size |"
    echo "|---------|------|"
} >> "$STAGING/MANIFEST.md"

# List all tar.gz archives with sizes
for f in "$STAGING"/*.tar.gz; do
    if [[ -f "$f" ]]; then
        echo "| $(basename "$f") | $(du -sh "$f" | cut -f1) |" >> "$STAGING/MANIFEST.md"
    fi
done

{
    echo ""
    total_archives=$(ls "$STAGING"/*.tar.gz 2>/dev/null | wc -l)
    total_size=$(du -shc "$STAGING"/*.tar.gz 2>/dev/null | tail -1 | cut -f1)
    echo "**Total: $total_archives archives, $total_size**"
} >> "$STAGING/MANIFEST.md"

log "  Manifest: $STAGING/MANIFEST.md"

# ── Phase G: Compress (optional) ───────────────────────────────────────
if $DO_COMPRESS; then
    log "Phase G: Creating compressed co-folding archives"
    cd "$STAGING"

    if command -v pigz &>/dev/null; then
        GZIP_CMD="pigz -p 8"
    else
        GZIP_CMD="gzip"
    fi

    create_archive() {
        local name="$1"
        local src="$2"
        log "  Compressing ${name}.tar.gz ..."
        tar cf - -C "$(dirname "$src")" "$(basename "$src")" | $GZIP_CMD > "${name}.tar.gz"
        log "    Done: $(du -sh "${name}.tar.gz" | cut -f1)"
    }

    create_archive "cofolding_results_tables" "cofolding_results_tables"
    for m in "${METHODS[@]}"; do
        create_archive "cofolding_poses_$m" "cofolding_poses/$m"
    done

    cd "$REPO_ROOT"
fi

# ── Phase H: Regenerate SHA256SUMS.txt over ALL archives ──────────────
log "Phase H: Regenerating SHA256SUMS.txt over all archives"
cd "$STAGING"
if ls *.tar.gz &>/dev/null; then
    sha256sum *.tar.gz > SHA256SUMS.txt
    log "  Hashed $(wc -l < SHA256SUMS.txt) archives"
else
    log "  No .tar.gz files to hash"
fi
cd "$REPO_ROOT"

# ── Phase I: Summary ──────────────────────────────────────────────────
log ""
log "=== Co-folding addition complete ==="
log "Staging total: $(du -sh "$STAGING" | cut -f1)"
log ""
log "Co-folding systems per method:"
for m in "${METHODS[@]}"; do
    count=$(ls "$STAGING/cofolding_poses/$m" 2>/dev/null | wc -l)
    echo "  $m: $count systems"
done
log ""
log "Co-folding prediction CSVs:"
wc -l "$STAGING"/cofolding_results_tables/predictions/*.csv 2>/dev/null || echo "  (none)"
log ""
log "Co-folding posebusters CSVs:"
wc -l "$STAGING"/cofolding_results_tables/posebusters_results/*.csv 2>/dev/null || echo "  (none)"
log ""
log "All archives in deposit:"
ls -lh "$STAGING"/*.tar.gz 2>/dev/null | awk '{print "  " $NF ": " $5}'
log "Done."
