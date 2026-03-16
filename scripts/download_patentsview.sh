#!/usr/bin/env bash
# Manual fallback: download PatentsView bulk data with curl.
# Run from the project root: bash scripts/download_patentsview.sh
#
# Uses curl -C - for automatic resume on interruption.
# S3 URLs verified working as of 2026-03-16.

set -euo pipefail

BASE="https://s3.amazonaws.com/data.patentsview.org/download"
DEST="data/raw"

mkdir -p "$DEST"

TABLES=(
    "g_patent"
    "g_us_patent_citation"
    "g_cpc_current"
)

for table in "${TABLES[@]}"; do
    url="${BASE}/${table}.tsv.zip"
    out="${DEST}/${table}.tsv.zip"
    tsv="${DEST}/${table}.tsv"

    if [ -f "$tsv" ]; then
        echo "[skip] $tsv already exists"
        continue
    fi

    echo "[download] $table..."
    curl -C - -L -o "$out" "$url"

    echo "[extract] $table..."
    unzip -o "$out" -d "$DEST"
    rm -f "$out"
    echo "[done] $tsv"
done

echo ""
echo "All tables downloaded to $DEST/"
ls -lh "$DEST"/*.tsv
