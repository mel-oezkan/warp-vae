#!/usr/bin/env bash
# Download CO3D v2 sequences for: bench, toyplane, backpack, car
# Stops after collecting MAX_SEQS sequences per category.
# Output layout: OUTPUT_DIR/co3d/<category>/<sequence_id>/

set -euo pipefail

OUTPUT_DIR="${1:-/visinf/projects_students/dlcv2025_groupZ}"
BASE_URL="https://dl.fbaipublicfiles.com/co3dv2_231130"
MAX_SEQS=50
TEMP_DIR="$OUTPUT_DIR/_tmp_zips"

declare -A CATEGORY_FILES=(
    ["bench"]="bench_000 bench_001 bench_002 bench_003 bench_004 bench_005 bench_006 bench_007 bench_008"
    ["toyplane"]="toyplane_000 toyplane_001 toyplane_002 toyplane_003"
    ["backpack"]="backpack_000 backpack_001 backpack_002 backpack_003 backpack_004 backpack_005 backpack_006 backpack_007 backpack_008 backpack_009 backpack_010 backpack_011 backpack_012"
    ["car"]="car_000 car_001 car_002 car_003 car_004 car_005"
)

mkdir -p "$TEMP_DIR"

for category in bench toyplane backpack car; do
    cat_dir="$OUTPUT_DIR/co3d/$category"
    mkdir -p "$cat_dir"
    echo "=== Category: $category (target: $MAX_SEQS sequences) ==="

    seq_count=$(find "$cat_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "  Already have $seq_count sequences."

    for stem in ${CATEGORY_FILES[$category]}; do
        if [ "$seq_count" -ge "$MAX_SEQS" ]; then
            echo "  Reached $MAX_SEQS sequences, stopping."
            break
        fi

        zip_path="$TEMP_DIR/${stem}.zip"
        url="$BASE_URL/${stem}.zip"

        echo "  Downloading ${stem}.zip ..."
        wget -q --show-progress -O "$zip_path" "$url"

        echo "  Extracting ${stem}.zip ..."
        # CO3D zips extract to <category>/<seq_id>/ — unzip into a temp staging dir
        stage_dir="$TEMP_DIR/${stem}_stage"
        mkdir -p "$stage_dir"
        unzip -q -o "$zip_path" -d "$stage_dir"
        rm "$zip_path"

        # Move sequences into cat_dir until we hit MAX_SEQS
        for seq_path in "$stage_dir/$category"/*/; do
            [ -d "$seq_path" ] || continue
            if [ "$seq_count" -ge "$MAX_SEQS" ]; then
                break
            fi
            seq_name=$(basename "$seq_path")
            if [ ! -d "$cat_dir/$seq_name" ]; then
                mv "$seq_path" "$cat_dir/$seq_name"
                seq_count=$((seq_count + 1))
                echo "    Added sequence $seq_name ($seq_count/$MAX_SEQS)"
            fi
        done

        rm -rf "$stage_dir"
    done

    echo "=== Done $category: $seq_count sequences in $cat_dir ==="
done

rm -rf "$TEMP_DIR"

echo ""
echo "All downloads complete."
echo "Data layout: $OUTPUT_DIR/co3d/<category>/<sequence_id>/"
