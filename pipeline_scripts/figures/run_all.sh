#!/usr/bin/env bash
# Run all figure scripts. All paths are resolved relative to this script's directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CFG="$REPO_ROOT/configs/default.yaml"

python "$SCRIPT_DIR/plot_trends_two_groups.py"         --config "$CFG" --out_dir "$SCRIPT_DIR/output/trends"
python "$SCRIPT_DIR/plot_donuts_triptych.py"           --config "$CFG" --out_dir "$SCRIPT_DIR/output/donuts"
python "$SCRIPT_DIR/plot_top_authors_dual.py"          --config "$CFG" --out_dir "$SCRIPT_DIR/output/authors"
python "$SCRIPT_DIR/plot_top_institutions_dual.py"     --config "$CFG" --out_dir "$SCRIPT_DIR/output/institutions"
python "$SCRIPT_DIR/plot_top_features_dual.py"         --config "$CFG" --out_dir "$SCRIPT_DIR/output/features"
python "$SCRIPT_DIR/plot_top_venues_dual.py"           --config "$CFG" --out_dir "$SCRIPT_DIR/output/venues"
python "$SCRIPT_DIR/plot_top_species_dual.py"          --config "$CFG" --out_dir "$SCRIPT_DIR/output/species"     --top_n 15
python "$SCRIPT_DIR/plot_facet_prepost_2016.py"        --config "$CFG" --out_dir "$SCRIPT_DIR/output/facets_prepost_2000" --top_n 10
python "$SCRIPT_DIR/plot_wordcloud_prepost_2016.py"    --config "$CFG" --out_dir "$SCRIPT_DIR/output/wordclouds_prepost_2000"
python "$SCRIPT_DIR/plot_tfidf_terms_prepost_2016.py"  --config "$CFG" --out_dir "$SCRIPT_DIR/output/tfidf_prepost_2000"  --top_n 20

echo "Done. Output written to $SCRIPT_DIR/output/"
