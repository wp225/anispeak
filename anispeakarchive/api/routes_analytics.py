# anispeakarchive/api/routes_analytics.py

from flask import Blueprint, jsonify
from typing import Any, Dict
import lancedb
import pandas as pd
from datetime import datetime

from anispeakarchive.core.config import load_settings
from anispeakarchive.core.json_sanitize import sanitize_for_json

bp = Blueprint('analytics', __name__)
settings = load_settings("configs/default.yaml")

def get_top_n(df: pd.DataFrame, col: str, n: int = 15) -> list[dict]:
    """Explodes a list column and returns top N counts efficiently."""
    if col not in df.columns: return []
    exploded = df.explode(col)
    exploded = exploded[exploded[col].apply(lambda x: isinstance(x, str) and bool(x.strip()))]

    # Demo hygiene / schema sentinels
    if col == "contexts":
        # Merge "not_applicable" into "Other"
        exploded[col] = exploded[col].apply(lambda s: "Other" if str(s).strip().lower() == "not_applicable" else s)
    if col == "species":
        # Hide "Others" by default (keeps taxonomy signal in leaderboards)
        exploded = exploded[exploded[col].apply(lambda s: str(s).strip().lower() not in {"others", "other"})]
    counts = exploded[col].value_counts().head(n)
    return [{"name": str(k), "count": int(v)} for k, v in counts.items()]

def get_detailed_profiles(df: pd.DataFrame, target_col: str, n: int = 100, years_back: int = 8) -> list[dict]:
    """Gets top N active entities over X years, plus their associated metadata tags using optimized groupby."""
    current_year = datetime.now().year
    years = list(range(current_year - years_back + 1, current_year + 1))

    if target_col not in df.columns: return []

    # 1. Explode only the target column to find the Top N entities
    exp_df = df.explode(target_col)
    exp_df = exp_df[exp_df[target_col].apply(lambda x: isinstance(x, str) and bool(x.strip()))]

    top_items = exp_df[target_col].value_counts().head(n).index.tolist()

    # 2. Filter down to ONLY these top 100 entities to make cross-referencing lightning fast
    top_df = exp_df[exp_df[target_col].isin(top_items)]

    profiles = []
    # 3. Group by the entity and extract their specific traits
    for item, group in top_df.groupby(target_col):
        year_counts = group['year'].value_counts().to_dict()

        def get_top_tags(col_name, limit=3):
            if col_name not in group.columns: return []
            tags = group.explode(col_name)[col_name]
            tags = tags[tags.apply(lambda x: isinstance(x, str) and bool(x.strip()))]
            return tags.value_counts().head(limit).index.tolist()

        profile = {
            "name": str(item),
            "total": len(group),
            "years": {str(y): int(year_counts.get(y, 0)) for y in years},
            "top_species": get_top_tags('species', 4),
            "top_venues": get_top_tags('venue', 2) if 'venue' in group.columns else []
        }

        if target_col == 'authors':
            profile["top_institutions"] = get_top_tags('institutions', 2)
        elif target_col == 'institutions':
            profile["top_authors"] = get_top_tags('authors', 3)

        profiles.append(profile)

    # Sort descending by total papers
    profiles.sort(key=lambda x: x['total'], reverse=True)
    return profiles

@bp.route("/api/analytics/extensive", methods=["GET"])
def get_extensive_analytics():
    db = lancedb.connect(str(settings.paths.lancedb_dir))
    tbl = db.open_table(settings.index.table_name)
    schema_names = tbl.schema.names

    cols_to_load = ["year", "species", "authors", "institutions", "domains", "features", "modalities", "contexts", "venue"]
    cols_to_load = [c for c in cols_to_load if c in schema_names]

    df = tbl.search().select(cols_to_load).to_pandas()

    # 1. Timeline
    year_counts = df['year'].dropna().astype(int).value_counts().sort_index()
    timeline = [{"year": int(y), "count": int(c)} for y, c in year_counts.items() if y > 1950]

    # 2. Avg Authors per Paper
    avg_authors = 0
    if 'authors' in df.columns:
        auth_lens = df['authors'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        avg_authors = round(auth_lens.mean(), 2) if not auth_lens.empty else 0

    current_year = datetime.now().year
    last_8_years = list(range(current_year - 7, current_year + 1))

    response_data = {
        "overview": {
            "papers_total": len(df),
            "species_total": df['species'].explode().nunique() if 'species' in df.columns else 0,
            "authors_total": df['authors'].explode().nunique() if 'authors' in df.columns else 0,
            "institutions_total": df['institutions'].explode().nunique() if 'institutions' in df.columns else 0,
            "avg_authors_per_paper": avg_authors
        },
        "timeline": timeline,
        "recent_years": last_8_years,
        "profiles": {
            "authors": get_detailed_profiles(df, "authors", 100, 8) if 'authors' in df.columns else [],
            "institutions": get_detailed_profiles(df, "institutions", 100, 8) if 'institutions' in df.columns else []
        },
        "leaderboards": {
            "venues": get_top_n(df, "venue", 15) if 'venue' in df.columns else [],
            "domains": get_top_n(df, "domains", 100),
            "modalities": get_top_n(df, "modalities", 100),
            "features": get_top_n(df, "features", 100),
            "contexts": get_top_n(df, "contexts", 100),
            "species": get_top_n(df, "species", 100)
        }
    }

    return jsonify(sanitize_for_json(response_data))
