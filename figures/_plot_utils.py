from __future__ import annotations

import re
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import lancedb
from matplotlib.transforms import offset_copy

# Ensure repo root is on PYTHONPATH so `import anispeakarchive` works when run from figures/
REPO_ROOT = Path(__file__).resolve().parents[1]
assert REPO_ROOT.exists(), f"Repo root not found: {REPO_ROOT}"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anispeakarchive.core.config import load_settings

TRAD_CATS = {"biological", "computational", "hybrid"}
TREND_SAME_EPS_PCT = 1.0
TREND_SYMBOLS = {
    "up": "↑",
    "down": "↓",
    "same": "-",
    "new": "*",
}
TREND_COLORS = {
    "up": "#2E8B57",
    "down": "#C43C39",
    "same": "#C4872A",
    "new": "#6A4C93",
}

# CANON_MAP: hard overrides for inflected domain terms post-lemmatization
CANON_MAP: dict[str, str] = {
    "bats": "bat", "whales": "whale", "dolphins": "dolphin", "birds": "bird",
    "frogs": "frog", "mice": "mouse", "rats": "rat", "seals": "seal",
    "finches": "finch", "sparrows": "sparrow", "monkeys": "monkey",
    "crickets": "cricket", "insects": "insect", "fishes": "fish",
    "songbirds": "songbird", "warblers": "warbler", "porpoises": "porpoise",
    "primates": "primate", "cetaceans": "cetacean", "elephants": "elephant",
    "orcas": "orca", "bees": "bee", "grasshoppers": "grasshopper",
    "gerbils": "gerbil", "canaries": "canary", "wolves": "wolf",
    "crows": "crow", "ravens": "raven", "parrots": "parrot",
    "chimpanzees": "chimpanzee", "gorillas": "gorilla",
    "males": "male", "females": "female",
    "responses": "response", "neurons": "neuron", "pulses": "pulse",
    "clicks": "click", "whistles": "whistle", "chirps": "chirp",
    "repertoires": "repertoire", "syllables": "syllable",
    "echolocate": "echolocation",
    "focusing": "focus", "producing": "produce", "produced": "produce",
    "measuring": "measure", "measured": "measure",
    "detecting": "detect", "detected": "detect",
    "recognizing": "recognize", "recognised": "recognize",
    "monitoring": "monitor",
}


def canonicalize(token: str) -> str:
    """Map a lowercased token to its canonical domain form."""
    return CANON_MAP.get(token, token)


# ── Authority / year suffix patterns in taxonomic names ───────────────────────
# e.g. "Taeniopygia guttata (Vieillot, 1817)"  →  "taeniopygia guttata"
_AUTHORITY_RE = re.compile(r"\s*\([^)]*\d{4}[^)]*\)\s*$")
_YEAR_ONLY_RE = re.compile(r",?\s*\d{4}\s*$")

# Known common-name / synonym mappings (lowercase binomial → preferred label)
_SPECIES_SYNONYMS: dict[str, str] = {
    "taeniopygia guttata":          "Zebra finch",
    "poephila guttata":             "Zebra finch",        # older synonym
    "mus musculus":                 "House mouse",
    "rattus norvegicus":            "Norway rat",
    "pan troglodytes":              "Chimpanzee",
    "tursiops truncatus":           "Bottlenose dolphin",
    "physeter macrocephalus":       "Sperm whale",
    "megaptera novaeangliae":       "Humpback whale",
    "orcinus orca":                 "Killer whale",
    "apis mellifera":               "Honeybee",
    "drosophila melanogaster":      "Fruit fly",
    "columba livia":                "Rock pigeon",
    "melospiza melodia":            "Song sparrow",
    "serinus canaria":              "Domestic canary",
    "canis lupus familiaris":       "Dog",
    "canis familiaris":             "Dog",
    "felis catus":                  "Domestic cat",
    "xenopus laevis":               "African clawed frog",
    "homo sapiens":                 "Human",
}

# Simple English plural → singular rules (applied after lower-casing)
_PLURAL_RULES = [
    (re.compile(r"idae$"),    "idae"),      # family endings: keep as-is
    (re.compile(r"inae$"),    "inae"),      # subfamily: keep as-is
    (re.compile(r"ves$"),     "f"),         # wolves → wolf
    (re.compile(r"ies$"),     "y"),         # butterflies → butterfly
    (re.compile(r"ses$"),     "s"),         # grasses → grass
    (re.compile(r"(?<!s)s$"), ""),          # generic trailing -s
]


def norm_species_label(x: object) -> str:
    """Normalize a species label to a canonical, display-ready string."""
    if x is None:
        return "Other"
    s = str(x).strip()
    if not s:
        return "Other"

    sl = s.lower()

    # Null-ish values
    _nullish = {
        "na", "n/a", "none", "null", "unknown", "unk", "undefined",
        "not_applicable", "not applicable", "not-applicable", "n.a.",
        "notapplicable", "other", "others", "other species", "misc",
        "miscellaneous", "unspecified", "multiple", "various", "several",
    }
    if sl in _nullish or sl.startswith("other "):
        return "Other"

    # Strip taxonomic authority + year: "Taeniopygia guttata (Vieillot, 1817)"
    cleaned = _AUTHORITY_RE.sub("", s).strip()
    cleaned = _YEAR_ONLY_RE.sub("", cleaned).strip()
    key = cleaned.lower()

    # Check synonym table
    if key in _SPECIES_SYNONYMS:
        return _SPECIES_SYNONYMS[key]

    # Singularize common name (e.g. "zebra finches" → "zebra finch")
    # Only apply to labels that look like common names (no italics signal, >1 word, all ascii)
    words = key.split()
    if len(words) >= 1:
        last = words[-1]
        for pattern, replacement in _PLURAL_RULES:
            if pattern.search(last) and last not in {"species", "series", "lens"}:
                # Don't singularize taxonomic family/subfamily endings
                if not (last.endswith("idae") or last.endswith("inae")):
                    last = pattern.sub(replacement, last)
                    break
        words[-1] = last
        key = " ".join(words)

    # Re-check synonyms after singularization
    if key in _SPECIES_SYNONYMS:
        return _SPECIES_SYNONYMS[key]

    # Title-case for display (preserve all-caps acronyms)
    return cleaned.title() if cleaned == cleaned.lower() else cleaned


@dataclass(frozen=True)
class LoadSpec:
    config: str
    year_min: int = 1950
    year_max: int = 2026
    col_year: str = "year"
    col_trad: str = "research_tradition"


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"Missing columns: {missing}. Present: {list(df.columns)}"


def _normalize_tradition(x: object) -> str:
    if x is None:
        return "unknown"
    s = str(x).strip().lower()
    mapping = {
        "biological": "biological",
        "bio": "biological",
        "biology": "biological",
        "computational": "computational",
        "comp": "computational",
        "ml": "computational",
        "nlp": "computational",
        "hybrid": "hybrid",
        "bio+comp": "hybrid",
        "computational+biological": "hybrid",
        "biological+computational": "hybrid",
    }
    out = mapping.get(s, s)
    return out if out in TRAD_CATS else "unknown"


def _abs_path(p: str, base: Path) -> str:
    assert isinstance(p, str) and p.strip(), f"Invalid path: {p!r}"
    pp = Path(p)
    return str(pp if pp.is_absolute() else (base / pp).resolve())


def load_df(spec: LoadSpec) -> pd.DataFrame:
    cfg = load_settings(spec.config)

    lancedb_dir = _abs_path(str(cfg.paths.lancedb_dir), REPO_ROOT)
    assert Path(lancedb_dir).exists(), f"LanceDB dir does not exist: {lancedb_dir}"

    db = lancedb.connect(lancedb_dir)

    wanted = str(getattr(cfg.index, "table_name", "") or "")
    assert wanted, f"cfg.index.table_name is empty. cfg.index={getattr(cfg, 'index', None)!r}"

    tables = db.table_names()
    assert isinstance(tables, list), f"Unexpected db.table_names() type: {type(tables)}"
    assert len(tables) > 0, f"No tables found in LanceDB dir: {lancedb_dir}"

    if wanted not in tables:
        if len(tables) == 1:
            picked = tables[0]
            print(f"[plot_utils] Table '{wanted}' not found. Using only table present: '{picked}'")
            wanted = picked
        else:
            raise ValueError(
                f"Table '{wanted}' was not found in {lancedb_dir}. "
                f"Available tables: {tables}. "
                f"Fix configs/default.yaml index.table_name or point paths.lancedb_dir to the right DB."
            )

    tbl = db.open_table(wanted)
    assert tbl.count_rows() > 0, f"Table '{wanted}' is empty in {lancedb_dir}"

    df = tbl.to_pandas()
    _require_cols(df, [spec.col_year, spec.col_trad])

    df = df.copy()
    df[spec.col_year] = pd.to_numeric(df[spec.col_year], errors="coerce")
    df = df.dropna(subset=[spec.col_year])
    df[spec.col_year] = df[spec.col_year].astype(int)
    df = df[(df[spec.col_year] >= spec.year_min) & (df[spec.col_year] <= spec.year_max)]
    assert len(df) > 0, f"No rows in range {spec.year_min} to {spec.year_max}."

    df["trad3"] = df[spec.col_trad].map(_normalize_tradition)
    assert df["trad3"].isin(list(TRAD_CATS)).any(), (
        "No rows mapped to a known tradition. "
        f"Top values: {df['trad3'].value_counts().head(10).to_dict()}"
    )

    df["group2"] = df["trad3"].map(
        lambda t: "biological" if t == "biological"
        else ("comp_or_hybrid" if t in {"computational", "hybrid"} else "drop")
    )
    df = df[df["group2"].isin(["biological", "comp_or_hybrid"])].copy()
    assert len(df) > 0, "All rows were dropped during group2 filtering."
    return df


def assert_listlike(df: pd.DataFrame, col: str) -> None:
    assert col in df.columns, f"Missing column: {col}"
    non_null = df[col].dropna()
    if len(non_null) == 0:
        return

    def ok_type(x: object) -> bool:
        return (
            x is None
            or isinstance(x, str)
            or isinstance(x, list)
            or isinstance(x, tuple)
            or isinstance(x, np.ndarray)
        )

    ok = non_null.map(ok_type).all()
    assert bool(ok), f"Column {col} contains unsupported values. Example: {non_null.iloc[0]!r}"


def explode_list_col(df: pd.DataFrame, col: str) -> pd.Series:
    assert col in df.columns, f"Missing column: {col}"
    s = df[col]
    out: list[str] = []

    for v in s:
        if v is None:
            continue

        if isinstance(v, str):
            vv = v.strip()
            if vv:
                out.append(vv)
            continue

        if isinstance(v, (list, tuple, np.ndarray)):
            for x in v:
                if isinstance(x, str):
                    xx = x.strip()
                    if xx:
                        out.append(xx)
            continue

        assert False, f"Unexpected type in column {col}: {type(v)} value={v!r}"

    return pd.Series(out, dtype=str)


def split_early_late(df: pd.DataFrame, year_col: str = "year") -> tuple[pd.DataFrame, pd.DataFrame]:
    assert year_col in df.columns, f"Missing year column: {year_col}"
    years = sorted(pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int).unique().tolist())
    assert len(years) >= 2, f"Need at least two unique years to compute a trend for {year_col}"

    split_idx = len(years) // 2
    early_years = set(years[:split_idx])
    late_years = set(years[split_idx:])
    early = df[df[year_col].isin(early_years)].copy()
    late = df[df[year_col].isin(late_years)].copy()
    assert len(early) > 0 and len(late) > 0, "Early/late split produced an empty partition."
    return early, late


def value_counts_scalar(
    df: pd.DataFrame,
    col: str,
    normalizer=None,
    drop_labels: set[str] | None = None,
) -> pd.Series:
    assert col in df.columns, f"Missing column: {col}"
    s = df[col].dropna()
    if normalizer is not None:
        s = s.map(normalizer)
    s = s.map(lambda x: str(x).strip() if x is not None else "")
    s = s[s != ""]
    if drop_labels:
        s = s[~s.isin(drop_labels)]
    return s.value_counts()


def value_counts_list(
    df: pd.DataFrame,
    col: str,
    normalizer=None,
    drop_labels: set[str] | None = None,
) -> pd.Series:
    assert_listlike(df, col)
    s = explode_list_col(df, col)
    if normalizer is not None:
        s = s.map(normalizer)
    s = s.map(lambda x: str(x).strip() if x is not None else "")
    s = s[s != ""]
    if drop_labels:
        s = s[~s.isin(drop_labels)]
    return s.value_counts()


def relative_rank_status_for_labels(
    focal_labels: list[str],
    other_labels: list[str],
    labels: list[str],
) -> dict[str, str]:
    focal_rank = {label: idx for idx, label in enumerate(focal_labels)}
    other_rank = {label: idx for idx, label in enumerate(other_labels)}

    out: dict[str, str] = {}
    for label in labels:
        f_rank = focal_rank.get(label)
        o_rank = other_rank.get(label)
        assert f_rank is not None, f"Label {label!r} missing from focal rank set."

        if o_rank is None:
            out[label] = "new"
        elif f_rank == o_rank:
            out[label] = "same"
        elif f_rank < o_rank:
            out[label] = "up"
        else:
            out[label] = "down"
    return out


def add_yaxis_status_markers(ax, status_by_label: dict[str, str], gap_points: float = 4.0) -> None:
    fig = ax.figure
    for tick in ax.get_yticklabels():
        label = tick.get_text()
        status = status_by_label.get(label)
        if status is None:
            continue
        x, y = tick.get_position()
        ax.text(
            x,
            y,
            TREND_SYMBOLS[status],
            transform=offset_copy(tick.get_transform(), fig=fig, x=gap_points, y=0.0, units="points"),
            ha="left",
            va="center",
            color=TREND_COLORS[status],
            fontsize=9.5,
            fontweight="bold",
            clip_on=False,
        )
