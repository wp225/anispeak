"""Build the static demo's data files from the OpenAlex sample.

Inputs:
  demo/build/raw/openalex.jsonl        (from fetch_openalex.py)
  demo/build/ask_answers.json          (pre-generated Ask AI answers; optional)

Outputs (served by GitHub Pages):
  demo/site/data/papers.json           paper records + facets
  demo/site/data/embeddings.bin        float32 [n_papers x dim], L2-normalised, same order as papers.json
  demo/site/data/meta.json             model name, dims, build info
  demo/site/data/ask.json              example questions with answers + references

Also writes demo/build/ask_context/<slug>.json: the records retrieved for each
example question, numbered [1..k], which the pre-generated answers cite.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
import sys
from collections import Counter
from datetime import date

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from corpus.schema import _abstract_from_inverted_index, _doi_to_str, _extract_authors_affils_insts  # noqa: E402

BUILD = ROOT / "demo" / "build"
SITE_DATA = ROOT / "demo" / "site" / "data"
RAW = BUILD / "raw" / "openalex.jsonl"
ANSWERS = BUILD / "ask_answers.json"
CONTEXT_DIR = BUILD / "ask_context"

MODEL_NAME = "BAAI/bge-small-en-v1.5"          # Python side
BROWSER_MODEL = "Xenova/bge-small-en-v1.5"     # ONNX export of the same model for transformers.js
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_ABSTRACT_CHARS = 2400
ASK_TOP_K = 10

# Same starter questions as the production Ask AI page.
ASK_QUESTIONS = [
    "What vocalizations do humpback whales use to communicate?",
    "How do birds learn songs from their parents?",
    "What is the role of infrasound in elephant communication?",
    "Which species show evidence of syntax in their calls?",
]

# --- Rule-based facet tagging --------------------------------------------------
# The production system annotates papers with an LLM (species, modality, stages, ...).
# The demo approximates a few of those facets with keyword rules over title + abstract.

TAXA = {
    "Birds": r"\bbirds?\b|songbirds?|avian|passerine|finch(es)?|sparrows?|parrots?|budgerigars?|chickadees?|babblers?|starlings?|warblers?|corvids?|crows?|ravens?|tits?\b|great tit|nightingales?|canar(y|ies)|wrens?|hummingbirds?|penguins?",
    "Cetaceans": r"whales?|dolphins?|cetaceans?|porpoises?|vaquita|orcas?|odontocetes?|mysticetes?|beluga|narwhal",
    "Primates": r"primates?|monkeys?|chimpanzees?|bonobos?|macaques?|gibbons?|marmosets?|lemurs?|baboons?|gorillas?|orangutans?|tamarins?|capuchins?|vervet",
    "Bats": r"\bbats?\b|chiroptera|microbats?|pipistrell",
    "Elephants": r"elephants?|loxodonta|elephas",
    "Amphibians": r"\bfrogs?\b|treefrogs?|\btoads?\b|anurans?|amphibians?",
    "Insects": r"insects?|crickets?|\bbees?\b|honeybees?|apis mellifera|cicadas?|katydids?|grasshoppers?|orthoptera|drosophila|\bants?\b|moths?|butterfl(y|ies)|planthoppers?|leafhoppers?|treehoppers?",
    "Fish": r"\bfish(es)?\b|teleost|cichlids?|toadfish|midshipm[ae]n",
    "Rodents": r"rodents?|\bmice\b|\bmouse\b|\brats?\b|gerbils?|squirrels?|prairie dogs?|mole-rats?",
    "Pinnipeds": r"\bseals?\b|sea lions?|walrus(es)?|pinnipeds?",
    "Other mammals": r"meerkats?|mongoose|\bdogs?\b|wolves|\bwolf\b|hyenas?|\bdeer\b|\bgoats?\b|\bhorses?\b|\bpigs?\b|\bcats?\b|giraffes?|hyrax(es)?|marsupials?|kangaroos?",
}
MODALITY = {
    "Acoustic": r"acoustic|vocal|vocali[sz]ations?|\bcalls?\b|calling|\bsongs?\b|sounds?|whistles?|ultrason|infrasound|infrasonic|echolocat|audit",
    "Visual": r"visual|colou?r(ation)?|gestur|\bdisplays?\b|facial expression",
    "Chemical": r"chemical|pheromon|olfact|scent|odou?r",
    "Vibrational": r"vibration|seismic|substrate-borne|substrate borne",
    "Tactile": r"tactile|\btouch",
}
METHODS = {
    "Machine learning": r"machine learning|deep learning|neural network|convolutional|\bcnns?\b|classifier|random forest|transformer model|self-supervised|unsupervised learning|clustering algorithm",
    "Playback experiments": r"playbacks?",
    "Passive acoustic monitoring": r"passive acoustic monitoring|\bpam\b|acoustic monitoring|hydrophones?|autonomous recorders?|acoustic recorders?",
    "Neural / physiological": r"neurons?|neural circuit|brain|cortex|ganglia|neurophysiolog|hormone|gene expression|\bfoxp2\b",
    "Field observation": r"field (study|observations?|recordings?)|in the wild|wild population|free-ranging|free-living",
    "Review / synthesis": r"\breview\b|we review|meta-analysis|synthesis|overview",
}
THEMES = {
    "Vocal learning": r"vocal learning|song learning|learn(ed|ing)? (songs?|calls?|vocal)|tutor|imitat|vocal production learning|mimicry",
    "Syntax & combinatorics": r"syntax|syntactic|combinator|compositional|call sequences?|sequence structure|grammar|phonolog",
    "Alarm & referential calls": r"alarm calls?|referential|predator-specific|anti-predator",
    "Individual recognition": r"individual (recognition|identity|signature|distinctiveness|identification)|signature whistles?|vocal identity|individuality",
    "Mate attraction & sexual selection": r"mate (choice|attraction)|sexual selection|courtship|female preference|male song",
    "Conservation & monitoring": r"conservation|endangered|monitoring|population (density|trends?)|anthropogenic noise|noise pollution|shipping noise",
    "Navigation & foraging": r"echolocat|foraging|navigation|waggle dance|food source",
}

_COMPILED = {name: {k: re.compile(v, re.I) for k, v in rules.items()} for name, rules in
             {"taxa": TAXA, "modality": MODALITY, "methods": METHODS, "themes": THEMES}.items()}


def tag(text: str) -> dict[str, list[str]]:
    return {facet: [label for label, rx in rules.items() if rx.search(text)]
            for facet, rules in _COMPILED.items()}


# --- Normalisation -------------------------------------------------------------

def clean_text(s: str | None) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", "", s)          # strip stray JATS/HTML tags
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(abstract|summary)[:.\s]+", "", s, flags=re.I)
    return s


def normalise(w: dict) -> dict | None:
    wid = (w.get("id") or "").rsplit("/", 1)[-1]
    title = clean_text(w.get("title") or w.get("display_name"))
    abstract = clean_text(_abstract_from_inverted_index(w.get("abstract_inverted_index")))
    if not wid or not title or len(abstract) < 200:
        return None
    if len(abstract) > MAX_ABSTRACT_CHARS:
        abstract = abstract[:MAX_ABSTRACT_CHARS].rsplit(" ", 1)[0] + " …"

    authors, _affils, insts = _extract_authors_affils_insts(w.get("authorships"))
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    oa = w.get("open_access") or {}
    topic = w.get("primary_topic") or {}
    year = w.get("publication_year")

    tags = tag(f"{title}. {abstract}")
    return {
        "id": wid,
        "title": title,
        "abstract": abstract,
        "authors": authors[:25],
        "n_authors": len(authors),
        "institutions": insts[:8],
        "year": year if isinstance(year, int) else None,
        "venue": (src.get("display_name") or "").strip() or None,
        "doi": _doi_to_str(w.get("doi")),
        "url": loc.get("landing_page_url"),
        "oa_url": oa.get("oa_url"),
        "is_oa": bool(oa.get("is_oa")),
        "cited_by": int(w.get("cited_by_count") or 0),
        "topic": (topic.get("display_name") or "").strip() or None,
        "field": ((topic.get("subfield") or {}).get("display_name") or "").strip() or None,
        **tags,
    }


# --- Retrieval (mirrors the app: BM25 + dense, fused with RRF) -----------------

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.toks = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(t.values()) for t in self.toks]
        self.avg = sum(self.lens) / max(1, len(self.lens))
        df = Counter(term for t in self.toks for term in t)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def scores(self, q: str) -> np.ndarray:
        out = np.zeros(len(self.toks))
        for term in set(tokenize(q)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.toks):
                f = tf.get(term)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
        return out


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    score: dict[int, float] = {}
    for ranking in rankings:
        for r, idx in enumerate(ranking):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + r + 1)
    return sorted(score, key=lambda i: -score[i])


def slug(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", q.lower()).strip("-")[:60]


def main() -> None:
    from sentence_transformers import SentenceTransformer

    raw = [json.loads(line) for line in RAW.read_text(encoding="utf-8").split("\n") if line.strip()]
    papers: list[dict] = []
    seen_titles: set[str] = set()
    for w in raw:
        p = normalise(w)
        if p is None:
            continue
        key = re.sub(r"\W+", "", p["title"].lower())
        if key in seen_titles:
            continue
        seen_titles.add(key)
        papers.append(p)
    papers.sort(key=lambda p: (-(p["year"] or 0), p["title"]))
    print(f"papers: {len(papers)} (from {len(raw)} raw)")

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    doc_texts = [f"{p['title']}. {p['abstract']}" for p in papers]
    emb = model.encode(doc_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)

    SITE_DATA.mkdir(parents=True, exist_ok=True)
    emb.tofile(SITE_DATA / "embeddings.bin")
    (SITE_DATA / "papers.json").write_text(json.dumps(papers, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (SITE_DATA / "meta.json").write_text(json.dumps({
        "n_papers": len(papers),
        "dim": int(emb.shape[1]),
        "browser_model": BROWSER_MODEL,
        "query_prefix": QUERY_PREFIX,
        "built": date.today().isoformat(),
        "source": "OpenAlex API (CC0)",
    }, indent=2), encoding="utf-8")

    # Retrieval context for the pre-generated Ask AI answers.
    bm25 = BM25([f"{p['title']} {p['abstract']}" for p in papers])
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    contexts: dict[str, list[dict]] = {}
    for q in ASK_QUESTIONS:
        qv = model.encode([QUERY_PREFIX + q], normalize_embeddings=True)[0]
        dense = list(np.argsort(-(emb @ qv))[:200])
        lex_s = bm25.scores(q)
        lex = [i for i in np.argsort(-lex_s)[:200] if lex_s[i] > 0]
        top = rrf([lex, dense])[:ASK_TOP_K]
        ctx = [{"citation_id": n + 1, "id": papers[i]["id"], "title": papers[i]["title"], "year": papers[i]["year"],
                "abstract": papers[i]["abstract"]} for n, i in enumerate(top)]
        contexts[q] = ctx
        (CONTEXT_DIR / f"{slug(q)}.json").write_text(json.dumps({"question": q, "records": ctx}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote retrieval context for {len(ASK_QUESTIONS)} questions -> {CONTEXT_DIR}")

    # Merge pre-generated answers, if present.
    answers = json.loads(ANSWERS.read_text(encoding="utf-8")) if ANSWERS.exists() else {}
    by_id = {p["id"]: p for p in papers}
    ask_out = []
    for q in ASK_QUESTIONS:
        a = answers.get(q)
        if not a:
            print(f"  (no answer yet for: {q})")
            continue
        ctx = contexts[q]
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", a["answer_markdown"])})
        bad = [n for n in cited if n < 1 or n > len(ctx)]
        if bad:
            raise SystemExit(f"answer for {q!r} cites records outside the context: {bad}")
        refs = []
        for n in cited:
            p = by_id[ctx[n - 1]["id"]]
            refs.append({"citation_id": n, "id": p["id"], "title": p["title"], "year": p["year"],
                         "authors": p["authors"][:3], "n_authors": p["n_authors"], "venue": p["venue"]})
        ask_out.append({"question": q, "answer_markdown": a["answer_markdown"], "confidence": a.get("confidence", "medium"),
                        "limitations": a.get("limitations"), "references": refs,
                        "n_retrieved": len(ctx)})
    (SITE_DATA / "ask.json").write_text(json.dumps(ask_out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"ask examples: {len(ask_out)}")

    for facet in ("taxa", "modality", "methods", "themes"):
        c = Counter(v for p in papers for v in p[facet])
        untagged = sum(1 for p in papers if not p[facet])
        print(f"{facet}: {dict(c.most_common())}  untagged={untagged}")


if __name__ == "__main__":
    main()
