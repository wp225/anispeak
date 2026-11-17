# AniSpeakArchive static demo

A small version of AniSpeakArchive that runs entirely in the browser, so it can be hosted on GitHub Pages with no server.

| Feature | Full system | Demo |
|---|---|---|
| Corpus | OpenAlex, filtered by a SciBERT classifier and an LLM | 476 papers from targeted OpenAlex searches |
| Keyword search | BM25 (LanceDB FTS) | BM25 (MiniSearch, in browser) |
| Semantic search | `bge-m3` on GPU | `bge-small-en-v1.5`: papers embedded at build time, queries embedded in the browser with transformers.js |
| Fusion / rerank | RRF + cross-encoder reranker | RRF, no reranker |
| Facets | LLM annotations | Keyword rules (taxa, modality, theme, methods) + OpenAlex topic |
| Ask AI | Live, GPU-hosted LLM | 4 pre-generated answers that cite records retrieved from the demo corpus |
| Accounts, bookmarks | Yes | No |

## Layout

```
demo/
  site/                    # what GitHub Pages serves
    index.html, app.js, style.css
    data/papers.json       # paper records + facets
    data/embeddings.bin    # float32 [n_papers x 384], same order as papers.json
    data/meta.json
    data/ask.json          # Ask AI examples with references
  build/
    fetch_openalex.py      # 1. pull the paper sample from the OpenAlex API
    build_site_data.py     # 2. normalise, tag, embed, write site/data/
    ask_answers.json       # pre-generated Ask AI answers (cite [n] from ask_context/)
    ask_context/           # records retrieved for each example question
```

## Run locally

```bash
cd demo/site
python -m http.server 8791
# open http://127.0.0.1:8791
```

Open it through a server, not `file://`, because the page fetches its data files.

## Rebuild the data

From the repo root, with the project venv active:

```bash
python demo/build/fetch_openalex.py          # writes demo/build/raw/ (git-ignored)
python demo/build/build_site_data.py
```

`build_site_data.py` writes the retrieval context for each Ask AI question to `ask_context/`, then merges `ask_answers.json`. If you change the corpus, retrieval may return different records. The build fails if an answer cites a record number that no longer exists, but an answer can still cite the wrong paper. So re-check each answer against its `ask_context/*.json` after any rebuild.

## Deploy

1. Push to GitHub.
2. In the repository settings, under **Pages → Build and deployment**, set **Source** to **GitHub Actions**.
3. `.github/workflows/deploy-demo.yml` publishes `demo/site/` on every push to `main` that touches it, or manually from the Actions tab.

The site will be at `https://<your-username>.github.io/<repo-name>/`.
