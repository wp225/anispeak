# AniSpeakArchive

AniSpeakArchive is a retrieval-oriented ML system for scholarly papers in animal communication, bioacoustics, and related computational biology domains. The repository combines a staged data pipeline, fine-tuned relevance filtering, LLM-assisted metadata enrichment, and a hybrid search stack exposed through a Flask application.


Yes, UI is AI made.
## What is implemented

The repository currently supports:

- a four-stage pipeline for downloading OpenAlex data, filtering for relevance, annotating papers, and rebuilding a search index;
- a SciBERT-based relevance classifier with training and inference code;
- dense embeddings and reranking for hybrid retrieval;
- a Flask web app with search APIs, faceted browsing, citations, bookmarks, auth, and an Ask-AI experience.

## Architecture at a glance

The system follows a simple flow:

1. ingest OpenAlex records;
2. select likely-relevant papers with a BERT classifier;
3. enrich the retained papers with metadata and domain annotations;
4. index the normalized documents and serve them through the app.

A fuller description is available in [docs/architecture.md](docs/architecture.md).

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the web app:

```bash
python app.py
```

Run the pipeline:

```bash
python pipeline.py
```

A single stage can be run for debugging:

```bash
python pipeline.py --stage bert
python pipeline.py --stage qwen
python pipeline.py --stage index
```

## Project structure

Key parts of the repository:

- [pipeline.py](pipeline.py) — orchestrates the four-stage pipeline.
- [pipeline_scripts/](pipeline_scripts) — download, filtering, annotation, training, and dataset-building scripts.
- [anispeakarchive/](anispeakarchive) — Flask app, API routes, index logic, and service layer.
- [configs/](configs) — runtime configuration for the app and the classifier.
- [tests/](tests) — lightweight regression tests for schema and API behavior.

## Evidence-based portfolio notes

This repository is strongest as a systems and ML engineering project rather than as a benchmark report. The code clearly implements the pipeline and the search stack, and the training script defines evaluation metrics such as accuracy, precision, recall, F1, AUROC, and macro-F1. However, the repository does not currently include published result artifacts, training logs, or a public evaluation dataset that would support stronger claims about retrieval quality.

## Documentation

- [docs/project-overview.md](docs/project-overview.md) — portfolio-style case study.
- [docs/architecture.md](docs/architecture.md) — system architecture and data flow.
- [docs/data-and-evaluation.md](docs/data-and-evaluation.md) — data provenance, preprocessing, and evaluation status.
- [docs/experiments.md](docs/experiments.md) — implemented experiments and current evidence gaps.
- [docs/reproducibility.md](docs/reproducibility.md) — setup and reproduction guidance.

## Configuration and secrets

The runtime configuration is driven by [configs/default.yaml](configs/default.yaml), [pipeline_config.yaml](pipeline_config.yaml), and [configs/bert_classifier.yaml](configs/bert_classifier.yaml). Sensitive values such as SMTP credentials should be provided via environment variables rather than committed into YAML files.

