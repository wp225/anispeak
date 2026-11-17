# Project overview

## 1. Problem and motivation

AniSpeakArchive is a search and retrieval system for scholarly literature in animal communication, bioacoustics, computational ethology, and related computational biology topics. The repository addresses a common research problem: when a scientist wants to find relevant papers, simple keyword search is often too shallow because the relevant signal may be distributed across titles, abstracts, topic metadata, and domain-specific terminology.

The project combines document ingestion, relevance filtering, metadata enrichment, and hybrid retrieval so that users can search a curated corpus of papers and ask follow-up questions about the indexed literature.

## 2. My contribution

The repository implements a full stack around that idea:

- a staged data pipeline that downloads OpenAlex snapshots and prepares a cleaned corpus;
- a SciBERT-based relevance filter for paper selection;
- an LLM-assisted annotation stage for metadata enrichment;
- a hybrid search stack using BM25, dense embeddings, and reranking;
- a Flask web application with search APIs, faceted browsing, auth, and an Ask-AI experience.

This is a systems-oriented ML project rather than a single-model demo: the work spans data preparation, model integration, indexing, API services, and deployment configuration.

## 3. System design

The system is built around four major stages:

1. Ingest and normalize OpenAlex records.
2. Filter for likely relevance with a fine-tuned text classifier.
3. Enrich the retained papers with structured metadata using an LLM-based annotation pass.
4. Build a hybrid search index and expose it via a web application and APIs.

The retrieval stack uses:

- BM25 lexical search for exact terminology;
- dense embeddings for semantic similarity;
- a reranker to refine candidate results;
- a retrieval-aware Ask-AI service that grounds responses in indexed records.

## 4. Data

The repository uses OpenAlex as the source corpus. The pipeline expects a local snapshot or a remote fetch step and produces intermediate and final JSONL files under the data directory.

Key evidence from the code:

- OpenAlex snapshot download and filtering are implemented in [pipeline.py](../pipeline.py) and [pipeline_scripts/download_snapshot.py](../pipeline_scripts/download_snapshot.py).
- The BERT training dataset is assembled from labeled records in [pipeline_scripts/build_dataset.py](../pipeline_scripts/build_dataset.py).
- The training config defines positive and negative label rules, train/validation/test ratios, and a minimum text length threshold in [configs/bert_classifier.yaml](../configs/bert_classifier.yaml).

Important caveats:

- The repository does not include a public benchmark set, hand-labeled evaluation file, or experiment log with reported metrics.
- The dataset is largely silver-labeled through the pipeline stages rather than manually curated from a public benchmark.

## 5. Models and experiments

The repository contains evidence for several model components:

| Component | Implementation | Evidence |
| --- | --- | --- |
| Relevance classifier | Fine-tuned SciBERT with asymmetric focal loss | [pipeline_scripts/train_bert.py](../pipeline_scripts/train_bert.py), [configs/bert_classifier.yaml](../configs/bert_classifier.yaml) |
| Embedding model | BAAI/bge-m3 via FlagEmbedding | [configs/default.yaml](../configs/default.yaml) |
| Reranker | BAAI/bge-reranker-base | [configs/default.yaml](../configs/default.yaml) |
| Annotation model | Qwen-based annotation flow via vLLM | [pipeline.py](../pipeline.py), [pipeline_scripts/annotate_qwen.py](../pipeline_scripts/annotate_qwen.py) |
| Retrieval strategy | Hybrid BM25 + dense + RRF fusion | [anispeakarchive/index/hybrid.py](../anispeakarchive/index/hybrid.py) |

The training script defines evaluation metrics such as accuracy, precision, recall, F1, AUROC, and a macro-F1 target, but no trained model results or experiment output files are checked into the repository.

## 6. Results

The repository does not contain a completed evaluation report with published metrics. What is present is the evaluation framework itself:

| Metric family | Where it is defined | Status |
| --- | --- | --- |
| Accuracy, precision, recall, F1, AUROC | [pipeline_scripts/train_bert.py](../pipeline_scripts/train_bert.py) | Implemented in training code |
| Macro-F1 | [pipeline_scripts/train_bert.py](../pipeline_scripts/train_bert.py) | Used as the primary metric in the training script |
| End-to-end retrieval quality | No result artifact in repo | Not evidenced in repository |

This means the documentation should emphasize implementation and system design over unverified benchmark claims.

## 7. Engineering and deployment

The repository includes practical engineering infrastructure:

- Flask web app and REST APIs in [app.py](../app.py) and [anispeakarchive](../anispeakarchive);
- Gunicorn entrypoint in [wsgi.py](../wsgi.py);
- Dockerfiles and docker-compose configuration in [Dockerfile](../Dockerfile), [Dockerfile.pipeline](../Dockerfile.pipeline), and [docker-compose.yml](../docker-compose.yml);
- LanceDB for the retrieval index and SQLite for auth and conversation storage.

The codebase also includes environment-driven configuration for SMTP, auth policy, species services, and optional LLM backends.

## 8. Failure cases and lessons

The repository makes several limitations clear through its structure:

- The pipeline depends on external model weights and GPU availability.
- The training setup uses silver labels and heuristics rather than fully curated ground truth.
- The current tests are lightweight and do not provide end-to-end retrieval evaluation.
- Some runtime paths assume lab-specific model locations, such as the Qwen model path in [pipeline_config.yaml](../pipeline_config.yaml).

These constraints are important for a portfolio project because they show maturity: the work is implemented, but the reliability story is still tied to scaling and evaluation discipline.

## 9. Reproduce

The core reproduction path is documented in [docs/reproducibility.md](reproducibility.md). In brief, the repository expects a Python environment, the application dependencies from [requirements.txt](../requirements.txt), and the relevant data and model assets at runtime.
