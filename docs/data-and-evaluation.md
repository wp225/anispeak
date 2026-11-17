# Data and evaluation

## Data provenance

The project is built on OpenAlex scholarly metadata and full-text-adjacent paper records. The data flow is implemented in [pipeline.py](../pipeline.py), [pipeline_scripts/download_snapshot.py](../pipeline_scripts/download_snapshot.py), and the corpus utilities in [corpus/](../corpus).

The repository expects local runtime data directories such as data/openalex, data/processed, and cache/lancedb. Those directories are part of the runtime layout described in [README.md](../README.md), but the current workspace snapshot did not include a populated data directory.

## Preprocessing and labeling

The preprocessing pipeline includes:

- normalization of OpenAlex records into a web-corpus representation;
- text construction for model inputs from title, venue, keywords, and abstract fields;
- a training dataset builder that defines positive and negative sampling rules.

The training label logic is defined in [pipeline_scripts/build_dataset.py](../pipeline_scripts/build_dataset.py) and [configs/bert_classifier.yaml](../configs/bert_classifier.yaml):

- positive labels are drawn from high-confidence records;
- negatives are drawn from lower-confidence or rejected records;
- the dataset uses a balanced positive/negative ratio and deduplication by DOI and normalized title.

## Splits and leakage controls

The training configuration defines train/validation/test splits as percentages in [configs/bert_classifier.yaml](../configs/bert_classifier.yaml). The dataset builder also deduplicates by DOI and normalized title to reduce repeated-paper leakage across splits.

The repository does not include a separate, externally versioned benchmark release or a documented holdout strategy for retrieval quality beyond the training split logic.

## Metrics and evaluation

The training code defines the following metrics in [pipeline_scripts/train_bert.py](../pipeline_scripts/train_bert.py):

- accuracy;
- precision;
- recall;
- F1;
- AUROC;
- macro-F1 as the primary metric for ablation and comparison.

However, no training logs, checkpoint summaries, or evaluation JSON/CSV artifacts were found in the repository. The available evidence therefore supports the presence of an evaluation framework, not a published benchmark result.

## Limitations

The main data and evaluation limitations are:

- the labels are pipeline-derived and not a manually curated gold benchmark;
- the repository does not include a public retrieval evaluation set with relevance judgments;
- the current tests are lightweight and cover schema transformation and API smoke behavior rather than end-to-end retrieval quality;
- runtime performance and retrieval quality will depend heavily on the local data availability and the installed model weights.
