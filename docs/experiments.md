# Experiments

## What is implemented

The repository contains a concrete experiment pipeline rather than a single one-off notebook. The main experiment surfaces are:

- a SciBERT fine-tuning workflow in [pipeline_scripts/train_bert.py](../pipeline_scripts/train_bert.py);
- a dataset builder that creates train/validation/test examples in [pipeline_scripts/build_dataset.py](../pipeline_scripts/build_dataset.py);
- a retrieval stack that can be configured for different embedding and reranker models in [configs/default.yaml](../configs/default.yaml).

## Experiment configuration

| Area | Configuration | Evidence |
| --- | --- | --- |
| Classifier backbone | allenai/scibert_scivocab_uncased | [configs/bert_classifier.yaml](../configs/bert_classifier.yaml) |
| Training loss | asymmetric focal loss with positive/negative weighting | [pipeline_scripts/train_bert.py](../pipeline_scripts/train_bert.py) |
| Inference threshold | 0.4 | [configs/bert_classifier.yaml](../configs/bert_classifier.yaml) |
| Embedding model | BAAI/bge-m3 | [configs/default.yaml](../configs/default.yaml) |
| Reranker | BAAI/bge-reranker-base | [configs/default.yaml](../configs/default.yaml) |
| LLM annotation | Qwen-based annotation path | [pipeline.py](../pipeline.py), [pipeline_scripts/annotate_qwen.py](../pipeline_scripts/annotate_qwen.py) |

## Hypotheses reflected in the code

The implementation suggests several practical hypotheses:

1. A domain-specific classifier can improve relevance selection over a raw keyword filter.
2. A hybrid retrieval stack can outperform a single retrieval signal by combining lexical and semantic matches.
3. Retrieval-grounded LLM responses can provide a more useful interface for paper exploration than a raw search box alone.

## Evaluation evidence currently present

The code defines the following metrics and logging targets:

- accuracy;
- precision;
- recall;
- F1;
- AUROC;
- macro-F1.

Those metrics are wired into the training loop in [pipeline_scripts/train_bert.py](../pipeline_scripts/train_bert.py), but the repository does not contain a saved run directory or evaluation artifact with concrete values. For that reason, the portfolio documentation should present this as an implemented experiment framework rather than a published result table.

## Failure analysis and next steps

The main failure modes are visible from the implementation rather than from logged results:

- model performance depends on the quality of the silver labels;
- the pipeline assumes GPU access for large models and may be expensive to reproduce on CPU-only hardware;
- retrieval quality can be affected by data freshness and the quality of the underlying metadata.

The next high-value follow-up is to add a documented benchmark set and save evaluation artifacts for each model variant so that the repository can support a credible performance comparison.
