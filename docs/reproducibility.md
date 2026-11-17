# Reproducibility

## Environment

The repository is a Python application with GPU-oriented ML dependencies. The dependency list is defined in [requirements.txt](../requirements.txt), and the deployment stack is described in [Dockerfile](../Dockerfile), [Dockerfile.pipeline](../Dockerfile.pipeline), and [docker-compose.yml](../docker-compose.yml).

The code is designed to run from the repository root with Python 3.11+ and a working CUDA-capable environment for the embedding and reranking components.

## Hardware expectations

The full pipeline is easiest to reproduce on a machine with:

- a modern NVIDIA GPU for the embedding/reranking/LLM stages;
- enough disk space for the OpenAlex snapshot, intermediate JSONL files, and the LanceDB index;
- access to the model weights referenced in [pipeline_config.yaml](../pipeline_config.yaml) and [configs/default.yaml](../configs/default.yaml).

## Data access constraints

The repository expects a local data layout that includes:

- OpenAlex snapshot files under data/openalex;
- processed outputs under data/processed;
- model assets under data/models or the lab-specific paths configured in the YAML files.

A full end-to-end run is therefore constrained by data availability and model access. The current workspace snapshot did not include the populated data directories, so the repository is best treated as a reproducible codebase rather than a turnkey demo without local assets.

## Reproduction commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the web application:

```bash
python app.py
```

Run the full pipeline:

```bash
python pipeline.py
```

Run a single stage for debugging:

```bash
python pipeline.py --stage bert
python pipeline.py --stage qwen
python pipeline.py --stage index
```

## Expected outputs

A successful run should produce:

- downloaded OpenAlex snapshot files;
- filtered and annotated JSONL outputs under data/processed;
- a searchable LanceDB index under the configured Lancedb path;
- a running Flask app on the configured host/port from [configs/default.yaml](../configs/default.yaml).

## Known limitations

- The repository does not ship a precomputed benchmark or a static evaluation artifact.
- Some configuration values point to lab-specific model paths, so the setup may need adjustment on a new machine.
- The app and the pipeline both rely on environment variables and local filesystem paths for runtime data.
- The current tests are limited and should be treated as smoke checks rather than strong validation of retrieval quality.
