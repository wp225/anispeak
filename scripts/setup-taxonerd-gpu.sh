#!/usr/bin/env bash
# setup-taxonerd-gpu.sh
# TaxoNERD + spaCy GPU for Blackwell (CUDA 13.x) + NumPy 2.x

set -euo pipefail

ENV_NAME="taxonerd-gpu"
NER_MODEL_URL="https://github.com/nleguillarme/taxonerd/releases/download/v1.5.4/en_ner_eco_biobert-1.1.0.tar.gz"

echo "========================================"
echo " TaxoNERD GPU environment setup"
echo " env:   $ENV_NAME"
echo " CUDA:  13.x (cupy-cuda13x)"
echo " NumPy: 2.x (required by CuPy 14)"
echo "========================================"

echo ""
echo "[1/6] Create env (python=3.11) ..."
mamba create -n "$ENV_NAME" -y python=3.11 pip

echo ""
echo "[2/6] Install GPU core (NumPy 2.x + CuPy CUDA 13.x) ..."
mamba run -n "$ENV_NAME" python -m pip install -U pip setuptools wheel
mamba run -n "$ENV_NAME" python -m pip install -U \
  "numpy>=2.0,<2.6" \
  "cupy-cuda13x>=14,<15"

echo ""
echo "[3/6] Verify CuPy import + GPU ..."
mamba run -n "$ENV_NAME" python -c "import cupy as cp; print('CuPy:', cp.__version__); print('GPUs:', cp.cuda.runtime.getDeviceCount())"

echo ""
echo "[4/6] Install spaCy stack (NumPy 2.x compatible) ..."
mamba run -n "$ENV_NAME" python -m pip install -U \
  "spacy>=3.8.0,<3.9.0" \
  "thinc>=8.3.0,<8.4.0" \
  "spacy-transformers>=1.3.4,<1.4.0" \
  "torch>=2.1" \
  pandas tqdm requests pydantic

echo ""
echo "[5/6] Install TaxoNERD + deps (bypassing old pins) ..."
# TaxoNERD without its pinned deps
mamba run -n "$ENV_NAME" python -m pip install -U taxonerd --no-deps

# textract workaround (broken metadata)
mamba run -n "$ENV_NAME" python -m pip install "pip>=23,<24.1"
mamba run -n "$ENV_NAME" python -m pip install textract
mamba run -n "$ENV_NAME" python -m pip install -U pip

# scispacy without pins (runtime deps only)
mamba run -n "$ENV_NAME" python -m pip install "scispacy==0.5.4" --no-deps
mamba run -n "$ENV_NAME" python -m pip install -U conllu nmslib pysbd

# remaining pipeline deps
mamba run -n "$ENV_NAME" python -m pip install -U \
  "pygbif>=0.6.3" gnparser "scikit-learn>=1.5.0" "click>=8.0.0"

echo ""
echo "[6/6] Install NER model (no-deps) ..."
mamba run -n "$ENV_NAME" python -m pip install --no-deps "$NER_MODEL_URL"

echo ""
echo "========================================"
echo " Verification ..."
echo "========================================"
mamba run -n "$ENV_NAME" python -c "
import numpy as np, cupy as cp, spacy
from taxonerd import TaxoNERD
print(f'numpy {np.__version__}, cupy {cp.__version__}, spacy {spacy.__version__}')
print(f'GPU: {cp.cuda.runtime.getDeviceCount()} device(s)')
print(f'spacy.prefer_gpu(): {spacy.prefer_gpu()}')
t = TaxoNERD(prefer_gpu=True)
nlp = t.load(model='en_ner_eco_biobert')
doc = nlp('Ursus arctos was observed.')
print(f'Entities: {[(e.text, e.label_) for e in doc.ents]}')
print('\nAll OK! Run: mamba activate taxonerd-gpu')
"
