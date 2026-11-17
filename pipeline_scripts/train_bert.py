#!/usr/bin/env python3
"""
train_bert_classifier.py  v3
=============================
Config-driven BERT/SciBERT fine-tuner for paper relevance classification.
Reads all hyperparameters from bert_classifier_config.yaml.
Imports build_text() from build_bert_dataset.py (single source of truth).

Usage:
  python train_bert_classifier.py --config bert_classifier_config.yaml
  python train_bert_classifier.py --config bert_classifier_config.yaml --resume

  # Ablation: change model.name + paths.run_name in config, then re-run
  # Dataset does NOT need to be rebuilt for ablation — only model changes.
"""

import argparse, json, os, shutil, sys, warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping, LearningRateMonitor, ModelCheckpoint,
)
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy, BinaryAUROC, BinaryF1Score,
    BinaryPrecision, BinaryRecall, BinaryConfusionMatrix,
)
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

try:
    import yaml
except ImportError:
    print("pip install pyyaml"); raise

warnings.filterwarnings("ignore", category=UserWarning)

# ── Import build_text (single source of truth) ────────────────────────────────
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_dataset import build_text, reconstruct_abstract, get_oaid
    print("[train] Imported build_text from build_dataset.py ✓")
except ImportError as e:
    print(f"[train] WARNING: Could not import from build_dataset.py ({e})")
    print("[train] Using inline fallback — keep in sync with build_dataset.py!")

    def reconstruct_abstract(inv) -> str:
        if not inv or not isinstance(inv, dict): return ""
        try:
            pairs = [(pos, w) for w, positions in inv.items() for pos in positions]
            pairs.sort()
            return " ".join(w for _, w in pairs)
        except Exception: return ""

    def get_oaid(paper):
        import re
        m = re.search(r"W\d+", str(paper.get("id") or paper.get("oaid") or ""))
        return m.group() if m else None

    def build_text(paper: dict, cfg: dict) -> str:
        f = cfg.get("input_fields", {})
        parts = []
        if f.get("use_title", True):
            t = (paper.get("title") or paper.get("display_name") or "").strip()
            if t: parts.append(f"[TITLE] {t}")
        if f.get("use_venue", True):
            v = (paper.get("venue") or "").strip()
            if v: parts.append(f"[VENUE] {v}")
        if f.get("use_primary_topic_field", True):
            pt = paper.get("primary_topic")
            if isinstance(pt, dict):
                fn = (pt.get("field") or pt.get("subfield") or "").strip()
                if fn: parts.append(f"[FIELD] {fn}")
        if f.get("use_topics", False):
            ts = [str(t).strip() for t in (paper.get("topics") or [])[:f.get("max_topics",3)] if str(t).strip()]
            if ts: parts.append(f"[TOPICS] {'; '.join(ts)}")
        if f.get("use_keywords", True):
            ks = [(k.get("display_name") if isinstance(k, dict) else str(k) or "").strip()
                  for k in (paper.get("keywords") or [])[:f.get("max_keywords",8)]]
            ks = [s for s in ks if s]
            if ks: parts.append(f"[KEYWORDS] {'; '.join(ks)}")
        if f.get("use_concepts", False):
            cs = [str(c).strip() for c in (paper.get("concepts") or [])[:f.get("max_concepts",3)] if str(c).strip()]
            if cs: parts.append(f"[CONCEPTS] {'; '.join(cs)}")
        if f.get("use_mesh_terms", False):
            ms = [str(m).strip() for m in (paper.get("mesh_terms") or [])[:f.get("max_mesh",5)] if str(m).strip()]
            if ms: parts.append(f"[MESH] {'; '.join(ms)}")
        if f.get("use_abstract", True):
            ab = (paper.get("abstract") or "").strip() or reconstruct_abstract(paper.get("abstract_inverted_index"))
            if ab: parts.append(f"[ABSTRACT] {ab}")
        return " ".join(parts)
# ─────────────────────────────────────────────────────────────────────────────


class PaperDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int):
        self.records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: self.records.append(json.loads(line))
                except Exception: continue
        self.tokenizer = tokenizer
        self.max_len   = max_len
        print(f"  Loaded {len(self.records):,} from {os.path.basename(path)}")

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        text  = rec.get("text") or ""
        label = int(rec.get("label", 0))
        enc = self.tokenizer(
            text, max_length=self.max_len, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get(
                "token_type_ids", torch.zeros(self.max_len, dtype=torch.long)
            ).squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
        }

    @staticmethod
    def collate(batch): return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss.
    - gamma_neg > 0: suppresses easy negatives → more signal from hard cases
    - gamma_pos = 0: all positives contribute fully → recall-biased
    - pos_weight > 1: false negatives penalised more → recall bias
    - label_smoothing: combats overconfidence on LLM-generated silver labels
    """
    def __init__(self, gamma_pos=0.0, gamma_neg=2.0, pos_weight=1.2, label_smoothing=0.05):
        super().__init__()
        self.gp = gamma_pos; self.gn = gamma_neg
        self.pw = pos_weight; self.ls = label_smoothing

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        tgt_s = targets * (1.0 - self.ls) + 0.5 * self.ls
        bce = F.binary_cross_entropy_with_logits(
            logits, tgt_s,
            pos_weight=torch.tensor(self.pw, device=logits.device),
            reduction="none",
        )
        p_t   = probs * targets + (1.0 - probs) * (1.0 - targets)
        gamma = self.gp * targets + self.gn * (1.0 - targets)
        return ((1.0 - p_t) ** gamma * bce).mean()


class PaperClassifier(pl.LightningModule):
    def __init__(self, cfg: dict, total_steps: int):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg; self._total_steps = total_steps

        m_cfg = cfg["model"]; l_cfg = cfg["loss"]; i_cfg = cfg["inference"]
        thresh = float(i_cfg.get("threshold", 0.4))

        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            m_cfg["name"], num_labels=1,
            hidden_dropout_prob=float(m_cfg.get("dropout", 0.1)),
            attention_probs_dropout_prob=float(m_cfg.get("dropout", 0.1)),
            ignore_mismatched_sizes=True,
        )
        self.encoder.gradient_checkpointing_enable()

        self.criterion = AsymmetricFocalLoss(
            gamma_pos=float(l_cfg.get("focal_gamma_pos", 0.0)),
            gamma_neg=float(l_cfg.get("focal_gamma_neg", 2.0)),
            pos_weight=float(l_cfg.get("pos_weight", 1.2)),
            label_smoothing=float(l_cfg.get("label_smoothing", 0.05)),
        )

        def _metrics():
            return MetricCollection({
                "acc":       BinaryAccuracy(threshold=thresh),
                "precision": BinaryPrecision(threshold=thresh),
                "recall":    BinaryRecall(threshold=thresh),
                "f1":        BinaryF1Score(threshold=thresh),
                "auroc":     BinaryAUROC(),
            })
        self.train_metrics = _metrics(); self.val_metrics = _metrics(); self.test_metrics = _metrics()
        self.val_cm  = BinaryConfusionMatrix(threshold=thresh)
        self.test_cm = BinaryConfusionMatrix(threshold=thresh)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids).logits.squeeze(-1)

    def _step(self, batch):
        logits = self(batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"))
        labels = batch["labels"]
        return self.criterion(logits, labels), torch.sigmoid(logits), labels

    def training_step(self, batch, _):
        loss, probs, labels = self._step(batch)
        self.train_metrics.update(probs, labels.int())
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        self.log_dict({f"train/{k}": v for k, v in self.train_metrics.compute().items()}, sync_dist=True)
        self.train_metrics.reset()

    def validation_step(self, batch, _):
        loss, probs, labels = self._step(batch)
        self.val_metrics.update(probs, labels.int()); self.val_cm.update(probs, labels.int())
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        m = self.val_metrics.compute()
        self.log_dict({f"val/{k}": v for k, v in m.items()}, prog_bar=True, sync_dist=True)
        cm = self.val_cm.compute()
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        prec_neg = tn/(tn+fn+1e-9); rec_neg = tn/(tn+fp+1e-9)
        f1_neg   = 2*prec_neg*rec_neg/(prec_neg+rec_neg+1e-9)
        macro_f1 = (m["f1"].item() + f1_neg.item()) / 2.0
        self.log("val/macro_f1", macro_f1, prog_bar=True, sync_dist=True)
        self.log_dict({"val/tp": tp.float(),"val/fp": fp.float(),"val/tn": tn.float(),"val/fn": fn.float()}, sync_dist=True)
        self.val_metrics.reset(); self.val_cm.reset()

    def test_step(self, batch, _):
        loss, probs, labels = self._step(batch)
        self.test_metrics.update(probs, labels.int()); self.test_cm.update(probs, labels.int())
        self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self):
        m = self.test_metrics.compute()
        cm = self.test_cm.compute()
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        prec_neg = tn/(tn+fn+1e-9); rec_neg = tn/(tn+fp+1e-9)
        f1_neg   = 2*prec_neg*rec_neg/(prec_neg+rec_neg+1e-9)
        macro_f1 = (m["f1"].item() + f1_neg.item()) / 2.0
        self.log("test/macro_f1", macro_f1, sync_dist=True)
        self.log_dict({f"test/{k}": v for k, v in m.items()}, sync_dist=True)
        print(f"\n{'='*60}\nTEST RESULTS  [{self.cfg['model']['name']}]\n{'='*60}")
        print(f"  Macro-F1  : {macro_f1:.4f}   <- primary metric for ablation")
        print(f"  F1        : {m['f1'].item():.4f}")
        print(f"  Recall    : {m['recall'].item():.4f}   <- maximise (don't miss relevant papers)")
        print(f"  Precision : {m['precision'].item():.4f}")
        print(f"  AUROC     : {m['auroc'].item():.4f}")
        print(f"  Threshold : {self.cfg['inference']['threshold']}")
        print(f"  Confusion : TP={int(tp)} FP={int(fp)} FN={int(fn)} TN={int(tn)}")
        print(f"{'='*60}\n")
        self.test_metrics.reset(); self.test_cm.reset()

    def configure_optimizers(self):
        t_cfg = self.cfg["training"]
        no_decay = ["bias", "LayerNorm.weight"]
        groups = [
            {"params": [p for n,p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": float(t_cfg.get("weight_decay", 0.01))},
            {"params": [p for n,p in self.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(groups, lr=float(t_cfg["lr"]), eps=1e-8)
        warmup = int(self._total_steps * float(t_cfg.get("warmup_ratio", 0.06)))
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup,
                                                    num_training_steps=self._total_steps)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}


class PaperDataModule(pl.LightningDataModule):
    def __init__(self, cfg): super().__init__(); self.cfg = cfg

    def setup(self, stage=None):
        m_cfg = self.cfg["model"]; ds_dir = self.cfg["paths"]["dataset_dir"]
        print(f"Loading tokenizer: {m_cfg['name']}")
        self.tokenizer = AutoTokenizer.from_pretrained(m_cfg["name"])
        max_len = int(m_cfg["max_len"])
        print("Loading datasets ...")
        self.train_ds = PaperDataset(os.path.join(ds_dir,"train.jsonl"), self.tokenizer, max_len)
        self.val_ds   = PaperDataset(os.path.join(ds_dir,"val.jsonl"),   self.tokenizer, max_len)
        self.test_ds  = PaperDataset(os.path.join(ds_dir,"test.jsonl"),  self.tokenizer, max_len)

    def _loader(self, ds, shuffle):
        t_cfg = self.cfg["training"]; nw = int(t_cfg.get("num_workers", 4))
        return DataLoader(ds, batch_size=int(t_cfg["batch_size"]), shuffle=shuffle,
                          num_workers=nw, pin_memory=True, collate_fn=PaperDataset.collate,
                          persistent_workers=(nw > 0))

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)

    def total_steps(self):
        t_cfg = self.cfg["training"]
        return (len(self.train_ds)//int(t_cfg["batch_size"]) * int(t_cfg["epochs"])) \
               // int(t_cfg.get("accumulate_grad_batches", 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="bert_classifier_config.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.resume: cfg["training"]["resume"] = True

    pl.seed_everything(cfg["dataset"]["seed"], workers=True)

    p_cfg = cfg["paths"]; t_cfg = cfg["training"]; m_cfg = cfg["model"]
    run_dir  = os.path.join(p_cfg["runs_dir"], p_cfg["run_name"])
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(run_dir, exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))

    dm = PaperDataModule(cfg); dm.setup()
    total_steps = dm.total_steps()

    print(f"\n{'='*60}")
    print(f"  Model       : {m_cfg['name']}")
    print(f"  max_len     : {m_cfg['max_len']}")
    print(f"  batch_size  : {t_cfg['batch_size']}")
    print(f"  epochs      : {t_cfg['epochs']}")
    print(f"  total_steps : {total_steps:,}")
    print(f"  lr          : {t_cfg['lr']}")
    print(f"  precision   : {t_cfg.get('precision','bf16-mixed')}")
    print(f"  threshold   : {cfg['inference']['threshold']}")
    print(f"  run_dir     : {run_dir}")
    print(f"{'='*60}\n")

    model = PaperClassifier(cfg, total_steps=total_steps)

    callbacks = [
        ModelCheckpoint(dirpath=ckpt_dir,
                        filename="epoch{epoch:02d}-macro_f1_{val/macro_f1:.4f}",
                        monitor="val/macro_f1", mode="max",
                        save_top_k=int(t_cfg.get("save_top_k",3)), save_last=True,
                        auto_insert_metric_name=False),
        EarlyStopping(monitor="val/macro_f1", patience=int(t_cfg.get("early_stopping_patience",5)),
                      mode="max", verbose=True),
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=20),
    ]
    loggers = [
        TensorBoardLogger(run_dir, name="tensorboard", default_hp_metric=False),
        CSVLogger(run_dir, name="csv_logs"),
    ]

    ckpt_path = None
    if t_cfg.get("resume", False):
        last = os.path.join(ckpt_dir, "last.ckpt")
        if os.path.exists(last): ckpt_path = last; print(f"Resuming from: {ckpt_path}")
        else: print("No last.ckpt found — starting fresh.")

    # ── PL version-safe precision + strategy ─────────────────────────────────────
    _pl_major  = int(pl.__version__.split(".")[0])
    _precision = t_cfg.get("precision", "bf16")
    # Normalise precision string across PL versions
    if _pl_major >= 2 and _precision == "bf16":
        _precision = "bf16-mixed"
    elif _pl_major < 2 and _precision in ("bf16-mixed", "mixed"):
        _precision = "bf16"


    trainer_kwargs = dict(
        max_epochs=int(t_cfg["epochs"]),
        precision=_precision,
        accelerator="gpu",
        devices=1,                                        # explicit single GPU
        accumulate_grad_batches=int(t_cfg.get("accumulate_grad_batches", 2)),
        gradient_clip_val=float(t_cfg.get("gradient_clip", 1.0)),
        val_check_interval=t_cfg.get("val_check_interval", 1.0),
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=int(t_cfg.get("log_every_n_steps", 50)),
        deterministic=False,
        enable_model_summary=True,
)
    # strategy= only works on PL 2.x; PL 1.x single-GPU needs it omitted entirely
    if _pl_major >= 2:
        trainer_kwargs["strategy"] = "auto"

    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    print("\nEvaluating best checkpoint on test set ...")
    trainer.test(model, datamodule=dm, ckpt_path="best")

    hf_out = os.path.join(run_dir, "hf_model")
    os.makedirs(hf_out, exist_ok=True)
    model.encoder.save_pretrained(hf_out)
    dm.tokenizer.save_pretrained(hf_out)
    with open(os.path.join(hf_out, "inference_config.json"), "w") as f:
        json.dump({"threshold": cfg["inference"]["threshold"],
                   "max_len": cfg["model"]["max_len"],
                   "model_name": cfg["model"]["name"],
                   "input_fields": cfg["input_fields"],
                   "run_name": cfg["paths"]["run_name"]}, f, indent=2)

    print(f"\nHuggingFace model saved to: {hf_out}")
    print(f"\nAblation: edit model.name + run_name in config, re-run train script.")
    print(f"TensorBoard: tensorboard --logdir {p_cfg['runs_dir']}")


if __name__ == "__main__":
    main()
