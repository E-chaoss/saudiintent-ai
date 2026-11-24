"""Task 4: fine-tune MARBERTv2 to classify the 8 intents.

What it does:
  1. Reads train.csv and validation.csv (normalized with normalize_ar).
  2. Tries every learning rate x epoch count from config.yaml (6 runs by default).
  3. Scores each run on the VALIDATION set (macro-F1) and keeps only the best model.
  4. Writes the scores of every run and the best settings to results/marbert/.

The Saudi test set is never read here: choosing settings on it would make the final test unfair.

Run from the repo root, on a GPU (e.g. Google Colab):
    python src/marbert/train.py                       # full search from config.yaml
    python src/marbert/train.py --lr 2e-5 --epochs 3  # one run only (quick try)
"""
import argparse
import json
import platform
import random
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import transformers
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data import ROOT, class_weights, load_split, search_grid   # data.py, next to this file
from labels import ID2LABEL, INTENTS, LABEL2ID                    # src/labels.py (path set by data.py)

CONFIG_PATH = ROOT / "src" / "marbert" / "config.yaml"


def set_seed(seed):
    """Same seed -> same random start (weights of the new classification head, data order)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(texts, labels, tokenizer, cfg, shuffle, seed):
    """Batches of (input_ids, attention_mask, labels). Each batch is padded to its longest message."""
    pairs = list(zip(texts, labels))

    def collate(batch):
        batch_texts, batch_labels = zip(*batch)
        enc = tokenizer(list(batch_texts), padding=True, truncation=True,
                        max_length=cfg["max_length"], return_tensors="pt")
        enc["labels"] = torch.tensor(batch_labels, dtype=torch.long)
        return enc

    gen = torch.Generator().manual_seed(seed)
    return DataLoader(pairs, batch_size=cfg["batch_size"], shuffle=shuffle,
                      collate_fn=collate, generator=gen if shuffle else None)


@torch.no_grad()
def predict_loader(model, loader, device):
    """Return predicted label ids and probabilities for every message in the loader."""
    model.eval()
    all_probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**batch).logits
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probs = np.concatenate(all_probs)
    return probs.argmax(axis=1), probs


def scores(gold, pred):
    return {
        "macro_f1": f1_score(gold, pred, average="macro", labels=list(range(len(INTENTS))), zero_division=0),
        "accuracy": accuracy_score(gold, pred),
    }


def train_one_run(lr, n_epochs, cfg, tokenizer, train_data, val_data, device):
    """Fine-tune a fresh copy of MARBERTv2 with one setting. Returns (model, run_info, val_pred, val_probs)."""
    set_seed(cfg["seed"])       # every run starts from the same random state -> fair comparison
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=len(INTENTS), id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(device)

    train_loader = make_loader(*train_data, tokenizer, cfg, shuffle=True, seed=cfg["seed"])
    val_loader = make_loader(*val_data, tokenizer, cfg, shuffle=False, seed=cfg["seed"])

    weight = None
    if cfg["class_weights"]:
        weight = torch.tensor(class_weights(train_data[1]), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    total_steps = len(train_loader) * n_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(cfg["warmup_ratio"] * total_steps), num_training_steps=total_steps)

    history = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            loss = loss_fn(model(**batch).logits, labels)
            loss.backward()                                     # compute gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optimizer.step()                                    # update the weights
            scheduler.step()                                    # update the learning rate
            optimizer.zero_grad()
            epoch_loss += loss.item()
        val_pred, val_probs = predict_loader(model, val_loader, device)
        s = scores(val_data[1], val_pred)
        history.append({"epoch": epoch, "train_loss": epoch_loss / len(train_loader), **s})
        print(f"  epoch {epoch}/{n_epochs}  train_loss={history[-1]['train_loss']:.4f}  "
              f"val_macro_f1={s['macro_f1']:.4f}  val_accuracy={s['accuracy']:.4f}")

    # The run is scored after its LAST epoch: the number of epochs is part of the setting.
    run = {"learning_rate": lr, "epochs": n_epochs, **history[-1], "history": history}
    return model, run, val_pred, val_probs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--lr", type=float, help="run only this learning rate")
    ap.add_argument("--epochs", type=int, help="run only this number of epochs")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.lr is not None:
        cfg["learning_rates"] = [args.lr]
    if args.epochs is not None:
        cfg["epochs"] = [args.epochs]
    metric = cfg["selection_metric"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no GPU found. Training will be very slow. On Colab: Runtime > Change runtime type > GPU.")

    train_data = load_split(cfg["paths"]["train"])
    val_data = load_split(cfg["paths"]["validation"])
    print(f"train: {len(train_data[0])} rows | validation: {len(val_data[0])} rows | device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model_dir = ROOT / cfg["paths"]["model_dir"]
    results_dir = ROOT / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    grid = search_grid(cfg)
    runs, best, started = [], None, time.time()
    for i, (lr, n_epochs) in enumerate(grid, 1):
        print(f"\nRun {i}/{len(grid)}: learning_rate={lr:g}, epochs={n_epochs}")
        model, run, val_pred, val_probs = train_one_run(lr, n_epochs, cfg, tokenizer, train_data, val_data, device)
        runs.append(run)
        # Keep this run only if it is strictly better (a tie keeps the earlier, simpler setting).
        if best is None or run[metric] > best[metric]:
            best = run
            if model_dir.exists():
                shutil.rmtree(model_dir)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            best_val = (val_pred, val_probs)
            print(f"  -> new best ({metric}={run[metric]:.4f}), saved to {model_dir.relative_to(ROOT)}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- results (small files, committed to Git) ----
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in runs])
    table.to_csv(results_dir / "search_results.csv", index=False)
    with open(results_dir / "search_history.json", "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)

    val_pred, val_probs = best_val
    gold = val_data[1]
    report = classification_report(gold, val_pred, labels=list(range(len(INTENTS))),
                                   target_names=INTENTS, digits=4, zero_division=0)
    (results_dir / "validation_report_best.txt").write_text(report, encoding="utf-8")
    cm = confusion_matrix(gold, val_pred, labels=list(range(len(INTENTS))))
    pd.DataFrame(cm, index=INTENTS, columns=INTENTS).to_csv(results_dir / "validation_confusion_matrix_best.csv")

    info = {
        "best": {"learning_rate": best["learning_rate"], "epochs": best["epochs"],
                 "val_macro_f1": best["macro_f1"], "val_accuracy": best["accuracy"]},
        "selected_by": f"{metric} on the validation set",
        "config": cfg,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__,
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "none"},
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "minutes": round((time.time() - started) / 60, 1),
    }
    with open(results_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    with open(model_dir / "training_info.json", "w", encoding="utf-8") as f:   # travels with the model
        json.dump(info, f, indent=2)

    print("\n" + table.to_string(index=False))
    print(f"\nBest: learning_rate={best['learning_rate']:g}, epochs={best['epochs']}, "
          f"val macro-F1={best['macro_f1']:.4f}, val accuracy={best['accuracy']:.4f}")
    print(f"Model: {model_dir.relative_to(ROOT)}   Results: {results_dir.relative_to(ROOT)}")
    print("\nValidation report (best run):\n" + report)


if __name__ == "__main__":
    main()
