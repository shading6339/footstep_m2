"""
train.py
--------
GroupKFold による再現性のある学習ループ。

使い方:
  python train.py --config ../config/base.yaml
  python train.py --config ../config/base.yaml --model ResNet18
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

# src/ を直接実行した場合でも import できるようにする
sys.path.insert(0, str(Path(__file__).parent))

from dataset import FootstepDataset, make_group_kfold_splits, load_splits
from models import build_model


# ---------- 乱数シード固定 ----------

def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------- Early Stopping ----------

class EarlyStopping:
    def __init__(self, patience: int, delta: float):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ---------- 1 エポック学習 ----------

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ---------- 検証 ----------

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


# ---------- 1 フォールド学習 ----------

def train_fold(fold_info: dict, dataset: FootstepDataset, cfg: dict,
               run_dir: Path, device: torch.device):
    fold_i = fold_info["fold"]
    fold_dir = run_dir / f"fold{fold_i}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_ds = dataset.subset(fold_info["train"])
    val_ds   = dataset.subset(fold_info["val"])

    t_cfg = cfg["training"]
    pin = device.type == "cuda"
    nw = 0 if device.type in ("mps", "cpu") else 4
    train_loader = DataLoader(train_ds, batch_size=t_cfg["batch_size"],
                              shuffle=True,  num_workers=nw, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=t_cfg["batch_size"],
                              shuffle=False, num_workers=nw, pin_memory=pin)

    model = build_model(cfg["model"]["name"], dataset.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=t_cfg["learning_rate"],
                           weight_decay=t_cfg["weight_decay"])

    early_stopping = EarlyStopping(
        patience=t_cfg["early_stopping_patience"],
        delta=t_cfg["early_stopping_delta"],
    )

    writer = SummaryWriter(log_dir=str(fold_dir / "tensorboard"))
    best_val_acc = 0.0
    history = []

    print(f"\n--- Fold {fold_i} | train={len(train_ds)}, val={len(val_ds)} ---")

    for epoch in range(t_cfg["num_epochs"]):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "val_acc": val_acc})

        print(f"  Epoch {epoch+1:>4d}/{t_cfg['num_epochs']} | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), fold_dir / "best_model.pth")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"  EarlyStopping at epoch {epoch+1}")
            break

    writer.close()
    with open(fold_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"  Fold {fold_i} 完了: best_val_acc = {best_val_acc:.2f}%")
    return best_val_acc


# ---------- メイン ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/base.yaml")
    parser.add_argument("--model",  default=None, help="config の model.name を上書き")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model:
        cfg["model"]["name"] = args.model

    fix_seed(cfg["seed"])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")
    print(f"model : {cfg['model']['name']}")
    print(f"seed  : {cfg['seed']}")

    # データセット（参照のみ、元データは変更しない）
    dataset = FootstepDataset(cfg["data"]["source_dir"])
    print(f"総サンプル数: {dataset.full_size}, クラス数: {dataset.num_classes}")

    # スプリット生成 or 読み込み
    splits_path = Path(cfg["data"]["splits_dir"]) / "group_kfold_splits.json"
    if splits_path.exists():
        folds = load_splits(str(splits_path))
        print(f"既存スプリット読み込み: {splits_path}")
    else:
        folds = make_group_kfold_splits(
            dataset,
            n_folds=cfg["data"]["n_folds"],
            save_dir=cfg["data"]["splits_dir"],
        )

    # 実験ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{cfg['experiment']['name']}_{cfg['model']['name']}_{timestamp}"
    run_dir = Path(cfg["experiment"]["output_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # config を実験ディレクトリにコピー（再現性のため）
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    # 全フォールド学習
    fold_accs = []
    for fold_info in folds:
        acc = train_fold(fold_info, dataset, cfg, run_dir, device)
        fold_accs.append(acc)

    # 結果サマリ
    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    summary = {
        "model": cfg["model"]["name"],
        "seed":  cfg["seed"],
        "fold_accs": fold_accs,
        "mean_acc": mean_acc,
        "std_acc":  std_acc,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== 結果サマリ ===")
    for i, acc in enumerate(fold_accs):
        print(f"  Fold {i}: {acc:.2f}%")
    print(f"  Mean ± Std: {mean_acc:.2f} ± {std_acc:.2f}%")
    print(f"  結果保存先: {run_dir}")


if __name__ == "__main__":
    main()
