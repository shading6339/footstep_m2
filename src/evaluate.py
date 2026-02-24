"""
evaluate.py
-----------
学習済みモデルを使って評価する。

使い方:
  # 特定フォールドのベストモデルを評価
  python evaluate.py --run_dir ../experiments/baseline_groupkfold_SimpleCNN_YYYYMMDD-HHMMSS

出力:
  - 混同行列 (PNG)
  - クラス別 F1 スコア (JSON)
  - フロア間誤分類ヒートマップ (PNG)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import FootstepDataset, load_splits
from models import build_model


# ---------- 推論 ----------

def predict(model, loader, device):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())
    return np.array(all_labels), np.array(all_preds)


# ---------- 混同行列プロット ----------

def plot_confusion_matrix(cm, class_names, save_path: Path, title: str = "Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


# ---------- フォールド別評価 ----------

def evaluate_run(run_dir: Path, device: torch.device):
    cfg_path = run_dir / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    dataset = FootstepDataset(cfg["data"]["source_dir"])
    splits_path = Path(cfg["data"]["splits_dir"]) / "group_kfold_splits.json"
    folds = load_splits(str(splits_path))

    fold_results = []

    for fold_info in folds:
        fold_i = fold_info["fold"]
        fold_dir = run_dir / f"fold{fold_i}"
        model_path = fold_dir / "best_model.pth"
        if not model_path.exists():
            print(f"  Fold {fold_i}: best_model.pth が見つかりません。スキップ。")
            continue

        val_ds = dataset.subset(fold_info["val"])
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

        model = build_model(cfg["model"]["name"], dataset.num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        labels, preds = predict(model, val_loader, device)

        acc = 100.0 * (labels == preds).mean()
        f1_w = f1_score(labels, preds, average="weighted")
        report = classification_report(
            labels, preds,
            target_names=[f"class{dataset.label_to_class[i]}" for i in range(dataset.num_classes)],
            output_dict=True,
        )
        cm = confusion_matrix(labels, preds)

        # 混同行列を保存
        class_names = [f"c{dataset.label_to_class[i]}" for i in range(dataset.num_classes)]
        plot_confusion_matrix(
            cm, class_names,
            save_path=fold_dir / "confusion_matrix.png",
            title=f"Fold {fold_i} — {cfg['model']['name']} (acc={acc:.1f}%)",
        )

        # クラス別 F1 を保存
        with open(fold_dir / "classification_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"  Fold {fold_i}: acc={acc:.2f}%  weighted-F1={f1_w:.4f}")
        fold_results.append({"fold": fold_i, "acc": acc, "f1_weighted": f1_w})

    if not fold_results:
        print("評価できるフォールドがありませんでした。")
        return

    accs = [r["acc"] for r in fold_results]
    f1s  = [r["f1_weighted"] for r in fold_results]

    summary = {
        "model": cfg["model"]["name"],
        "folds": fold_results,
        "mean_acc": float(np.mean(accs)),
        "std_acc":  float(np.std(accs)),
        "mean_f1":  float(np.mean(f1s)),
        "std_f1":   float(np.std(f1s)),
    }
    with open(run_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== 評価サマリ ({cfg['model']['name']}) ===")
    print(f"  Accuracy : {summary['mean_acc']:.2f} ± {summary['std_acc']:.2f} %")
    print(f"  F1 (weighted): {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
    print(f"  結果保存先: {run_dir}")


# ---------- メイン ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="train.py が生成した実験ディレクトリ")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_run(Path(args.run_dir), device)


if __name__ == "__main__":
    main()
