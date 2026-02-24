"""
dataset.py
----------
ファイル命名規則: f14_class{C}_{session}_{clip}_{start_ms}_p.png
  C       : クラスラベル (1-14)
  session : 録音セッションID (同一セッション内のクリップは同じ録音)
  clip    : セッション内のクリップ番号
  start_ms: 元録音内での開始時刻(ms)

GroupKFold はセッション単位 (class + session) で行う。
同一セッションのクリップが train/val に混在しないことを保証する。
"""

import os
import re
import json
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from sklearn.model_selection import GroupKFold


# ---------- ファイル名パーサ ----------

_PATTERN = re.compile(
    r"^f14_class(\d+)_(\d+)_(\d+)_(\d+)_p\.png$"
)

def parse_filename(fname: str) -> dict | None:
    """ファイル名を解析して辞書を返す。パターン不一致なら None。"""
    m = _PATTERN.match(fname)
    if m is None:
        return None
    cls, session, clip, start_ms = m.groups()
    return {
        "filename": fname,
        "class_id": int(cls),
        "session_id": int(session),
        "clip_id": int(clip),
        "start_ms": int(start_ms),
        # GroupKFold 用のグループキー: 同クラス×同セッションを 1 グループとする
        "group": f"class{cls}_session{session}",
    }


# ---------- Dataset ----------

class FootstepDataset(Dataset):
    """
    卒研アーカイブの PNG スペクトログラムを読み込む Dataset。
    source_dir は読み取り専用（変更しない）。
    """

    def __init__(self, source_dir: str, indices=None, transform=None):
        self.source_dir = Path(source_dir)
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
        ])

        # 全ファイルをスキャン
        records = []
        for fname in sorted(self.source_dir.iterdir()):
            if not fname.name.endswith(".png"):
                continue
            info = parse_filename(fname.name)
            if info is None:
                continue
            records.append(info)

        if not records:
            raise RuntimeError(f"PNG ファイルが見つかりません: {source_dir}")

        self.records = records

        # class_id → label (0-indexed)
        class_ids = sorted({r["class_id"] for r in records})
        self.class_to_label = {c: i for i, c in enumerate(class_ids)}
        self.label_to_class = {i: c for c, i in self.class_to_label.items()}
        self.num_classes = len(class_ids)

        # numpy 配列化（GroupKFold に渡すため）
        self.all_labels = np.array([self.class_to_label[r["class_id"]] for r in records])
        self.all_groups = np.array([r["group"] for r in records])

        # indices が指定された場合はサブセット
        self._indices = list(indices) if indices is not None else list(range(len(records)))

    # --- 全データへのアクセサ（GroupKFold 生成時に使う）---

    @property
    def full_size(self) -> int:
        return len(self.records)

    def get_all_labels(self) -> np.ndarray:
        return self.all_labels

    def get_all_groups(self) -> np.ndarray:
        return self.all_groups

    # --- torch Dataset インタフェース ---

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        record = self.records[self._indices[idx]]
        img_path = self.source_dir / record["filename"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = self.class_to_label[record["class_id"]]
        return image, label

    def subset(self, indices) -> "FootstepDataset":
        """indices（全データ基準）でサブセットを返す。変換は引き継ぐ。"""
        ds = FootstepDataset.__new__(FootstepDataset)
        ds.source_dir = self.source_dir
        ds.transform = self.transform
        ds.records = self.records
        ds.class_to_label = self.class_to_label
        ds.label_to_class = self.label_to_class
        ds.num_classes = self.num_classes
        ds.all_labels = self.all_labels
        ds.all_groups = self.all_groups
        ds._indices = list(indices)
        return ds


# ---------- GroupKFold スプリット生成 ----------

def make_group_kfold_splits(
    dataset: FootstepDataset,
    n_folds: int = 5,
    save_dir: str | None = None,
) -> list[dict]:
    """
    録音セッション単位で GroupKFold を生成する。

    Returns
    -------
    folds : list of dict
        [{"train": [idx, ...], "val": [idx, ...]}, ...]
        idx は dataset.records の全データ基準のインデックス。
    """
    gkf = GroupKFold(n_splits=n_folds)
    X = np.arange(dataset.full_size)
    y = dataset.get_all_labels()
    groups = dataset.get_all_groups()

    folds = []
    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        folds.append({
            "fold": fold_i,
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
        })
        _check_leakage(train_idx, val_idx, groups)

    if save_dir is not None:
        save_path = Path(save_dir) / "group_kfold_splits.json"
        with open(save_path, "w") as f:
            json.dump(folds, f, indent=2)
        print(f"Splits saved: {save_path}")

    return folds


def _check_leakage(train_idx, val_idx, groups):
    """train と val でグループが被っていないか確認する。"""
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    overlap = train_groups & val_groups
    if overlap:
        raise RuntimeError(f"データリーク検出: 重複グループ = {overlap}")


def load_splits(splits_path: str) -> list[dict]:
    with open(splits_path) as f:
        return json.load(f)


# ---------- 動作確認 ----------

if __name__ == "__main__":
    import argparse, yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/base.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds = FootstepDataset(cfg["data"]["source_dir"])
    print(f"総サンプル数: {ds.full_size}")
    print(f"クラス数: {ds.num_classes}")
    print(f"グループ数 (録音セッション): {len(set(ds.get_all_groups()))}")

    unique, counts = np.unique(ds.get_all_labels(), return_counts=True)
    print("\nクラス別サンプル数:")
    for lbl, cnt in zip(unique, counts):
        print(f"  class{ds.label_to_class[lbl]:>2d}: {cnt:>4d}")

    folds = make_group_kfold_splits(
        ds,
        n_folds=cfg["data"]["n_folds"],
        save_dir=cfg["data"]["splits_dir"],
    )
    print(f"\n{cfg['data']['n_folds']}-fold スプリット生成完了")
    for fold in folds:
        print(f"  fold {fold['fold']}: train={len(fold['train'])}, val={len(fold['val'])}")
