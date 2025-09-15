# -*- coding: utf-8 -*-
"""
Train BloodMNIST (MedMNIST) — no k-fold, metrics: ACC and AUC only.
Early stopping now uses validation ACC.
"""
import os
import sys
import json
import time
import random
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

# ===== MedMNIST =====
try:
    import medmnist
    from medmnist import INFO, BloodMNIST
except Exception as e:
    print("Please install medmnist first: pip install medmnist")
    raise e

# ===== Your models (import as in your codebase) =====
from MedMamba import VSSM as vssm
from MedMamba import VSSMEdgeEnhanced as edge_enhanced
from MedMamba import DualBranchVSSM as dual_branch
from MedMamba import DualBranchVSSMEnhanced as dual_branch_enhanced

MODEL_MAP = {
    'vssm': vssm,
    'edge_enhanced': edge_enhanced,
    'dual_branch': dual_branch,
    'dual_branch_enhanced': dual_branch_enhanced,
}

SIZE_CONFIG = {
    't': {'depths': [2, 2, 4, 2],  'dims': [96, 192, 384, 768]},
    's': {'depths': [2, 2, 8, 2],  'dims': [96, 192, 384, 768]},
    'b': {'depths': [2, 2, 12, 2], 'dims': [128, 256, 512, 1024]},
}

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logger_and_saver(exp_name="bloodmnist"):
    current_time = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join("./logs", f"{exp_name}-{current_time}")
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, "log.txt")
    ckpt_path = os.path.join(out_dir, "best.pth")
    cfg_path = os.path.join(out_dir, "args.json")
    metrics_path = os.path.join(out_dir, "metrics.json")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return out_dir, ckpt_path, cfg_path, metrics_path

def build_dataloaders(size: int, batch_size: int, num_workers: int, download: bool=True):
    """BloodMNIST loaders for train/val/test. Labels are converted to int class ids."""
    def _target_transform(y):
        try:
            return int(y)
        except Exception:
            return int(y[0])

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_set = BloodMNIST(split="train", transform=transform,
                           target_transform=_target_transform, download=download, size=size)
    val_set   = BloodMNIST(split="val",   transform=transform,
                           target_transform=_target_transform, download=download, size=size)
    test_set  = BloodMNIST(split="test",  transform=transform,
                           target_transform=_target_transform, download=download, size=size)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, drop_last=False)

    info = INFO['bloodmnist']
    n_classes = len(info['label']) if isinstance(info['label'], dict) else int(info['n_classes'])

    return train_loader, val_loader, test_loader, n_classes

@torch.no_grad()
def evaluate_acc_auc(model, loader, device, num_classes):
    model.eval()
    all_logits = []
    all_labels = []

    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)  # (B, C)
        all_logits.append(outputs.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0)              # (N, C)
    labels = torch.cat(all_labels, dim=0).numpy()      # (N,)
    probs = torch.softmax(logits, dim=1).numpy()       # (N, C)

    preds = probs.argmax(axis=1)
    acc = accuracy_score(labels, preds)

    try:
        if num_classes == 2:
            auc = roc_auc_score(labels, probs[:, 1])
        else:
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception as e:
        logging.warning(f"AUC computation failed ({e}); setting AUC=nan")
        auc = float("nan")

    return {"acc": float(acc), "auc": float(auc)}

def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()
    running_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{running_loss/n:.4f}")
    return running_loss / max(n, 1)

def main():
    parser = argparse.ArgumentParser(description="Train BloodMNIST (Early stop on ACC)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--size", type=int, default=224, choices=[28, 224],
                        help="28 for standard, 224 for MedMNIST+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model-type", type=str, default="vssm",
                        choices=list(MODEL_MAP.keys()))
    parser.add_argument("--size-type", type=str, default="t", choices=list(SIZE_CONFIG.keys()),
                        help="Backbone size (t/s/b) for your MedMamba models")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=10, help="early stop patience (epochs)")
    args = parser.parse_args()

    seed_everything(args.seed)
    out_dir, ckpt_path, cfg_path, metrics_path = setup_logger_and_saver(exp_name="bloodmnist")
    logging.info(json.dumps(vars(args), indent=2))
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device)
    train_loader, val_loader, test_loader, num_classes = build_dataloaders(
        size=args.size, batch_size=args.batch_size, num_workers=args.num_workers
    )

    size_cfg = SIZE_CONFIG[args.size_type]
    model_fn = MODEL_MAP[args.model_type]
    model = model_fn(in_chans=3, num_classes=num_classes,
                     depths=size_cfg["depths"], dims=size_cfg["dims"])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = -1.0
    best_epoch = -1
    history = {"train_loss": [], "val_acc": [], "val_auc": []}

    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        logging.info(f"Epoch {epoch}/{args.epochs} (lr={optimizer.param_groups[0]['lr']:.6f})")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        scheduler.step()

        val_metrics = evaluate_acc_auc(model, val_loader, device, num_classes)
        logging.info(f"Val ACC: {val_metrics['acc']:.4f} | Val AUC: {val_metrics['auc']:.4f} | TrainLoss: {train_loss:.4f}")

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_metrics["acc"])
        history["val_auc"].append(val_metrics["auc"])

        # Early stopping on ACC
        current_acc = val_metrics["acc"]
        if current_acc > best_val_acc:
            best_val_acc = current_acc
            best_epoch = epoch
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "val_acc": best_val_acc}, ckpt_path)
            logging.info(f"New best ACC: {best_val_acc:.4f} at epoch {epoch}. Model saved to {ckpt_path}.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logging.info(f"Early stopping at epoch {epoch} (no improve for {args.patience} epochs).")
                break

    # Load best and evaluate on test
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        logging.info(f"Loaded best checkpoint from epoch {ckpt.get('epoch', '?')} with Val ACC {ckpt.get('val_acc', 'NA')}")

    test_metrics = evaluate_acc_auc(model, test_loader, device, num_classes)
    logging.info(f"TEST — ACC: {test_metrics['acc']:.4f} | AUC: {test_metrics['auc']:.4f}")

    summary = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_acc": test_metrics["acc"],
        "test_auc": test_metrics["auc"],
        "history": history
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Metrics saved to:", metrics_path)

if __name__ == "__main__":
    main()
