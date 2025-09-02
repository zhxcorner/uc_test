# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import argparse
import logging
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, models
from tqdm import tqdm
from sklearn.model_selection import KFold


# ========== Utils ==========
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger_and_saver(model_name="ResNet101"):
    current_time = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("../logs", current_time)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "log.txt")
    model_save_path = os.path.join(log_dir, f"{model_name}_best.pth")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"Logging to: {log_file}")
    logging.info(f"Best model (per fold) will be saved under: {log_dir}")
    return log_dir, model_save_path


def compute_mean_std(dataset_root, indices, batch_size=64, num_workers=4):
    """
    计算指定 indices 图像的 mean 和 std（不修改原始 dataset）
    """
    temp_dataset = datasets.ImageFolder(
        root=dataset_root,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    )
    subset = Subset(temp_dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    with torch.no_grad():
        for images, _ in loader:
            batch_size = images.size(0)
            mean += images.mean(dim=[0, 2, 3]) * batch_size
            std += images.std(dim=[0, 2, 3]) * batch_size
            total_images += batch_size

    mean /= total_images
    std /= total_images

    return mean.tolist(), std.tolist()


# ========== Model ==========
def build_resnet101(num_classes: int = 2, pretrained: bool = False):
    if pretrained:
        try:
            weights = models.ResNet101_Weights.IMAGENET1K_V2
        except AttributeError:
            weights = "IMAGENET1K_V2"
        model = models.resnet101(weights=weights)
    else:
        model = models.resnet101(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# ========== Train / Evaluate ==========
def train_one_epoch(model, loader, device, optimizer, loss_fn):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False, file=sys.stdout)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Validating", leave=False, file=sys.stdout)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="Train ResNet101 with 5-Fold Cross Validation")
    # 数据 & 训练相关
    parser.add_argument("--dataset", type=str, default="单个细胞分类数据集二分类S2L", help="ImageFolder 根目录名（挂载在 /data 下）")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # 模型配置
    parser.add_argument("--model_name", type=str, default="ResNet101")

    args = parser.parse_args()
    seed_everything(args.seed)

    # 日志与保存
    log_dir, base_save_path = setup_logger_and_saver(args.model_name)
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)
    logging.info(f"Training parameters saved to: {config_path}")

    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    # 数据集（先不加 transform，后面每折单独加）
    dataset_root = f"/data/{args.dataset}"
    full_dataset = datasets.ImageFolder(root=dataset_root, transform=None)  # 先不设 transform
    indices = np.arange(len(full_dataset))

    # 五折
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices), start=1):
        print(f"\n========== Fold {fold}/{args.k_folds} ==========")

        # --- 计算当前 fold 训练集的 mean 和 std ---
        print(f"Fold {fold}: Computing mean and std on training set...")
        mean, std = compute_mean_std(dataset_root, train_idx, batch_size=args.batch_size, num_workers=args.num_workers)
        logging.info(f"Fold {fold} - Computed mean: {mean}, std: {std}")

        # --- 构建每折的 transform ---
        data_transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        data_transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        # --- 为子集设置 transform ---
        train_dataset = datasets.ImageFolder(root=dataset_root, transform=data_transform_train)
        val_dataset = datasets.ImageFolder(root=dataset_root, transform=data_transform_val)

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(val_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # 模型 & 优化器 & 调度器
        model = build_resnet101(num_classes=args.num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs // 2)  # 👈 新增调度器
        loss_fn = nn.CrossEntropyLoss()

        best_acc = 0.0
        fold_save_path = base_save_path.replace("_best.pth", f"_fold{fold}_best.pth")

        for epoch in range(1, args.epochs + 1):
            avg_loss = train_one_epoch(model, train_loader, device, optimizer, loss_fn)
            val_acc = evaluate(model, val_loader, device)

            scheduler.step()  # 👈 更新学习率

            current_lr = optimizer.param_groups[0]['lr']
            print(f"[Fold {fold}][Epoch {epoch}/{args.epochs}] "
                  f"Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.2e}")
            logging.info(f"[Fold {fold}][Epoch {epoch}/{args.epochs}] "
                         f"Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.2e}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), fold_save_path)
                print(f"  -> Saved new best model to: {fold_save_path} (Val Acc: {best_acc:.4f})")
                logging.info(f"Saved new best model to: {fold_save_path} (Val Acc: {best_acc:.4f})")

        fold_results.append(best_acc)

    # 汇总
    avg_acc = float(np.mean(fold_results))
    var_acc = float(np.var(fold_results))
    print("\n========== Cross-Validation Results ==========")
    print(f"Per-fold best accuracies: {fold_results}")
    print(f"Average accuracy across all folds: {avg_acc:.4f}")
    print(f"Variance of accuracy across all folds: {var_acc:.4f}")
    logging.info(f"Per-fold best accuracies: {fold_results}")
    logging.info(f"Average accuracy across all folds: {avg_acc:.4f}")
    logging.info(f"Variance of accuracy across all folds: {var_acc:.4f}")
    logging.info("Finished Training")
    print("Finished Training")


if __name__ == "__main__":
    main()