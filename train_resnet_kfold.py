# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import argparse
import logging
import random
import numpy as np
import pickle
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, models
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


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


# ========== Evaluate (Multi-metric) ==========
@torch.no_grad()
def evaluate_with_metrics(model, loader, device, num_classes):
    """
    返回 Accuracy, Precision, Recall, F1
    """
    model.eval()
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Evaluating", leave=False, file=sys.stdout)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 转为 numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Accuracy
    acc = accuracy_score(all_labels, all_preds)

    # Precision, Recall, F1 (weighted for multi-class)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    return {
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ========== Train One Epoch ==========
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


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="Train ResNet101 with 5-Fold Cross Validation")
    parser.add_argument("--dataset", type=str, default="单个细胞分类数据集二分类S2L", help="ImageFolder 根目录名（挂载在 /data 下）")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
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
    print(f"Using device: {device}")

    # 数据集路径
    dataset_root = f"/data/{args.dataset}"
    full_dataset = datasets.ImageFolder(root=dataset_root, transform=None)
    logging.info(f"Loaded dataset from: {dataset_root}, total samples: {len(full_dataset)}")

    # ================================
    # 🚀 加载 kfold_splits.pkl（包含 mean/std）
    # ================================
    kfold_pkl_path = os.path.join(dataset_root, "kfold_splits.pkl")
    if not os.path.exists(kfold_pkl_path):
        raise FileNotFoundError(f"❌ 找不到划分文件: {kfold_pkl_path}\n请先运行划分生成脚本。")

    logging.info(f"✅ 加载划分文件: {kfold_pkl_path}")
    with open(kfold_pkl_path, 'rb') as f:
        data = pickle.load(f)

    splits = data['splits']
    fold_data = []

    # ✅ 构建完整路径 → 索引 映射
    path_to_idx = {
        os.path.join(dataset_root, img_path).replace("\\", "/").replace("//", "/"): idx
        for idx, (img_path, _) in enumerate(full_dataset.imgs)
    }

    for fold_idx, split in enumerate(splits):
        rel_train_paths = split['train']
        rel_val_paths = split['val']
        mean = split['mean']
        std = split['std']

        # ✅ 拼接完整路径
        full_train_paths = [
            os.path.join(dataset_root, p).replace("\\", "/") for p in rel_train_paths
        ]
        full_val_paths = [
            os.path.join(dataset_root, p).replace("\\", "/") for p in rel_val_paths
        ]

        # 映射到索引
        train_idx = [path_to_idx[p] for p in full_train_paths if p in path_to_idx]
        val_idx = [path_to_idx[p] for p in full_val_paths if p in path_to_idx]

        # 调试输出
        print(f"\nFold {fold_idx + 1} 路径匹配情况:")
        print(f"  请求的训练图像数: {len(full_train_paths)}")
        print(f"  成功映射的训练图像数: {len(train_idx)}")
        print(f"  请求的验证图像数: {len(full_val_paths)}")
        print(f"  成功映射的验证图像数: {len(val_idx)}")

        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(f"Fold {fold_idx + 1} 的训练或验证集为空，请检查路径一致性")

        fold_data.append({
            'train_idx': train_idx,
            'val_idx': val_idx,
            'mean': mean,
            'std': std
        })

    logging.info(f"✅ 成功加载 {len(fold_data)} 折划分（含 mean/std）")

    # ================================
    # 开始 K-Fold 训练
    # ================================
    fold_results = []

    for fold, data in enumerate(fold_data, start=1):
        print(f"\n========== Fold {fold}/{len(fold_data)} ==========")
        train_idx = data['train_idx']
        val_idx = data['val_idx']
        mean = data['mean']
        std = data['std']

        logging.info(f"Fold {fold} - mean: {mean}, std: {std}")

        # 数据增强与归一化
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

        # 构建数据集
        train_dataset = datasets.ImageFolder(root=dataset_root, transform=data_transform_train)
        val_dataset = datasets.ImageFolder(root=dataset_root, transform=data_transform_val)

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(val_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # 模型、优化器、损失
        model = build_resnet101(num_classes=args.num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs // 2)
        loss_fn = nn.CrossEntropyLoss()

        best_metrics = None
        fold_save_path = base_save_path.replace("_best.pth", f"_fold{fold}_best.pth")

        for epoch in range(1, args.epochs + 1):
            avg_loss = train_one_epoch(model, train_loader, device, optimizer, loss_fn)
            metrics = evaluate_with_metrics(model, val_loader, device, args.num_classes)
            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            log_msg = (f"[Fold {fold}][Epoch {epoch}/{args.epochs}] "
                       f"Loss: {avg_loss:.4f} | "
                       f"Acc: {metrics['acc']:.4f} | "
                       f"Precision: {metrics['precision']:.4f} | "
                       f"Recall: {metrics['recall']:.4f} | "
                       f"F1: {metrics['f1']:.4f} | "
                       f"LR: {current_lr:.2e}")
            print(log_msg)
            logging.info(log_msg)

            # 保存最佳模型（以 Accuracy 为标准）
            if best_metrics is None or metrics['acc'] > best_metrics['acc']:
                best_metrics = metrics.copy()
                torch.save(model.state_dict(), fold_save_path)
                logging.info(f"✅ Saved best model (Acc: {metrics['acc']:.4f}) to {fold_save_path}")

        # 记录本折最佳指标
        fold_results.append(best_metrics)
        print(f"📌 Fold {fold} Best Metrics: {best_metrics}")

    # ================================
    # 汇总结果
    # ================================
    all_acc = [r['acc'] for r in fold_results]
    all_prec = [r['precision'] for r in fold_results]
    all_rec = [r['recall'] for r in fold_results]
    all_f1 = [r['f1'] for r in fold_results]

    summary = {
        "Average Accuracy": float(np.mean(all_acc)),
        "Std Accuracy": float(np.std(all_acc)),
        "Average Precision": float(np.mean(all_prec)),
        "Average Recall": float(np.mean(all_rec)),
        "Average F1": float(np.mean(all_f1)),
        "Per-fold Results": [
            {
                "fold": i + 1,
                "acc": r['acc'],
                "precision": r['precision'],
                "recall": r['recall'],
                "f1": r['f1']
            }
            for i, r in enumerate(fold_results)
        ]
    }

    # 打印汇总
    print("\n========== Cross-Validation Results ==========")
    for k, v in summary.items():
        if k != "Per-fold Results":
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # 保存汇总
    summary_path = os.path.join(log_dir, "cv_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    logging.info(f"✅ CV Summary saved to: {summary_path}")
    logging.info("✅ Training completed.")


if __name__ == "__main__":
    main()