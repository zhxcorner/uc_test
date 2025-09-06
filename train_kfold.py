# -*- coding: utf-8 -*-
import os
import sys
import json
import logging
import time
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
from collections import defaultdict
import pickle
# 动态导入模型
from MedMamba import VSSM as vssm
from MedMamba import VSSMEdgeEnhanced as edge_enhanced
from MedMamba import DualBranchVSSM as dual_branch
from MedMamba import DualBranchVSSMEnhanced as dual_branch_enhanced

# 模型字典
MODEL_MAP = {
    'vssm': vssm,
    'edge_enhanced': edge_enhanced,
    'dual_branch': dual_branch,
    'dual_branch_enhanced': dual_branch_enhanced,
}

# ========== Utils ==========
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger_and_saver(model_name="UC"):
    current_time = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("../logs", current_time)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "log.txt")
    model_save_path = os.path.join(log_dir, f"{model_name}Net_best.pth")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info(f"Logging to: {log_file}")
    logging.info(f"Best model will be saved to: {model_save_path}")

    return log_dir, model_save_path


@torch.no_grad()
def evaluate_with_metrics(model, loader, device, num_classes):
    """
    多指标评估：Accuracy, Precision, Recall, F1 (weighted)
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

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    return {
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="Train MedMamba Model with Model Selection and Edge Fusion Options")
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['vssm', 'edge_enhanced', 'dual_branch', 'dual_branch_enhanced'],
                        help='Type of model to use')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of output classes (default: 2)')
    parser.add_argument('--model_name', type=str, default='UC',
                        help='Model name for saving (default: UC)')

    # 边缘增强相关参数
    parser.add_argument('--edge_layer_idx', type=int, default=0,
                        help='Index of the edge extraction layer (default: 0)')
    parser.add_argument('--fusion_levels', nargs='+', type=int, default=[1, 2],
                        help='List of levels to fuse edge features (default: [1, 2])')
    parser.add_argument('--edge_attention', type=str, default='none',
                        choices=['none', 'se', 'cbam'],
                        help='Type of attention used in edge fusion (default: none)')
    parser.add_argument('--fusion_mode', type=str, default='concat',
                        choices=['concat', 'gate', 'dual'],
                        help='Fusion method to use in edge fusion module (default: concat)')
    parser.add_argument('--dataset', type=str, default='单个细胞分类数据集二分类S2L')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--early_stop', type=int, default=10)

    args = parser.parse_args()
    seed_everything(args.seed)

    # 日志与保存
    log_dir, base_save_path = setup_logger_and_saver(args.model_name)
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    logging.info(f"Training parameters saved to: {config_path}")

    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据集路径
    dataset_root = f"/data/{args.dataset}"
    full_dataset = datasets.ImageFolder(root=dataset_root, transform=None)
    logging.info(f"Loaded dataset from: {dataset_root}, total samples: {len(full_dataset)}")

    # ================================
    # 🚀 加载 kfold_splits.pkl（包含 mean/std 和相对路径）
    # ================================
    kfold_pkl_path = os.path.join(dataset_root, "kfold_splits.pkl")
    if not os.path.exists(kfold_pkl_path):
        raise FileNotFoundError(f"❌ 找不到划分文件: {kfold_pkl_path}\n请先运行划分生成脚本。")

    logging.info(f"✅ 加载划分文件: {kfold_pkl_path}")
    with open(kfold_pkl_path, 'rb') as f:
        data = pickle.load(f)

    splits = data['splits']
    fold_data = []

    # 构建完整路径 → 索引 映射
    path_to_idx = {
        os.path.join(dataset_root, img_path).replace("\\", "/").replace("//", "/"): idx
        for idx, (img_path, _) in enumerate(full_dataset.imgs)
    }

    for fold_idx, split in enumerate(splits):
        rel_train_paths = split['train']
        rel_val_paths = split['val']
        mean = split['mean']
        std = split['std']

        # 拼接完整路径
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
        if fold == 1 or fold == 5:
            continue
        print(f"\n========== Fold {fold}/{len(fold_data)} ==========")
        train_idx = data['train_idx']
        val_idx = data['val_idx']
        mean = data['mean']
        std = data['std']
        # mean = 0.5
        # std = 0.5
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

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # 构建模型
        model_class = MODEL_MAP[args.model_type]
        model_kwargs = {}

        if args.model_type == 'edge_enhanced':
            model_kwargs.update({
                'edge_layer_idx': args.edge_layer_idx,
                'fusion_levels': args.fusion_levels,
                'edge_attention': args.edge_attention,
                'fusion_mode': args.fusion_mode,
            })
        elif args.model_type in ['dual_branch', 'dual_branch_enhanced']:
            model_kwargs.update({
                'fusion_levels': args.fusion_levels,
                'edge_attention': args.edge_attention,
                'fusion_mode': args.fusion_mode,
            })

        net = model_class(
            depths=[2, 2, 4, 2],
            dims=[96, 192, 384, 768],
            num_classes=args.num_classes,
            **model_kwargs
        ).to(device)

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_metrics = None
        fold_save_path = base_save_path.replace("_best.pth", f"_fold{fold}_best.pth")
        best_acc = -float('inf')
        epochs_no_improve = 0

        for epoch in range(args.epochs):
            # 训练
            net.train()
            running_loss = 0.0
            train_bar = tqdm(train_loader, file=sys.stdout, desc=f"Epoch [{epoch + 1}/{args.epochs}] Training")
            for step, (images, labels) in enumerate(train_bar):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = net(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                train_bar.set_postfix(loss=f"{loss.item():.3f}")

            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # 验证（多指标）
            metrics = evaluate_with_metrics(net, val_loader, device, args.num_classes)

            log_msg = (f"[Fold {fold}][Epoch {epoch + 1}/{args.epochs}] "
                       f"Loss: {running_loss / len(train_loader):.3f} | "
                       f"Acc: {metrics['acc']:.4f} | "
                       f"Precision: {metrics['precision']:.4f} | "
                       f"Recall: {metrics['recall']:.4f} | "
                       f"F1: {metrics['f1']:.4f} | "
                       f"LR: {current_lr:.2e}")
            print(log_msg)
            logging.info(log_msg)

            # 保存最佳模型（以 Accuracy 为准）
            # [ES] Save best and update patience counter (Accuracy-based)
            improved = metrics['acc'] > best_acc
            if best_metrics is None or improved:
                best_acc = metrics['acc']
                best_metrics = metrics.copy()
                epochs_no_improve = 0
                torch.save(net.state_dict(), fold_save_path)
                logging.info(f"✅ Saved best model (Acc: {metrics['acc']:.4f}) to {fold_save_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.early_stop:
                    logging.info(
                        f"⏹ Early stopping on fold {fold} at epoch {epoch + 1}: "
                    )
                    break

        # 记录本折最佳指标
        fold_results.append(best_metrics)
        print(f"📌 Fold {fold} Best Metrics: {best_metrics}")

    # ================================
    # 汇总结果（包含模型复杂度）
    # ================================
    all_acc = [r['acc'] for r in fold_results]
    all_prec = [r['precision'] for r in fold_results]
    all_rec = [r['recall'] for r in fold_results]
    all_f1 = [r['f1'] for r in fold_results]

    # 临时构建一次模型，用于计算 FLOPs 和 Parameters（只在 CPU/GPU 上临时用）
    try:
        from thop import profile
        model_class = MODEL_MAP[args.model_type]
        model_kwargs = {}

        if args.model_type == 'edge_enhanced':
            model_kwargs.update({
                'edge_layer_idx': args.edge_layer_idx,
                'fusion_levels': args.fusion_levels,
                'edge_attention': args.edge_attention,
                'fusion_mode': args.fusion_mode,
            })
        elif args.model_type in ['dual_branch', 'dual_branch_enhanced']:
            model_kwargs.update({
                'fusion_levels': args.fusion_levels,
                'edge_attention': args.edge_attention,
                'fusion_mode': args.fusion_mode,
            })

        net_for_flops = model_class(
            depths=[2, 2, 4, 2],
            dims=[96, 192, 384, 768],
            num_classes=args.num_classes,
            **model_kwargs
        ).to(device)

        input_tensor = torch.randn(1, 3, 224, 224).to(device)
        flops, params = profile(net_for_flops, inputs=(input_tensor,), verbose=False)
        flops_str = f"{flops / 1e9:.3f}G" if flops > 1e9 else f"{flops / 1e6:.3f}M"
        params_str = f"{params / 1e6:.3f}M" if params > 1e6 else f"{params / 1e3:.3f}K"
    except Exception as e:
        logging.warning(f"⚠️ Could not compute model complexity: {e}")
        flops_str = "N/A"
        params_str = "N/A"

    # 构建 summary
    summary = {
        "Model FLOPs": flops_str,
        "Model Parameters": params_str,
        "Average Accuracy": float(np.mean(all_acc)),
        "Std Accuracy": float(np.std(all_acc)),
        "Average Precision": float(np.mean(all_prec)),
        "Std Precision": float(np.std(all_prec)),
        "Average Recall": float(np.mean(all_rec)),
        "Std Recall": float(np.std(all_rec)),
        "Average F1": float(np.mean(all_f1)),
        "Std F1": float(np.std(all_f1)),
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

    print("\n========== Cross-Validation Results ==========")
    for k, v in summary.items():
        if k == "Per-fold Results":
            continue
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    summary_path = os.path.join(log_dir, "cv_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    logging.info(f"✅ CV Summary saved to: {summary_path}")
    logging.info("✅ Training completed.")


if __name__ == '__main__':
    main()