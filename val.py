# -*- coding: utf-8 -*-
"""
对 MedMamba 系列模型进行 K-Fold 验证
保持与训练代码完全一致的评估方式：
    - accuracy_score
    - precision_recall_fscore_support(average='macro')
不修改任何指标定义
"""
import os
import sys
import json
import pickle
import argparse
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 导入模型（确保 MedMamba.py 在路径中）
from MedMamba import VSSM as vssm
from MedMamba import VSSMEdgeEnhanced as edge_enhanced
from MedMamba import DualBranchVSSM as dual_branch
from MedMamba import DualBranchVSSMEnhanced as dual_branch_enhanced

# 模型映射
MODEL_MAP = {
    'vssm': vssm,
    'edge_enhanced': edge_enhanced,
    'dual_branch': dual_branch,
    'dual_branch_enhanced': dual_branch_enhanced,
}

# ========== Utils ==========
def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_dir):
    log_file = os.path.join(log_dir, "evaluate_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Evaluation log saved to: {log_file}")


@torch.no_grad()
def evaluate_with_metrics(model, loader, device, num_classes):
    """
    使用与训练代码完全一致的方式计算指标：
        - Accuracy: accuracy_score
        - Precision, Recall, F1: average='macro'
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

    # ✅ 完全复现原逻辑
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    return {
        'acc': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="Evaluate MedMamba K-Fold Models (Original Metric Logic)")
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Path to the training log directory (contains best models)')
    parser.add_argument('--dataset', type=str, default='单个细胞分类数据集二分类S2L',
                        help='Dataset name under /data/')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['vssm', 'edge_enhanced', 'dual_branch', 'dual_branch_enhanced'],
                        help='Model type used in training')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    # 模型结构参数（需与训练一致）
    parser.add_argument('--edge_layer_idx', type=int, default=0)
    parser.add_argument('--fusion_levels', nargs='+', type=int, default=[1, 2])
    parser.add_argument('--edge_attention', type=str, default='none', choices=['none', 'se', 'cbam'])
    parser.add_argument('--fusion_mode', type=str, default='concat', choices=['concat', 'gate', 'dual'])

    args = parser.parse_args()
    seed_everything(args.seed)

    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 日志
    setup_logger(args.log_dir)

    # 数据集路径
    dataset_root = f"/data/{args.dataset}"
    full_dataset = datasets.ImageFolder(root=dataset_root, transform=None)
    logging.info(f"Loaded dataset from: {dataset_root}, total samples: {len(full_dataset)}")

    # ================================
    # 加载 kfold_splits.pkl
    # ================================
    kfold_pkl_path = os.path.join(dataset_root, "kfold_splits.pkl")
    if not os.path.exists(kfold_pkl_path):
        raise FileNotFoundError(f"❌ 找不到划分文件: {kfold_pkl_path}")

    logging.info(f"✅ 加载划分文件: {kfold_pkl_path}")
    with open(kfold_pkl_path, 'rb') as f:
        data = pickle.load(f)

    splits = data['splits']
    fold_data = []

    # 构建路径 → 索引 映射
    path_to_idx = {
        os.path.join(dataset_root, img).replace("\\", "/").replace("//", "/"): idx
        for idx, (img, _) in enumerate(full_dataset.imgs)
    }

    for fold_idx, split in enumerate(splits):
        rel_train = split['train']
        rel_val = split['val']
        mean = split['mean']
        std = split['std']

        full_train = [os.path.join(dataset_root, p).replace("\\", "/") for p in rel_train]
        full_val = [os.path.join(dataset_root, p).replace("\\", "/") for p in rel_val]

        train_idx = [path_to_idx[p] for p in full_train if p in path_to_idx]
        val_idx = [path_to_idx[p] for p in full_val if p in path_to_idx]

        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(f"Fold {fold_idx + 1} 数据为空")

        fold_data.append({
            'train_idx': train_idx,
            'val_idx': val_idx,
            'mean': mean,
            'std': std
        })

    logging.info(f"✅ 成功加载 {len(fold_data)} 折划分")

    # ================================
    # 开始验证每折模型
    # ================================
    fold_results = []

    for fold, data in enumerate(fold_data, start=1):
        print(f"\n========== Evaluating Fold {fold} ==========")
        val_idx = data['val_idx']
        mean = data['mean']
        std = data['std']
        mean = 0.5
        std = 0.5
        # 数据变换
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        dataset = datasets.ImageFolder(root=dataset_root, transform=transform)
        val_subset = Subset(dataset, val_idx)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

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

        model = model_class(
            depths=[2, 2, 4, 2],
            dims=[96, 192, 384, 768],
            num_classes=args.num_classes,
            **model_kwargs
        ).to(device)

        # 加载权重
        model_path = os.path.join(args.log_dir, f"UCNet_fold{fold}_best.pth")
        if not os.path.exists(model_path):
            logging.warning(f"⚠️ 模型文件不存在: {model_path}")
            continue

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f"✅ 加载模型权重: {model_path}")

        # 评估
        metrics = evaluate_with_metrics(model, val_loader, device, args.num_classes)
        fold_results.append(metrics)

        # 打印结果
        print(f"📌 Fold {fold} Results:")
        print(f"    Accuracy : {metrics['acc']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall   : {metrics['recall']:.4f}")
        print(f"    F1       : {metrics['f1']:.4f}")

    # ================================
    # 汇总结果
    # ================================
    if not fold_results:
        logging.error("❌ 未评估任何 fold，检查模型路径")
        return

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

    print("\n========== Final Cross-Validation Evaluation ==========")
    for k, v in summary.items():
        if k != "Per-fold Results":
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # 保存结果
    summary_path = os.path.join(args.log_dir, "evaluation_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    logging.info(f"✅ 评估结果已保存至: {summary_path}")


if __name__ == '__main__':
    main()