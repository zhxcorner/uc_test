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
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
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
def compute_mean_std(dataset_root, indices, transform, batch_size=32, num_workers=4):
    temp_dataset = datasets.ImageFolder(root=dataset_root, transform=transform)
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


def evaluate_model(model, data_loader, device, dataset_size):
    model.eval()
    acc = 0.0
    with torch.no_grad():
        test_bar = tqdm(data_loader, file=sys.stdout, desc="Validating")
        for images, labels in test_bar:
            outputs = model(images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, labels.to(device)).sum().item()

    accuracy = acc / dataset_size
    return accuracy


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
    args = parser.parse_args()
    seed_everything()
    valid_params = {k: v for k, v in vars(args).items()}

    log_dir, save_path = setup_logger_and_saver(args.model_name)
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(valid_params, f, indent=4)
    logging.info(f"Training parameters saved to: {config_path}")

    # 数据预处理（临时用于计算 mean/std）
    to_tensor_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 数据预处理（最终用的）
    data_transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
    ])
    data_transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 加载数据集（用于分折）
    all_data = datasets.ImageFolder(root=f"/data/{args.dataset}", transform=None)
    indices = np.arange(len(all_data))  # 修正：原代码中 full_dataset 未定义

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"Fold {fold + 1}/5")

        # --- 计算当前 fold 训练集的 mean 和 std ---
        mean, std = compute_mean_std(
            dataset_root=f"/data/{args.dataset}",
            indices=train_idx,
            transform=to_tensor_transform,
            batch_size=args.batch_size,
            num_workers=4
        )
        logging.info(f"Fold {fold+1} - Computed mean: {mean}, std: {std}")

        # --- 构建每折的 transform ---
        data_transform_train_norm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        data_transform_val_norm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        # 使用 KFold 索引划分训练集和验证集
        train_data = Subset(
            datasets.ImageFolder(root=f"/data/{args.dataset}", transform=data_transform_train_norm),
            train_idx
        )
        val_data = Subset(
            datasets.ImageFolder(root=f"/data/{args.dataset}", transform=data_transform_val_norm),
            val_idx
        )

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

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
        elif args.model_type == 'dual_branch' or args.model_type == 'dual_branch_enhanced':
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
        optimizer = optim.Adam(net.parameters(), lr=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)  # 余弦退火

        best_acc = 0.0
        train_steps = len(train_loader)

        for epoch in range(args.epochs):
            # Train
            net.train()
            running_loss = 0.0
            train_bar = tqdm(train_loader, file=sys.stdout, desc=f"Epoch [{epoch + 1}/{args.epochs}] Training")
            for step, data in enumerate(train_bar):
                images, labels = data
                optimizer.zero_grad()
                outputs = net(images.to(device))
                loss = loss_function(outputs, labels.to(device))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                train_bar.set_postfix(loss=f"{loss.item():.3f}")

            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Validate
            net.eval()
            acc = 0
            total = 0
            with torch.no_grad():
                val_bar = tqdm(val_loader, file=sys.stdout, desc="Validating")
                for val_images, val_labels in val_bar:
                    outputs = net(val_images.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                    total += val_labels.size(0)

            val_accurate = acc / total
            print(f"[Epoch {epoch + 1}] Train Loss: {running_loss / train_steps:.3f} | "
                  f"Val Acc: {val_accurate:.4f} | LR: {current_lr:.2e}")
            logging.info('[epoch %d] train_loss: %.3f  val_accuracy: %.4f  lr: %.2e',
                         epoch + 1, running_loss / train_steps, val_accurate, current_lr)

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path.replace('best', f'best_{fold + 1}'))
                print(f"Saved new best model with val accuracy: {best_acc:.4f}")
                logging.info(f"Saved new best model with val accuracy: {best_acc:.4f}")

        fold_results.append(best_acc)

    # 计算五折交叉验证的平均准确率和方差
    average_accuracy = sum(fold_results) / len(fold_results)
    variance = sum((x - average_accuracy) ** 2 for x in fold_results) / len(fold_results)
    print(f"Results: {fold_results}")
    print(f"Average accuracy across all folds: {average_accuracy:.4f}")
    print(f"Variance of accuracy across all folds: {variance:.4f}")
    logging.info(f"Average accuracy across all folds: {average_accuracy:.4f}")
    logging.info(f"Variance of accuracy across all folds: {variance:.4f}")

    logging.info('Finished Training')
    print('Finished Training')


if __name__ == '__main__':
    main()
