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

def setup_logger_and_saver(model_name="UC"):
    current_time = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", current_time)
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

    args = parser.parse_args()

    valid_params = {k: v for k, v in vars(args).items()}

    log_dir, save_path = setup_logger_and_saver(args.model_name)
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(valid_params, f, indent=4)
    logging.info(f"Training parameters saved to: {config_path}")

    # 数据预处理
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    train_dataset = datasets.ImageFolder(root="/data/单个细胞分类数据集二分类S2L/train",
                                         transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root="/data/单个细胞分类数据集二分类S2L/val",
                                       transform=data_transform["val"])

    train_num = len(train_dataset)
    val_num = len(val_dataset)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

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

    net = model_class(num_classes=args.num_classes, **model_kwargs).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    best_acc = 0.0
    epochs = 100
    train_steps = len(train_loader)

    for epoch in range(epochs):
        # Train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout, desc=f"Epoch [{epoch + 1}/{epochs}] Training")
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.3f}")

        # Validate
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout, desc="Validating")
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print(f"[Epoch {epoch + 1}] Train Loss: {running_loss / train_steps:.3f} | Val Acc: {val_accurate:.3f}")
        logging.info('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                     (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            print(f"Saved new best model with val accuracy: {best_acc:.4f}")
            logging.info(f"Saved new best model with val accuracy: {best_acc:.4f}")

    logging.info('Finished Training')
    print('Finished Training')


if __name__ == '__main__':
    main()
