import os
import json
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np

# === 导入模型类 ===
from MedMamba import (
    VSSM as vssm,
    VSSMEdgeEnhanced as edge_enhanced,
    DualBranchVSSM as dual_branch,
    DualBranchVSSMEnhanced as dual_branch_enhanced
)

import argparse

# === 解析命令行参数 ===
parser = argparse.ArgumentParser(description='Visualize edge features from edge_generator using config from log directory.')
parser.add_argument('--log_dir', type=str, required=True,
                    help='Path to the log directory containing config.json and best.pth')
parser.add_argument('--image_path', type=str, default='/data/单个细胞分类数据集二分类S2L/train/HGUC/01-006-1a-20230321071302829_HGUC_0.jpg',
                    help='Path to the input image for visualization')
args = parser.parse_args()

LOG_DIR = args.log_dir
IMAGE_PATH = args.image_path
BASE_SAVE_DIR = "./edge_feat_visual"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# === 可视化模式 ===
# VIS_MODES = ['mean', 'abs_mean', 'max', 'l2']
VIS_MODES = ['mean']

# === 模型映射表 ===
MODEL_MAP = {
    'vssm': vssm,
    'edge_enhanced': edge_enhanced,
    'dual_branch': dual_branch,
    'dual_branch_enhanced': dual_branch_enhanced,
}

# === 1. 加载配置文件 ===
config_path = os.path.join(LOG_DIR, "config.json")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")

with open(config_path, 'r') as f:
    config = json.load(f)

# 查找 best.pth
model_path = None
for f in os.listdir(LOG_DIR):
    if f.endswith("best.pth"):
        model_path = os.path.join(LOG_DIR, f)
        break
if not model_path:
    raise FileNotFoundError(f"No best.pth found in {LOG_DIR}")

print(f"✅ Loading config and model from:\n   {LOG_DIR}")

# === 2. 解析模型类型和参数 ===
model_type = config.get('model_type')
if model_type not in MODEL_MAP:
    raise ValueError(f"Unsupported model_type: {model_type}. Must be one of {list(MODEL_MAP.keys())}")

num_classes = config.get('num_classes')
if num_classes is None:
    raise ValueError("config.json must contain 'num_classes'")

# 构建模型参数 kwargs
model_kwargs = {}
if model_type == 'edge_enhanced':
    model_kwargs.update({
        'edge_layer_idx': config.get('edge_layer_idx'),
        'fusion_levels': config.get('fusion_levels'),
        'edge_attention': config.get('edge_attention'),
        'fusion_mode': config.get('fusion_mode', 'concat')
    })
elif model_type in ['dual_branch', 'dual_branch_enhanced']:
    model_kwargs.update({
        'fusion_levels': config.get('fusion_levels'),
        'edge_attention': config.get('edge_attention'),
        'fusion_mode': config.get('fusion_mode', 'concat')
    })

# === 3. 构建并加载模型 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_class = MODEL_MAP[model_type]

model = model_class(num_classes=num_classes, **model_kwargs).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f"✅ Model loaded: {model_type} (num_classes={num_classes}, kwargs={model_kwargs})")

# === 4. 图像预处理 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

try:
    img = Image.open(IMAGE_PATH).convert('RGB')
except Exception as e:
    raise FileNotFoundError(f"Cannot open image: {IMAGE_PATH}, error: {e}")

img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

# === 5. 特征处理函数 ===
def process_feature(feat: torch.Tensor, mode: str) -> torch.Tensor:
    """
    将 [C, H, W] 的特征图按指定模式压缩为 [H, W]
    """
    if mode == 'mean':
        return feat.mean(dim=0)
    elif mode == 'abs_mean':
        return feat.abs().mean(dim=0)
    elif mode == 'max':
        return feat.max(dim=0)[0]
    elif mode == 'l2':
        return torch.sqrt((feat ** 2).sum(dim=0))
    else:
        raise ValueError(f"Unsupported VIS_MODE: {mode}")

# === 6. 提取边缘特征 ===
with torch.no_grad():
    if hasattr(model, 'edge_generator'):
        edge_feats = model.edge_generator(img_tensor)  # List[Tensor] or Tensor
        if isinstance(edge_feats, torch.Tensor):
            edge_feats = [edge_feats]  # 统一为 list
        raw_sobel = model.edge_generator.sc(img_tensor)  # Sobel 源头
    else:
        raise AttributeError(f"Model {model_type} does not have 'edge_generator' attribute.")

# === 7. 为每种可视化模式生成图像 ===
for mode in VIS_MODES:
    print(f"🖼️ Generating visualizations for mode: '{mode}'")
    SAVE_DIR = os.path.join(BASE_SAVE_DIR, os.path.basename(LOG_DIR), mode)  # 按 log_dir 分开保存
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 可视化每一层的边缘特征
    for lvl_idx, feat in enumerate(edge_feats):
        feat = feat[0].cpu()  # [C, H, W]
        vis_map = process_feature(feat, mode)
        # 归一化到 [0,1]
        norm_map = (vis_map - vis_map.min()) / (vis_map.max() - vis_map.min() + 1e-8)
        # Gamma 校正增强对比度
        norm_map = torch.pow(norm_map, 0.4)

        plt.figure(figsize=(4, 4))
        plt.imshow(norm_map.numpy(), cmap='gray')
        plt.axis('off')
        plt.title(f"L{lvl_idx} ({mode})", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"edge_l{lvl_idx}_{mode}.png"),
                    bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()

    # 可视化原始 Sobel 边缘图
    sobel_feat = raw_sobel[0].cpu()  # [C, H, W]
    sobel_map = process_feature(sobel_feat, mode)
    sobel_map = (sobel_map - sobel_map.min()) / (sobel_map.max() - sobel_map.min() + 1e-8)
    sobel_map = torch.pow(sobel_map, 0.4)

    plt.figure(figsize=(4, 4))
    plt.imshow(sobel_map.numpy(), cmap='gray')
    plt.axis('off')
    plt.title(f"Sobel ({mode})", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"sobel_raw_{mode}.png"),
                bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()

print(f"✅ All edge feature visualizations saved to:\n   {os.path.join(BASE_SAVE_DIR, os.path.basename(LOG_DIR))}")