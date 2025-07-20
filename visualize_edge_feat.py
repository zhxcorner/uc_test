import os
import json
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np

# ✅ 导入模型结构
from MedMamba import VSSM as vssm
from MedMamba import VSSMEdgeEnhanced as edge_enhanced
from MedMamba import DualBranchVSSM as dual_branch
from MedMamba import DualBranchVSSMEnhanced as dual_branch_enhanced

# ==== 配置路径 ====
LOG_DIR = "/root/logs/20250719-071320"
IMAGE_PATH = '/data/单个细胞分类数据集二分类S2L/train/HGUC/01-006-1a-20230321071302829_HGUC_0.jpg'
BASE_SAVE_DIR = "./edge_feat_visual"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# ==== 所有可视化模式 ====
VIS_MODES = ['mean', 'abs_mean', 'max', 'l2']  # ✅ 一次性遍历所有模式

# ==== 模型类型映射 ====
MODEL_MAP = {
    'vssm': vssm,
    'edge_enhanced': edge_enhanced,
    'dual_branch': dual_branch,
    'dual_branch_enhanced': dual_branch_enhanced,
}

# ==== 加载 config 和 best.pth ====
config_path = os.path.join(LOG_DIR, "config.json")
model_path = next((os.path.join(LOG_DIR, f) for f in os.listdir(LOG_DIR) if f.endswith("best.pth")), None)
if not model_path:
    raise FileNotFoundError(f"No best.pth found in {LOG_DIR}")

with open(config_path, 'r') as f:
    config = json.load(f)

model_type = config['model_type']
model_class = MODEL_MAP.get(model_type)
if model_class is None:
    raise ValueError(f"Unknown model type: {model_type}")

# ==== 构造模型结构参数 ====
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model_class(num_classes=config['num_classes'], **model_kwargs).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ==== 图像加载与预处理 ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
img = Image.open(IMAGE_PATH).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]

# ==== 特征处理函数 ====
def process_feature(feat: torch.Tensor, mode: str) -> torch.Tensor:
    """将 [C, H, W] 的特征图按指定模式压缩为 [H, W]"""
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

# ==== 提取并保存每种可视化模式的图像 ====
with torch.no_grad():
    edge_feats = model.edge_generator(img_tensor)  # List of [1, C, H, W]
    raw_sobel = model.edge_generator.sc(img_tensor)  # [1, C, H, W]

for mode in VIS_MODES:
    print(f"🔍 Visualizing mode: {mode}")
    SAVE_DIR = os.path.join(BASE_SAVE_DIR, mode)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 每一层边缘特征
    for lvl_idx, feat in enumerate(edge_feats):
        feat = feat[0].cpu()  # [C, H, W]
        vis_map = process_feature(feat, mode)
        norm_map = (vis_map - vis_map.min()) / (vis_map.max() - vis_map.min() + 1e-6)
        norm_map = norm_map ** 0.4  # gamma增强

        plt.imshow(norm_map.numpy(), cmap='gray')
        plt.axis('off')
        plt.title(f"L{lvl_idx} ({mode})")
        plt.savefig(os.path.join(SAVE_DIR, f"edge_{mode}_l{lvl_idx}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()

    # 原始 sobel 图
    sobel_feat = raw_sobel[0].cpu()  # [C, H, W]
    sobel_map = process_feature(sobel_feat, mode)
    sobel_map = (sobel_map - sobel_map.min()) / (sobel_map.max() - sobel_map.min() + 1e-6)
    sobel_map = sobel_map ** 0.4

    plt.imshow(sobel_map.numpy(), cmap='gray')
    plt.axis('off')
    plt.title(f"Sobel ({mode})")
    plt.savefig(os.path.join(SAVE_DIR, f"sobel_raw_{mode}.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

print("✅ All visualizations saved to:", BASE_SAVE_DIR)
