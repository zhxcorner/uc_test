import os
import json
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# ✅ 模型导入
from MedMamba import VSSM as vssm
from MedMamba import VSSMEdgeEnhanced as edge_enhanced
from MedMamba import DualBranchVSSM as dual_branch
from MedMamba import DualBranchVSSMEnhanced as dual_branch_enhanced

# ==== 配置路径 ====
LOG_DIR = "/root/logs/20250719-071320"
IMAGE_PATH = '/data/单个细胞分类数据集二分类S2L/train/HGUC/01-006-1a-20230321071302829_HGUC_0.jpg'
SAVE_DIR = "./edge_feat_visual"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==== 模型类型映射 ====
MODEL_MAP = {
    'vssm': vssm,
    'edge_enhanced': edge_enhanced,
    'dual_branch': dual_branch,
    'dual_branch_enhanced': dual_branch_enhanced,
}

# ==== 加载 config 和 best.pth ====
config_path = os.path.join(LOG_DIR, "config.json")
model_path = None
for f in os.listdir(LOG_DIR):
    if f.endswith("best.pth"):
        model_path = os.path.join(LOG_DIR, f)
        break
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

# ==== 提取边缘特征并可视化（各层平均图） ====
with torch.no_grad():
    edge_feats = model.edge_generator(img_tensor)  # List of [1, C, H, W]

for lvl_idx, feat in enumerate(edge_feats):
    feat = feat[0]                      # [C, H, W]
    mean_map = feat.mean(dim=0).cpu()  # [H, W]

    norm_map = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min() + 1e-6)
    norm_np = norm_map.numpy()

    plt.imshow(norm_np, cmap='gray')
    plt.axis('off')
    plt.title(f"Edge Mean L{lvl_idx}")
    plt.savefig(os.path.join(SAVE_DIR, f"edge_mean_l{lvl_idx}.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

print(f"✅ Saved edge visualizations to: {SAVE_DIR}")

# ==== 保存未经处理的 Sobel 原始边缘图（Baseline 对比） ====
with torch.no_grad():
    raw_sobel = model.edge_generator.sc(img_tensor)  # [1, C, H, W]

# 通道平均 + Min-Max 归一化
mean_map = raw_sobel[0].mean(dim=0).cpu()  # [H, W]
norm_map = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min() + 1e-6)

# 保存灰度图
plt.imshow(norm_map.numpy(), cmap='gray')
plt.axis('off')
plt.title("Raw Sobel Output (Pre-Pooling/Conv)")
plt.savefig(os.path.join(SAVE_DIR, "sobel_raw_mean.png"), bbox_inches='tight', pad_inches=0)
plt.close()
