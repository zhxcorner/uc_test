import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import argparse
import json

# === 解析参数 ===
parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations using config from log directory.')
parser.add_argument('--log_dir', type=str, required=True,
                    help='Path to the log directory containing config.json and best.pth')
args = parser.parse_args()

# === Settings ===
log_dir = args.log_dir
config_path = os.path.join(log_dir, 'config.json')
model_path = None
# 查找 best.pth 文件
for file in os.listdir(log_dir):
    if file.endswith('best.pth'):
        model_path = os.path.join(log_dir, file)
        break

if not model_path:
    raise FileNotFoundError(f"No best.pth found in {log_dir}")

if not os.path.exists(config_path):
    raise FileNotFoundError(f"config.json not found in {log_dir}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = '../gradcam_output_multilayer'

# === 动态导入模型类 ===
from MedMamba import (
    VSSM as vssm,
    VSSMEdgeEnhanced as edge_enhanced,
    DualBranchVSSM as dual_branch,
    DualBranchVSSMEnhanced as dual_branch_enhanced
)

MODEL_MAP = {
    'vssm': vssm,
    'edge_enhanced': edge_enhanced,
    'dual_branch': dual_branch,
    'dual_branch_enhanced': dual_branch_enhanced,
}

# === 读取配置并构建模型 ===
with open(config_path, 'r') as f:
    config = json.load(f)

model_type = config.get('model_type')
num_classes = config.get('num_classes')
img_size = config.get('img_size', 224)  # 默认 224

if model_type not in MODEL_MAP:
    raise ValueError(f"Unsupported model_type: {model_type}, must be one of {list(MODEL_MAP.keys())}")

# 构建 kwargs
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

# 实例化模型
model_class = MODEL_MAP[model_type]
model = model_class(num_classes=num_classes, **model_kwargs).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f"✅ Loaded model: {model_type}")
print(f"   Config: num_classes={num_classes}, img_size={img_size}, kwargs={model_kwargs}")

# === 预处理 ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * 3, (0.5,) * 3)
])

# === Sobel 边缘检测 ===
def sobel_edge_detection(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (img_size, img_size))
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge = cv2.magnitude(grad_x, grad_y)
    edge = (edge / edge.max()) * 255
    edge_map = np.uint8(edge)
    return cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)

# === Grad-CAM 多层 Hook ===
features = {}
gradients = {}

def get_activation(name):
    def hook(module, input, output):
        features[name] = output.detach()
    return hook

def get_gradient(name):
    def hook(module, grad_input, grad_output):
        gradients[name] = grad_output[0].detach()
    return hook

# 自动根据模型类型和 fusion_levels 确定目标层
target_layer_names = []
target_layers = []

if hasattr(model, 'fusers') and isinstance(config.get('fusion_levels'), list):
    for level in config['fusion_levels']:
        name = f"fusion{level}"
        layer = model.fusers[level]
        target_layer_names.append(name)
        target_layers.append(layer)
    print(f"🎯 Registered fusion layers: {target_layer_names}")
else:
    raise RuntimeError("Model does not have 'fusers' or 'fusion_levels' not defined in config.")

# 注册钩子
hooks = []
for name, layer in zip(target_layer_names, target_layers):
    hooks.append(layer.register_forward_hook(get_activation(name)))
    hooks.append(layer.register_full_backward_hook(get_gradient(name)))

# === 生成 Grad-CAM ===
def generate_cams(image_tensor, class_idx):
    global features, gradients
    features.clear()
    gradients.clear()

    image_tensor = image_tensor.unsqueeze(0).to(device)
    output = model(image_tensor)
    pred = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, class_idx].backward()

    cam_maps = []
    for name in target_layer_names:
        fmap = features[name].squeeze(0)  # [C, H, W]
        grad = gradients[name].squeeze(0)  # [C, H, W]
        weights = grad.mean(dim=(1, 2))    # [C]
        cam = (weights[:, None, None] * fmap).sum(dim=0)
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        cam_np = cam.cpu().numpy()
        cam_np = cv2.resize(cam_np, (img_size, img_size))
        cam_maps.append(cam_np)

    return cam_maps, pred

# === 叠加热力图 ===
def overlay_cam(image_path, cam_map):
    image = Image.open(image_path).convert('RGB').resize((img_size, img_size))
    img_np = np.array(image)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = 0.5 * img_np + 0.5 * heatmap
    return np.uint8(overlay)

# === 处理图像 ===
val_path = '/data/单个细胞分类数据集二分类S2L/train'  # 可考虑也从 config 读取
os.makedirs(output_dir, exist_ok=True)
class_dirs = sorted(os.listdir(val_path))

for cls_name in class_dirs:
    input_dir = os.path.join(val_path, cls_name)
    if not os.path.isdir(input_dir):
        continue
    save_dir = os.path.join(output_dir, os.path.basename(log_dir), cls_name)  # 按 log_dir 分文件夹
    os.makedirs(save_dir, exist_ok=True)

    class_idx = 0 if cls_name.lower() == 'lguc' else 1
    img_files = sorted(os.listdir(input_dir))[:10]

    for i, img_file in enumerate(img_files):
        img_path = os.path.join(input_dir, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image)

            cam_maps, pred = generate_cams(input_tensor, class_idx)

            # 构建可视化图：原始图 | Sobel 边缘 | 多层 CAM
            base_img = np.array(image.resize((img_size, img_size)))
            sobel_map = sobel_edge_detection(img_path)
            cam_overlays = [overlay_cam(img_path, cam) for cam in cam_maps]

            combined_images = [base_img, sobel_map] + cam_overlays
            combined = np.hstack(combined_images)

            # 保存路径包含 log_dir 名称，避免冲突
            save_path = os.path.join(save_dir, f'gradcam_{i:03d}.jpg')
            cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

            pred_label = 'HGUC' if pred == 0 else 'LGUC'
            print(f"[{cls_name}] Saved: {save_path} | Pred: {pred_label}")
        except Exception as e:
            print(f"[Error] Failed on {img_path}: {e}")

# 清理钩子
for hook in hooks:
    hook.remove()