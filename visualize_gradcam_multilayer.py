import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import argparse

# 解析参数
parser = argparse.ArgumentParser(description='Test a trained model from log directory.')
parser.add_argument('--log_dir', type=str, required=True,
                    help='Path to the log directory containing config.json and best.pth')
args = parser.parse_args()

# === Settings ===
model_path = os.path.join(args.log_dir, 'UCNet_best.pth')
val_path = '/data/单个细胞分类数据集二分类S2L/train'
output_dir = '../gradcam_output_multilayer'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 224
num_classes = 2

# === Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])


def sobel_edge_detection(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (img_size, img_size))

    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge = cv2.magnitude(grad_x, grad_y)
    edge = (edge / edge.max()) * 255
    edge_map = np.uint8(edge)

    # 转换为三通道便于拼接
    edge_map = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)
    return edge_map

# === Load Model ===
# from MedMamba import DualBranchVSSM
from MedMamba import DualBranchVSSMEnhanced as DualBranchVSSM
model = DualBranchVSSM(
    num_classes=num_classes,
    fusion_levels=[0, 1, 2],
    fusion_mode='gate',
    edge_attention='none'
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Hook Setup for Multiple Layers ===
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

# Layer names and modules
# target_layer_names = ["edge_conv0", "edge_conv1", "edge_conv2"]
# target_layers = [
#     model.edge_convs[0].conv,
#     model.edge_convs[1].conv,
#     model.edge_convs[2].conv,
# ]
target_layer_names = ["fusion0", "fusion1", "fusion2"]
target_layers = [
    model.fusers[0],
    model.fusers[1],
    model.fusers[2],
]

# Register hooks
hooks = []
for name, layer in zip(target_layer_names, target_layers):
    hooks.append(layer.register_forward_hook(get_activation(name)))
    hooks.append(layer.register_full_backward_hook(get_gradient(name)))

# === Helper to apply Grad-CAM ===
def generate_cams(image_tensor, class_idx):
    global features, gradients
    features = {}
    gradients = {}

    image_tensor = image_tensor.unsqueeze(0).to(device)
    output = model(image_tensor)
    pred = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, class_idx].backward()

    cam_maps = []
    for name in target_layer_names:
        fmap = features[name].squeeze(0)     # shape: [C, H, W]
        grad = gradients[name].squeeze(0)    # shape: [C, H, W]

        weights = grad.mean(dim=(1, 2))      # shape: [C]
        cam = (weights[:, None, None] * fmap).sum(dim=0)
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        cam_np = cam.cpu().numpy()
        cam_np = cv2.resize(cam_np, (img_size, img_size))
        cam_maps.append(cam_np)

    return cam_maps, pred

# === Overlay CAM on image ===
def overlay_cam(image_path, cam_map):
    image = Image.open(image_path).convert('RGB').resize((img_size, img_size))
    img_np = np.array(image)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = 0.5 * img_np + 0.5 * heatmap
    return np.uint8(overlay)

# === Process all images (only first 10 per class) ===
os.makedirs(output_dir, exist_ok=True)
class_dirs = sorted(os.listdir(val_path))

for cls_name in class_dirs:
    input_dir = os.path.join(val_path, cls_name)
    save_dir = os.path.join(output_dir, cls_name)
    os.makedirs(save_dir, exist_ok=True)

    class_idx = 0 if cls_name.lower() == 'lguc' else 1
    img_files = sorted(os.listdir(input_dir))[:10]  # Only take first 10 images

    for i, img_file in enumerate(img_files):
        img_path = os.path.join(input_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image)

        # 在 for 循环内找到下面这段代码：
        cam_maps, pred = generate_cams(input_tensor, class_idx)

        # === 新增内容开始 ===
        # 获取原始图像和 Sobel 边缘图
        base_img = Image.open(img_path).convert('RGB').resize((img_size, img_size))
        base_np = np.array(base_img)

        # 生成 Sobel 边缘图
        sobel_map = sobel_edge_detection(img_path)

        # 生成 CAM 的 overlay 图像
        cam_overlays = [overlay_cam(img_path, cam) for cam in cam_maps]

        # 插入 Sobel 图像到第1个位置（索引为1）
        combined_images = [base_np, sobel_map] + cam_overlays
        combined = np.hstack(combined_images)
        # === 新增内容结束 ===

        save_path = os.path.join(save_dir, f'cam_{cls_name}_{i:03d}.jpg')
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        print(f"[{cls_name}] Saved: {save_path} | Predicted: {'LGUC' if pred == 0 else 'HGUC'}")