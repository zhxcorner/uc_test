import torch
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from MedMamba import DualBranchVSSM

# === Settings ===
model_path = '/MedMamba/logs/20250618-100001/UCNet_best.pth'
val_path = '/data/单个细胞分类数据集二分类S2L/val'
output_dir = './gradcam_output'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 224
num_classes = 2

# === Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

# === Load Model ===
model = DualBranchVSSM(
    num_classes=num_classes,
    fusion_levels=[0, 1, 2],
    fusion_mode='gate',
    edge_attention='none'  # or 'none'
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

for name, module in model.named_modules():
    print(f"{name} -> {module}")
# === Hook Setup ===
features = []
gradients = []

def forward_hook(module, input, output):
    features.append(output.detach())

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0].detach())

# === Register hook on last conv layer of left branch ===
target_layer = model.layers[-1].blocks[-1].conv33conv33conv11[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# === Helper to apply Grad-CAM ===
def generate_cam(image_tensor, class_idx):
    features.clear()
    gradients.clear()

    image_tensor = image_tensor.unsqueeze(0).to(device)
    output = model(image_tensor)
    pred = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, class_idx].backward()

    fmap = features[0].squeeze(0)
    grads = gradients[0].squeeze(0)
    weights = grads.mean(dim=(1, 2))
    cam = torch.sum(weights[:, None, None] * fmap, dim=0)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam_np = cam.cpu().numpy()
    cam_np = cv2.resize(cam_np, (img_size, img_size))
    return cam_np, pred

# === Overlay CAM on image ===
def overlay_cam(image_path, cam_map):
    image = Image.open(image_path).convert('RGB').resize((img_size, img_size))
    img_np = np.array(image)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = 0.5 * img_np + 0.5 * heatmap
    return np.uint8(overlay)

# === Process all images ===
os.makedirs(output_dir, exist_ok=True)
class_dirs = sorted(os.listdir(val_path))

for cls_name in class_dirs:
    input_dir = os.path.join(val_path, cls_name)
    save_dir = os.path.join(output_dir, cls_name)
    os.makedirs(save_dir, exist_ok=True)

    class_idx = 0 if cls_name.lower() == 'lguc' else 1
    img_files = sorted(os.listdir(input_dir))

    for i, img_file in enumerate(img_files):
        img_path = os.path.join(input_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image)

        cam, pred = generate_cam(input_tensor, class_idx)
        overlay = overlay_cam(img_path, cam)

        save_path = os.path.join(save_dir, f'cam_{cls_name}_{i:03d}.jpg')
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        print(f"[{cls_name}] Saved: {save_path} | Predicted: {class_dirs[pred]}")
