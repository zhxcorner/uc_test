import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from PIL import Image
from torchvision import transforms

# === Settings ===
model_path = os.path.join(args.log_dir, 'UCNet_best.pth')
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
from MedMamba import DualBranchVSSM
model = DualBranchVSSM(
    num_classes=num_classes,
    fusion_levels=[0, 1, 2],
    fusion_mode='gate',
    edge_attention='none'
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Hook Setup for Multiple Layers ===
features_list = []
gradients_list = []

def forward_hook(module, input, output):
    features_list.append(output.detach())

def backward_hook(module, grad_input, grad_output):
    gradients_list.append(grad_output[0].detach())

# Register hooks on multiple layers of the left branch
target_layers = [
    model.edge_convs[0].conv,
    model.edge_convs[1].conv,
    model.edge_convs[2].conv,
]

for layer in target_layers:
    layer.register_forward_hook(forward_hook)
    layer.register_full_backward_hook(backward_hook)

# === Helper to apply Grad-CAM ===
def generate_cams(image_tensor, class_idx):
    features_list.clear()
    gradients_list.clear()

    image_tensor = image_tensor.unsqueeze(0).to(device)
    output = model(image_tensor)
    pred = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, class_idx].backward()

    cam_maps = []
    for i in range(len(target_layers)):
        fmap = features_list[i].squeeze(0)
        grads = gradients_list[i].squeeze(0)
        weights = grads.mean(dim=(1, 2))
        cam = torch.sum(weights[:, None, None] * fmap, dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
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

        cam_maps, pred = generate_cams(input_tensor, class_idx)

        # Generate overlays
        base_img = Image.open(img_path).convert('RGB').resize((img_size, img_size))
        base_np = np.array(base_img)
        cam_overlays = [overlay_cam(img_path, cam) for cam in cam_maps]

        # Concatenate images horizontally: original | cam1 | cam2 | cam3
        combined = np.hstack([base_np] + cam_overlays)

        save_path = os.path.join(save_dir, f'cam_{cls_name}_{i:03d}.jpg')
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        print(f"[{cls_name}] Saved: {save_path} | Predicted: {'LGUC' if pred == 0 else 'HGD'}")