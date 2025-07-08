import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === 设置 ===
model_path = 'UCNet_best.pth'
val_path = '/data/单个细胞分类数据集二分类S2L/val'
output_dir = './multi_layer_gradcam_output'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 224
num_classes = 2

# === 数据预处理 ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

# === 加载模型 ===
from MedMamba import DualBranchVSSM
model = DualBranchVSSM(
    num_classes=num_classes,
    fusion_levels=[0, 1, 2],
    fusion_mode='gate',
    edge_attention='none'
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Hook 管理器类 ===
class FeatureHookManager:
    def __init__(self):
        self.hooks = []
        self.features = {}
        self.gradients = {}

    def register_hook(self, name, module):
        def forward_hook(module, input, output):
            self.features[name] = output.detach().cpu()

        def backward_hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0].detach().cpu()

        hook_forward = module.register_forward_hook(forward_hook)
        hook_backward = module.register_full_backward_hook(backward_hook)
        self.hooks.append(hook_forward)
        self.hooks.append(hook_backward)

    def clear(self):
        self.features.clear()
        self.gradients.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

# === 定义要 hook 的多个目标层 ===
target_layers = {
    # 边缘分支
    "edge_conv_level0": model.edge_convs[0].conv,
    "edge_conv_level1": model.edge_convs[1].conv,
    "edge_conv_level2": model.edge_convs[2].conv,
}

hook_manager = FeatureHookManager()
for name, layer in target_layers.items():
    hook_manager.register_hook(name, layer)

# === Grad-CAM 生成函数 ===
def generate_cams(image_tensor, class_idx, hook_manager):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    hook_manager.clear()

    output = model(image_tensor)
    pred = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, class_idx].backward()

    cams = {}
    for name in hook_manager.features:
        features = hook_manager.features[name]
        gradients = hook_manager.gradients[name]

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * features).sum(dim=1)
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = F.interpolate(cam.unsqueeze(0), size=(img_size, img_size), mode='bilinear', align_corners=False)
        cams[name] = cam.squeeze().cpu().numpy()

    return cams, pred

# === 图像叠加 CAM 函数 ===
def overlay_cam(image_path, cam_map):
    image = Image.open(image_path).convert('RGB').resize((img_size, img_size))
    img_np = np.array(image)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = 0.5 * img_np + 0.5 * heatmap
    return np.uint8(overlay)

# === 处理验证集中的所有图像 ===
os.makedirs(output_dir, exist_ok=True)
class_dirs = sorted(os.listdir(val_path))

for cls_name in class_dirs:
    input_dir = os.path.join(val_path, cls_name)
    save_dir = os.path.join(output_dir, cls_name)
    os.makedirs(save_dir, exist_ok=True)

    class_idx = 0 if cls_name.lower() == 'lguc' else 1
    img_files = sorted(os.listdir(input_dir))[:10]

    for i, img_file in enumerate(img_files):
        img_path = os.path.join(input_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image)

        cams, pred = generate_cams(input_tensor, class_idx, hook_manager)
        print(f"[{cls_name}] Predicted: {['LGUC', 'SUSC'][pred]}")

        # 可视化和保存每个 CAM
        fig, axes = plt.subplots(1, len(cams) + 1, figsize=(5 * (len(cams) + 1), 5))
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis('off')

        for idx, (name, cam) in enumerate(cams.items()):
            overlay = overlay_cam(img_path, cam)
            axes[idx+1].imshow(overlay)
            axes[idx+1].set_title(f"{name}")
            axes[idx+1].axis('off')

            # 保存为文件
            save_path = os.path.join(save_dir, f'cam_{name}_{i:03d}.jpg')
            cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'compare_{i:03d}.jpg'), bbox_inches='tight')
        plt.close()

        print(f"[{cls_name}] Saved: {save_dir} | Predicted: {['LGUC', 'SUSC'][pred]}")