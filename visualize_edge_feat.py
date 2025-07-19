import torch
import matplotlib.pyplot as plt
import os
import torchvision.transforms as T

# 假设模型和 MultiScaleEdgeInfoGenerator_422 已定义并可用
from MedMamba import DualBranchVSSMEnhanced

# 可选：设置图像保存目录
SAVE_DIR = "/data/单个细胞分类数据集二分类S2L/train/HGUC/01-006-1a-20230321071302829_HGUC_0.jpg"
os.makedirs(SAVE_DIR, exist_ok=True)

# 创建模型并加载权重（如果有）
model = DualBranchVSSMEnhanced()
model.eval()

# 准备一个示例输入
# 假设你有一个 RGB 图像 tensor, 形状为 [1, 3, H, W]，数值范围为 [0, 1]
# 示例加载一张图像：
from PIL import Image
img = Image.open('your_image.jpg').convert('RGB')
transform = T.Compose([
    T.Resize((224, 224)),        # 你模型输入的尺寸
    T.ToTensor()
])
img_tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]

# 前向边缘分支
with torch.no_grad():
    edge_feats = model.edge_generator(img_tensor)  # List of [B, C, H, W]

# 遍历每个融合层级的边缘特征
for lvl_idx, ef in enumerate(edge_feats):
    B, C, H, W = ef.shape
    for b in range(B):
        # 可视化前几个通道（比如最多显示前 4 个）
        for ch in range(min(C, 4)):
            feat_map = ef[b, ch].cpu().numpy()
            plt.imshow(feat_map, cmap='gray')
            plt.axis('off')
            plt.title(f"EdgeFeat L{lvl_idx} B{b} C{ch}")
            fname = f"edge_feat_l{lvl_idx}_b{b}_c{ch}.png"
            plt.savefig(os.path.join(SAVE_DIR, fname), bbox_inches='tight')
            plt.close()
