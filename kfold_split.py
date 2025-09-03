# kfold_generator.py
import os
import pickle
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch


# 自定义 Dataset（用于计算 mean/std）
class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


def compute_mean_std_from_paths(image_paths, batch_size=64, num_workers=4):
    """
    给定图像路径列表，计算 mean 和 std
    """
    if len(image_paths) == 0:
        raise ValueError("No images provided for mean/std calculation")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = ImagePathDataset(image_paths, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    with torch.no_grad():
        for images in loader:
            batch_size = images.size(0)
            mean += images.mean(dim=[0, 2, 3]) * batch_size
            std += images.std(dim=[0, 2, 3]) * batch_size
            total_images += batch_size

    mean /= total_images
    std /= total_images

    return mean.tolist(), std.tolist()


def generate_kfold_path_splits_with_stats(dataset_root, k=5, shuffle=True, random_state=42, save_path='kfold_splits.pkl'):
    """
    生成 K-Fold 划分，并为每折计算训练集的 mean 和 std
    ✅ 所有路径保存为相对于 dataset_root 的格式（如 'LGUC/img1.jpg'）
    """
    print(f"🔍 扫描数据集目录: {dataset_root}")

    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"数据集目录不存在: {dataset_root}")

    image_paths = []
    labels = []
    class_to_idx = {}

    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}

    # 获取所有类别（子文件夹），排序确保顺序固定
    classes = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))])
    if len(classes) == 0:
        raise ValueError(f"在 {dataset_root} 中未找到任何类别文件夹")

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    print(f"📊 找到 {len(classes)} 个类别: {classes}")

    # 遍历每个类别，收集图像路径和标签
    for class_name in classes:
        class_dir = os.path.join(dataset_root, class_name)
        img_count = 0

        for fname in sorted(os.listdir(class_dir)):
            ext = os.path.splitext(fname.lower())[-1]
            if ext in image_extensions:
                abs_path = os.path.join(class_dir, fname)
                if os.path.isfile(abs_path):
                    image_paths.append(abs_path)
                    labels.append(class_name)
                    img_count += 1

        print(f"📁 {class_name}: {img_count} 张图像")

    if len(image_paths) == 0:
        raise ValueError("未找到任何图像文件")

    print(f"✅ 共收集到 {len(image_paths)} 张图像，开始生成 {k}-Fold 划分...")

    # 使用分层 K 折
    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)
    fold_splits = []

    # 生成每一折
    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
        train_abs_paths = [image_paths[i] for i in train_idx]
        val_abs_paths = [image_paths[i] for i in val_idx]

        # 计算该 fold 训练集的 mean 和 std
        print(f"Fold {fold + 1}/{k}: 正在计算训练集 mean/std...")
        mean, std = compute_mean_std_from_paths(train_abs_paths)

        # ✅ 转换为相对路径（关键修改）
        train_rel_paths = [
            os.path.relpath(p, dataset_root).replace("\\", "/") for p in train_abs_paths
        ]
        val_rel_paths = [
            os.path.relpath(p, dataset_root).replace("\\", "/") for p in val_abs_paths
        ]

        fold_splits.append({
            'train': train_rel_paths,  # ✅ 保存相对路径
            'val': val_rel_paths,     # ✅ 保存相对路径
            'mean': mean,
            'std': std
        })

        # 统计类别分布
        train_counter = defaultdict(int)
        val_counter = defaultdict(int)
        for i in train_idx:
            train_counter[labels[i]] += 1
        for i in val_idx:
            val_counter[labels[i]] += 1

        print(f"\n📌 Fold {fold + 1}/{k} 类别分布:")
        print(f"{'类别':<15} {'训练数':<8} {'验证数':<8}")
        print("-" * 35)
        for cls in classes:
            t = train_counter[cls]
            v = val_counter[cls]
            print(f"{cls:<15} {t:<8} {v:<8}")
        print(f"{'总计':<15} {sum(train_counter.values()):<8} {sum(val_counter.values()):<8}")
        print(f"📌 计算得到的归一化参数: mean={mean}, std={std}\n")

    # 保存所有信息
    with open(save_path, 'wb') as f:
        pickle.dump({
            'splits': fold_splits,
            'classes': classes,
            'class_to_idx': class_to_idx,
            'k': k,
            'random_state': random_state,
            'shuffle': shuffle,
            'dataset_root': dataset_root  # 仍保存原始路径信息用于调试
        }, f)

    print(f"\n🎉 K-Fold 划分完成！")
    print(f"📌 划分结果已保存至: {save_path}")
    print(f"📌 ✅ 所有图像路径已保存为相对于 '{dataset_root}' 的格式（如 'LGUC/img.jpg'）")
    print(f"📌 后续训练时，请使用 os.path.join(dataset_root, rel_path) 拼接完整路径")


# ================================
# 使用示例
# ================================
if __name__ == "__main__":
    DATASET_ROOT = "bloodmnist_kfold"  # 修改为你的路径，支持相对或绝对路径
    SAVE_PATH = "bloodmnist_kfold/kfold_splits.pkl"
    K_FOLDS = 5
    RANDOM_STATE = 42

    # 确保保存目录存在
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    generate_kfold_path_splits_with_stats(
        dataset_root=DATASET_ROOT,
        k=K_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
        save_path=SAVE_PATH
    )