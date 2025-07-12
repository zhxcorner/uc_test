import os
import numpy as np
from PIL import Image
import cv2

def sobel_binary_map(pil_img, threshold=50, size=(224, 224)):
    img = np.array(pil_img.convert("L").resize(size))  # Grayscale
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edge = cv2.magnitude(grad_x, grad_y)

    edge = (edge / edge.max()) * 255
    binary = (edge > threshold).astype(np.float32)
    return binary  # shape: (H, W)

def precompute_edges(image_root, save_root, threshold=50):
    os.makedirs(save_root, exist_ok=True)
    class_names = sorted(os.listdir(image_root))

    for cls in class_names:
        input_dir = os.path.join(image_root, cls)
        output_dir = os.path.join(save_root, cls)
        os.makedirs(output_dir, exist_ok=True)

        for fname in sorted(os.listdir(input_dir)):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(input_dir, fname)
            save_path = os.path.join(output_dir, fname.replace('.jpg', '.npy').replace('.png', '.npy'))

            image = Image.open(img_path).convert("RGB")
            edge_map = sobel_binary_map(image, threshold=threshold)

            np.save(save_path, edge_map.astype(np.float32))
            print(f"Saved edge map: {save_path}")

# Example usage:
image_dir = r"C:\Users\zhx00\Desktop\MedMamba\单个细胞分类数据集二分类S2L\train"
edge_save_dir = r"C:\Users\zhx00\Desktop\MedMamba\单个细胞分类数据集二分类S2L\edge_labels"
precompute_edges(image_dir, edge_save_dir)
