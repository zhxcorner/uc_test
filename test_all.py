import os
import json
import torch
from tqdm import tqdm
from torchvision import transforms, datasets

# 动态导入模型
from MedMamba import VSSM as vssm
from MedMamba import VSSMEdgeEnhanced as edge_enhanced
from MedMamba import DualBranchVSSM as dual_branch
from sobel import *
# 模型字典
MODEL_MAP = {
    'vssm': vssm,
    'edge_enhanced': edge_enhanced,
    'dual_branch': dual_branch
}

def evaluate_model(model, data_loader, device, dataset_size):
    model.eval()
    acc = 0.0
    with torch.no_grad():
        test_bar = tqdm(data_loader, desc="Testing", leave=False)
        for images, labels in test_bar:
            outputs = model(images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, labels.to(device)).sum().item()
    return acc / dataset_size


def load_config_and_test_model(log_dir):
    config_path = os.path.join(log_dir, "config.json")
    model_path = None
    for file in os.listdir(log_dir):
        if file.endswith("best.pth"):
            model_path = os.path.join(log_dir, file)
            break

    if not model_path:
        print(f"[{log_dir}] ❌ No best model found.")
        return

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"[{log_dir}] ❌ Failed to load config: {e}")
        return

    # 构建模型
    model_type = config['model_type']
    model_class = MODEL_MAP.get(model_type)
    if not model_class:
        print(f"[{log_dir}] ❌ Unknown model type: {model_type}")
        return

    model_kwargs = {}
    if model_type == 'edge_enhanced':
        model_kwargs.update({
            'edge_layer_idx': config.get('edge_layer_idx'),
            'fusion_levels': config.get('fusion_levels'),
            'edge_attention': config.get('edge_attention'),
            'fusion_mode': config.get('fusion_mode', 'concat')
        })
    elif model_type == 'dual_branch':
        model_kwargs.update({
            'fusion_levels': config.get('fusion_levels'),
            'edge_attention': config.get('edge_attention'),
            'fusion_mode': config.get('fusion_mode', 'concat')
            })

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        model = model_class(num_classes=config['num_classes'], **model_kwargs).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"[{log_dir}] ❌ Failed to load model: {e}")
        return

    # 加载测试集
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = datasets.ImageFolder(root="/data/单个细胞分类数据集二分类S2L/test", transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    test_acc = evaluate_model(model, test_loader, device, len(test_dataset))
    print(f"[{log_dir}] ✅ Test Accuracy: {test_acc:.4f}")
    print(json.dumps(config, indent=2))


def main():
    logs_dir = "logs"
    if not os.path.isdir(logs_dir):
        print(f"❌ Logs directory '{logs_dir}' not found.")
        return

    print(f"🔍 Testing all models under '{logs_dir}/'")
    for folder in os.listdir(logs_dir):
        full_path = os.path.join(logs_dir, folder)
        if os.path.isdir(full_path):
            load_config_and_test_model(full_path)


if __name__ == "__main__":
    main()
