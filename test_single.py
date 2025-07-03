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

def evaluate_model(model, data_loader, device, dataset_size, class_names):
    model.eval()
    acc = 0.0
    error_list = []

    with torch.no_grad():
        test_bar = tqdm(data_loader, desc="Testing", leave=False)
        for images, labels in test_bar:
            image_paths = data_loader.dataset.samples[data_loader.batch_size * len(test_bar):][:len(images)]
            outputs = model(images.to(device))
            predict_ys = torch.max(outputs, dim=1)[1]

            for i in range(len(labels)):
                label = labels[i].item()
                pred = predict_ys[i].item()
                if label != pred:
                    rel_path = os.path.relpath(image_paths[i][0], start=data_loader.dataset.root)
                    error_list.append({
                        "image": rel_path,
                        "true_label": class_names[label],
                        "pred_label": class_names[pred]
                    })
            acc += torch.eq(predict_ys, labels.to(device)).sum().item()

    return acc / dataset_size, error_list


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

    # 加载测试数据
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = datasets.ImageFolder(root="/data/某个细胞分类数据集二分类S2L/test", transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    class_names = test_dataset.classes

    test_acc, errors = evaluate_model(model, test_loader, device, len(test_dataset), class_names)

    print(f"[{log_dir}] ✅ Test Accuracy: {test_acc:.4f}")
    print(json.dumps(config, indent=2))

    print("\n❌ Prediction Errors:")
    for err in errors:
        print(f"Image: {err['image']} | True Label: {err['true_label']} | Predicted Label: {err['pred_label']}")

    # 可选：保存错误列表为 JSON 文件
    error_file = os.path.join(log_dir, "prediction_errors.json")
    with open(error_file, 'w') as f:
        json.dump(errors, f, indent=2)
    print(f"\nPrediction errors saved to {error_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test a trained model from log directory.')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Path to the log directory containing config.json and best.pth')

    args = parser.parse_args()

    load_config_and_test_model(args.log_dir)