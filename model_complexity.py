# model_complexity.py
import argparse
import torch
import torchvision.models as models
from thop import profile

def build_model(model_name: str, num_classes: int = 2):
    """
    与 train.py 中完全一致的模型构建函数（便于复用）
    """
    model_map = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        'convnext_tiny': models.convnext_tiny,
        'convnext_small': models.convnext_small,
        'convnext_base': models.convnext_base,
        'convnext_large': models.convnext_large,
        'vit_b_16': models.vit_b_16,
        'vit_b_32': models.vit_b_32,
        'vit_l_16': models.vit_l_16,
        'vit_l_32': models.vit_l_32,
        'swin_t': models.swin_t,
        'swin_s': models.swin_s,
        'swin_b': models.swin_b,
        'swin_v2_t': models.swin_v2_t,
        'swin_v2_s': models.swin_v2_s,
        'swin_v2_b': models.swin_v2_b,
    }

    if model_name not in model_map:
        raise ValueError(f"Unsupported model: {model_name}")

    model = model_map[model_name](weights=None)

    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, torch.nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
        else:
            in_features = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_features, num_classes)
    elif hasattr(model, 'heads'):
        in_features = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(in_features, num_classes)
    else:
        raise NotImplementedError(f"Head replacement not implemented for {model_name}")

    return model


def main():
    parser = argparse.ArgumentParser(description="计算模型参数量和 FLOPs")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=[
                            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                            'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large',
                            'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32',
                            'swin_t', 'swin_s', 'swin_b',
                            'swin_v2_t', 'swin_v2_s', 'swin_v2_b',
                        ],
                        help="模型名称")
    parser.add_argument("--num_classes", type=int, default=2, help="分类头输出维度")
    parser.add_argument("--input_size", type=int, default=224, help="输入图像尺寸")

    args = parser.parse_args()

    # 构建模型
    model = build_model(args.model_name, args.num_classes)
    model.eval()  # 确保在 eval 模式下计算

    # 创建虚拟输入
    input_tensor = torch.randn(1, 3, args.input_size, args.input_size)

    # 计算 FLOPs 和 Params
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)

    # 格式化输出
    flops_str = f"{flops / 1e9:.3f}G" if flops > 1e9 else f"{flops / 1e6:.3f}M"
    params_str = f"{params / 1e6:.3f}M" if params > 1e6 else f"{params / 1e3:.3f}K"

    print("="*50)
    print(f"📊 Model: {args.model_name}")
    print(f"   Input Size: {args.input_size}x{args.input_size}")
    print(f"   Parameters: {params_str}")
    print(f"   FLOPs: {flops_str}")
    print("="*50)

    # 保存到文件（可选）
    result = {
        "model_name": args.model_name,
        "input_size": args.input_size,
        "parameters": params_str,
        "flops": flops_str,
        "parameters_raw": int(params),
        "flops_raw": int(flops)
    }

    output_file = f"{args.model_name}_complexity.json"
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"💾 结果已保存到: {output_file}")


if __name__ == "__main__":
    main()