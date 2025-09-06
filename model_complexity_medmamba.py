# model_complexity_medmamba.py
import argparse
import torch
import sys
import os
import json

# 假设 MedMamba 在当前目录或 PYTHONPATH 中
from MedMamba import VSSM as vssm
from MedMamba import VSSMEdgeEnhanced as edge_enhanced
from MedMamba import DualBranchVSSM as dual_branch
from MedMamba import DualBranchVSSMEnhanced as dual_branch_enhanced

# 模型字典
MODEL_MAP = {
    'vssm': vssm,
    'edge_enhanced': edge_enhanced,
    'dual_branch': dual_branch,
    'dual_branch_enhanced': dual_branch_enhanced,
}

# 尺寸配置映射
SIZE_CONFIG = {
    't': {'depths': [2, 2, 4, 2], 'dims': [96, 192, 384, 768]},
    's': {'depths': [2, 2, 8, 2], 'dims': [96, 192, 384, 768]},
    'b': {'depths': [2, 2, 12, 2], 'dims': [128, 256, 512, 1024]},
}

def build_model(model_type, size, num_classes=2, **kwargs):
    model_class = MODEL_MAP[model_type]
    size_config = SIZE_CONFIG[size]

    model_kwargs = {}
    if model_type == 'edge_enhanced':
        model_kwargs.update({
            'edge_layer_idx': kwargs.get('edge_layer_idx', 0),
            'fusion_levels': kwargs.get('fusion_levels', [1, 2]),
            'edge_attention': kwargs.get('edge_attention', 'none'),
            'fusion_mode': kwargs.get('fusion_mode', 'concat'),
        })
    elif model_type in ['dual_branch', 'dual_branch_enhanced']:
        model_kwargs.update({
            'fusion_levels': kwargs.get('fusion_levels', [1, 2]),
            'edge_attention': kwargs.get('edge_attention', 'none'),
            'fusion_mode': kwargs.get('fusion_mode', 'concat'),
        })

    model = model_class(
        depths=size_config['depths'],
        dims=size_config['dims'],
        num_classes=num_classes,
        **model_kwargs
    )
    return model

def main():
    parser = argparse.ArgumentParser(description="计算 MedMamba 模型参数量和 FLOPs")
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['vssm', 'edge_enhanced', 'dual_branch', 'dual_branch_enhanced'],
                        help='模型类型')
    parser.add_argument('--size', type=str, required=True, choices=['t', 's', 'b'],
                        help='模型尺寸: t(tiny), s(small), b(base)')
    parser.add_argument('--num_classes', type=int, default=2, help='分类数')
    parser.add_argument('--input_size', type=int, default=224, help='输入尺寸')
    parser.add_argument('--edge_layer_idx', type=int, default=0, help='边缘层索引')
    parser.add_argument('--fusion_levels', nargs='+', type=int, default=[1, 2], help='融合层级')
    parser.add_argument('--edge_attention', type=str, default='none', choices=['none', 'se', 'cbam'])
    parser.add_argument('--fusion_mode', type=str, default='concat', choices=['concat', 'gate', 'dual'])

    args = parser.parse_args()

    # 构建模型
    extra_kwargs = {}
    if args.model_type == 'edge_enhanced':
        extra_kwargs.update({
            'edge_layer_idx': args.edge_layer_idx,
            'fusion_levels': args.fusion_levels,
            'edge_attention': args.edge_attention,
            'fusion_mode': args.fusion_mode,
        })
    elif args.model_type in ['dual_branch', 'dual_branch_enhanced']:
        extra_kwargs.update({
            'fusion_levels': args.fusion_levels,
            'edge_attention': args.edge_attention,
            'fusion_mode': args.fusion_mode,
        })

    model = build_model(
        model_type=args.model_type,
        size=args.size,
        num_classes=args.num_classes,
        **extra_kwargs
    )
    model.eval()

    # 创建虚拟输入
    input_tensor = torch.randn(1, 3, args.input_size, args.input_size)

    # 计算 FLOPs 和 Params
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops_str = f"{flops / 1e9:.3f}G" if flops > 1e9 else f"{flops / 1e6:.3f}M"
        params_str = f"{params / 1e6:.3f}M" if params > 1e6 else f"{params / 1e3:.3f}K"

        result = {
            "model_type": args.model_type,
            "size": args.size,
            "num_classes": args.num_classes,
            "input_size": args.input_size,
            "parameters": params_str,
            "flops": flops_str,
            "parameters_raw": int(params),
            "flops_raw": int(flops)
        }

        print("="*60)
        print(f"📊 MedMamba Model Complexity")
        print(f"   Model Type: {args.model_type}")
        print(f"   Size: {args.size}")
        print(f"   Input: {args.input_size}x{args.input_size}")
        print(f"   Parameters: {params_str}")
        print(f"   FLOPs: {flops_str}")
        print("="*60)

        # 保存结果
        filename = f"medmamba_{args.model_type}_{args.size}_complexity.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"💾 结果已保存到: {filename}")

    except Exception as e:
        print(f"❌ 计算失败: {e}")
        print("请确保已安装: pip install thop")


if __name__ == "__main__":
    main()