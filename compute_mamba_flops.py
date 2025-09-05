# -*- coding: utf-8 -*-

import torch
import argparse
from thop import profile, clever_format

# 动态导入模型（请确保 MedMamba.py 在路径中）
from MedMamba import VSSM as vssm
from MedMamba import VSSMEdgeEnhanced as edge_enhanced
from MedMamba import DualBranchVSSM as dual_branch
from MedMamba import DualBranchVSSMEnhanced as dual_branch_enhanced

# 模型映射
MODEL_MAP = {
    'vssm': vssm,
    'edge_enhanced': edge_enhanced,
    'dual_branch': dual_branch,
    'dual_branch_enhanced': dual_branch_enhanced,
}


def count_params_and_flops(model, input_size=(3, 224, 224)):
    """
    计算模型的参数量和FLOPs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = torch.randn(1, *input_size).to(device)

    # 计算 FLOPs 和参数量
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    return params, flops


def main():
    parser = argparse.ArgumentParser(description="Compute Params and FLOPs for MedMamba Models")
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['vssm', 'edge_enhanced', 'dual_branch', 'dual_branch_enhanced'],
                        help='Type of model to analyze')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size (default: 224)')

    # Edge fusion 相关参数（仅对特定模型生效）
    parser.add_argument('--edge_layer_idx', type=int, default=0)
    parser.add_argument('--fusion_levels', nargs='+', type=int, default=[1, 2])
    parser.add_argument('--edge_attention', type=str, default='none', choices=['none', 'se', 'cbam'])
    parser.add_argument('--fusion_mode', type=str, default='concat', choices=['concat', 'gate', 'dual'])

    args = parser.parse_args()

    print(f"🔍 Analyzing model: {args.model_type.upper()}")
    print(f"🖼️  Input size: {(3, args.input_size, args.input_size)}")
    print("-" * 60)

    # 构建模型
    model_class = MODEL_MAP[args.model_type]
    model_kwargs = {}

    if args.model_type == 'edge_enhanced':
        model_kwargs.update({
            'edge_layer_idx': args.edge_layer_idx,
            'fusion_levels': args.fusion_levels,
            'edge_attention': args.edge_attention,
            'fusion_mode': args.fusion_mode,
        })
    elif args.model_type in ['dual_branch', 'dual_branch_enhanced']:
        model_kwargs.update({
            'fusion_levels': args.fusion_levels,
            'edge_attention': args.edge_attention,
            'fusion_mode': args.fusion_mode,
        })

    # 实例化模型
    net = model_class(
        depths=[2, 2, 4, 2],
        dims=[96, 192, 384, 768],
        num_classes=args.num_classes,
        **model_kwargs
    )

    # 方法1：使用 thop 计算 FLOPs 和 Params
    params_raw, flops_raw = profile(
        net,
        inputs=(torch.randn(1, 3, args.input_size, args.input_size),),
        verbose=False
    )
    params_formatted, flops_formatted = clever_format([params_raw, flops_raw], "%.3f")

    print(f"🧮 Model: {args.model_type}")
    print(f"   Parameters: {params_formatted} ({params_raw:,})")
    print(f"   FLOPs: {flops_formatted} ({flops_raw:,}) @ {(args.input_size, args.input_size)}")


if __name__ == '__main__':
    import os
    import sys

    sys.path.append(".")  # 确保能导入 MedMamba
    main()