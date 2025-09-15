# import torch, pandas as pd, sys, traceback
#
# # ---- 依赖探测
# def _has(mod):
#     try:
#         __import__(mod); return True
#     except Exception:
#         return False
#
# HAS_PTFLOPS = _has("ptflops")
# HAS_FVCORE  = _has("fvcore.nn")
# HAS_THOP    = _has("thop")
# import timm
#
# if HAS_PTFLOPS:
#     from ptflops import get_model_complexity_info
# if HAS_FVCORE:
#     from fvcore.nn import FlopCountAnalysis
# if HAS_THOP:
#     from thop import profile
#
# IMG = 224
# DUMMY = torch.randn(1, 3, IMG, IMG)
#
# SERIES = {
#     "resnet": [
#         "resnet18","resnet34","resnet50","resnet101","resnet152"
#     ],
#     "convnext": [
#         "convnext_tiny","convnext_small","convnext_base","convnext_large"
#     ],
#     "densenet": [
#         "densenet121","densenet169","densenet201","densenet161"
#     ],
#     "vgg": [
#         "vgg11_bn","vgg13_bn","vgg16_bn","vgg19_bn"
#     ],
#     "swin": [
#         "swin_tiny_patch4_window7_224",
#         "swin_small_patch4_window7_224",
#         "swin_base_patch4_window7_224"
#     ],
#     "vit": [
#         "vit_tiny_patch16_224",
#         "vit_small_patch16_224",
#         "vit_base_patch16_224",
#         "vit_large_patch16_224"
#     ],
#     "efficientnet": [
#         "efficientnet_b0","efficientnet_b1","efficientnet_b2","efficientnet_b3",
#         "efficientnet_b4","efficientnet_b5","efficientnet_b6","efficientnet_b7"
#     ],
# }
#
# def count_params(model):
#     return sum(p.numel() for p in model.parameters())
#
# def try_ptflops(model):
#     if not HAS_PTFLOPS: return None
#     try:
#         macs, _ = get_model_complexity_info(
#             model, (3, IMG, IMG),
#             as_strings=False, print_per_layer_stat=False, verbose=False
#         )
#         return {"macs": float(macs), "flops": float(macs)*2}
#     except Exception:
#         return None
#
# def try_fvcore(model):
#     if not HAS_FVCORE: return None
#     try:
#         model.eval()
#         with torch.no_grad():
#             flops = float(FlopCountAnalysis(model, DUMMY).total())
#         return {"macs": flops/2.0, "flops": flops}
#     except Exception:
#         return None
#
# def try_thop(model):
#     if not HAS_THOP: return None
#     try:
#         model.eval()
#         with torch.no_grad():
#             macs, _ = profile(model, inputs=(DUMMY,), verbose=False)
#         return {"macs": float(macs), "flops": float(macs)*2}
#     except Exception:
#         return None
#
# def measure_one(name):
#     model = timm.create_model(name, pretrained=False, num_classes=2)
#     params = count_params(model)
#     r = try_ptflops(model) or try_fvcore(model) or try_thop(model)
#     flopsG = (r["flops"]/1e9) if r is not None else None
#     return {"model": name, "params": params, "params(M)": params/1e6, "flops(G)": flopsG}
#
# all_rows = []
# for family, names in SERIES.items():
#     for n in names:
#         try:
#             all_rows.append({"family": family, **measure_one(n)})
#         except Exception as e:
#             all_rows.append({"family": family, "model": n, "params": None, "params(M)": None, "flops(G)": None,
#                              "note": "ERROR: " + repr(e)})
#
# df = pd.DataFrame(all_rows)
#
# # 保存一份带模型名的完整结果（便于留档/查错）
# df_full = df.sort_values(by=["family","params"], ascending=[True,True], na_position="last")
# df_full.to_csv("all_series_full_with_names.csv", index=False)
# print("Saved full results with names -> all_series_full_with_names.csv")
#
# # 控制台仅输出两列；按“各系列内部参数量升序”分块打印
# for family in SERIES.keys():
#     sub = df[df["family"] == family].copy()
#     sub = sub.sort_values(by=["params"], ascending=True, na_position="last")
#     two_cols = sub[["params(M)", "flops(G)"]]
#     print(f"\n===== {family} (sorted by params) =====")
#     print(two_cols.to_string(index=False))


import torch
import torchvision.models as models
from thop import profile
from thop import clever_format
import sys

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型列表：名称和对应函数
DENSENET_MODELS = {
    "densenet121": models.densenet121,
    "densenet161": models.densenet161,
    "densenet169": models.densenet169,
    "densenet201": models.densenet201,
}

# 输入配置
INPUT_SIZE = (3, 224, 224)  # 3通道，224x224
NUM_CLASSES = 2  # 二分类


def modify_classifier(model, num_classes):
    """替换最后的分类层为二分类"""
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, num_classes)
    return model


def compute_model_stats(model_name, model_func):
    print(f"\n📊 正在分析 {model_name} ...")

    # 加载模型并修改分类头
    model = model_func(pretrained=False)  # 不加载预训练权重，避免干扰
    model = modify_classifier(model, NUM_CLASSES)
    model.eval()  # 设置为评估模式（影响BN/ dropout行为）

    # 创建虚拟输入
    dummy_input = torch.randn(1, *INPUT_SIZE).to(device)
    model.to(device)

    # 计算参数量（仅可训练参数）
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_million = total_params / 1e6

    # 计算 FLOPs（使用 thop）
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    flops_giga = flops / 1e9

    # 格式化输出（可选）
    flops_formatted, params_formatted = clever_format([flops, total_params], "%.4f")

    print(f"  参数量: {params_million:.6f} M ({params_formatted})")
    print(f"  计算量: {flops_giga:.6f} G ({flops_formatted})")

    return model_name, params_million, flops_giga


def main():
    print("🔍 计算 DenseNet 系列模型在 224×224 输入、二分类下的参数量与 FLOPs")
    print("=" * 70)

    results = []
    for name, func in DENSENET_MODELS.items():
        try:
            name, params, flops = compute_model_stats(name, func)
            results.append((name, params, flops))
        except Exception as e:
            print(f"❌ 失败: {name} - {e}")

    # 打印汇总表格
    print("\n" + "=" * 70)
    print("📋 最终汇总表（224×224 输入，二分类）")
    print("=" * 70)
    print(f"{'模型名称':<12} {'参数量 (M)':<12} {'FLOPs (G)':<12}")
    print("-" * 70)
    for name, params, flops in results:
        print(f"{name:<12} {params:<12.6f} {flops:<12.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
