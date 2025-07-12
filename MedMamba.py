# -*- coding: utf-8 -*-
import time
import math
from functools import partial
from typing import Optional, Callable
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    估算 Selective Scan 的理论 FLOPs（乘加运算次数）

    参数说明：
        B: 批量大小（batch size）
        L: 序列长度（通常为 H × W）
        D: 通道维度（投影维度）
        N: 状态空间维度（每通道的状态维）
        with_D: 是否包含残差项 D ⋅ u
        with_Z: 是否包含门控项 z
        with_Group: 是否使用 group 模式（B×N×L 格式）
        with_complex: 是否包含复数计算（目前未启用）

    变量说明：
        u: 输入序列 [B, D, L]
        delta: 时间步长 [B, D, L]
        A: 状态转移矩阵 [D, N]
        B/C: 状态激活矩阵 [B, N, L]
        D: 残差参数 [D]
        z: 门控向量 [B, D, L]
        delta_bias: Δt 偏置 [D]，忽略 FLOPs
    """

    import numpy as np

    # 定义辅助函数：基于 numpy.einsum_path 获取乘加操作 FLOPs
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]  # 构造零数组
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]  # 获取最优路径报告
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # 除以 2 是因为 MAC（乘加）视为一次 FLOP
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex  # 暂不支持复数模式

    flops = 0  # 总 FLOPs 初始化

    # einsum：bdl × dn → bdln（输入 × A）
    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")

    # 状态计算部分：
    if with_Group:
        # bdl × bnl × bdl → bdln（共享状态路径）
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        # bdl × bdnl × bdl → bdln（不共享状态路径）
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")

    # 每一步状态更新内部的循环运算 FLOPs（外层 for 循环 × L 次）
    in_for_flops = B * D * N
    if with_Group:
        # 状态更新（bdn × bdn → bd）
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        # 状态更新（bdn × bn → bd）
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")

    flops += L * in_for_flops  # 总体乘以 L（每个 token 都会执行）

    # 可选项：加上 D ⋅ u 的残差路径
    if with_D:
        flops += B * D * L

    # 可选项：门控 z ⋅ y
    if with_Z:
        flops += B * D * L

    return flops  # 返回总 FLOPs 估算值


class PatchEmbed2D(nn.Module):
    r""" 图像转为 Patch 嵌入（Patch Embedding）

    参数说明:
        patch_size (int): 每个 patch 的尺寸（高和宽），默认 4。
        in_chans (int): 输入图像的通道数（如 RGB 为 3），默认 3。
        embed_dim (int): 输出的嵌入通道维度，默认 96。
        norm_layer (nn.Module): 归一化层，默认 None。
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()

        # 如果 patch_size 是整数，转为 tuple 形式
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        # 使用 Conv2D 实现 Patch 提取与嵌入：
        # 输入为 [B, C, H, W]，输出为 [B, embed_dim, H/P, W/P]
        self.proj = nn.Conv2d(
            in_chans,  # 输入通道数（如 3）
            embed_dim,  # 输出通道数，即 patch embedding 维度
            kernel_size=patch_size,  # 卷积核大小等于 patch 大小
            stride=patch_size  # 步长也等于 patch 大小，实现非重叠 patch 划分
        )

        # 如果指定了归一化层，则创建；否则设为 None
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # 输入 x: [B, C, H, W]
        x = self.proj(x).permute(0, 2, 3, 1)  # 输出变为 [B, H/P, W/P, embed_dim]

        # 如果使用归一化，则应用
        if self.norm is not None:
            x = self.norm(x)

        return x  # 返回 patch 嵌入特征图


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer（补丁合并层，用于下采样）

    参数说明:
        dim (int): 输入通道数。
        norm_layer (nn.Module): 归一化层，默认使用 LayerNorm。
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim  # 输入特征的通道数

        # 线性层：将拼接后的 4*C 维度特征压缩为 2*C
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

        # 对拼接的 4*C 特征做归一化
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape  # 输入尺寸：[batch, height, width, channels]

        SHAPE_FIX = [-1, -1]  # 用于修正奇数高宽的补丁分割问题
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        # 将输入的 feature map 拆成 4 个 2x2 邻域位置：
        # 每个子块大小为 H/2 × W/2 × C
        x0 = x[:, 0::2, 0::2, :]  # 左上角像素
        x1 = x[:, 1::2, 0::2, :]  # 左下角像素
        x2 = x[:, 0::2, 1::2, :]  # 右上角像素
        x3 = x[:, 1::2, 1::2, :]  # 右下角像素

        # 如果原始图像 H/W 为奇数，截掉多余的最后一行/列，确保对齐
        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        # 沿 channel 维拼接 4 个子块：最终维度为 [B, H/2, W/2, 4*C]
        x = torch.cat([x0, x1, x2, x3], -1)

        # 调整形状（其实 shape 不变，因为上一行已拼成 [B, H/2, W/2, 4*C]）
        x = x.view(B, H // 2, W // 2, 4 * C)

        # 对 4*C 的特征做归一化
        x = self.norm(x)

        # 用线性层将通道数从 4*C 降维为 2*C
        x = self.reduction(x)

        return x  # 返回下采样后的特征图

    

class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2                          # 输入通道数扩大两倍（通常因为跳跃连接会拼接两个特征图）
        self.dim_scale = dim_scale                  # 上采样的空间倍数（高和宽方向各扩大 dim_scale 倍）

        # 线性映射：将输入通道扩展为 dim_scale^2 倍的空间大小所需的通道数
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)

        # 输出通道会变为 self.dim // dim_scale，设置对应的归一化层
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape  # 输入维度：[batch, height, width, channels]
        x = self.expand(x)    # 线性扩展通道维度

        # rearrange：将通道维重排为更大的空间分辨率（上采样）
        # 将 [B, H, W, dim_scale^2 * C'] → [B, H * dim_scale, W * dim_scale, C']
        x = rearrange(
            x,
            'b h w (p1 p2 c) -> b (h p1) (w p2) c',
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=C // self.dim_scale
        )

        x = self.norm(x)  # 对上采样结果做归一化处理

        return x          # 返回上采样后的特征图

    

class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim                            # 输入通道维度
        self.dim_scale = dim_scale                # 每个 patch 在高宽方向的扩展倍数
        # 线性映射，将每个 token 的通道数扩展为 dim_scale * dim
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        # 对重排后的输出进行归一化处理
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape                      # 输入尺寸：[批量, 高, 宽, 通道]
        x = self.expand(x)                        # 通道维度扩展

        # rearrange：将扩展后的通道重排为更大的空间尺寸
        # 例如：从 [B, H, W, 4*C] → [B, 2*H, 2*W, C]（假设 dim_scale=2）
        x = rearrange(
            x,
            'b h w (p1 p2 c) -> b (h p1) (w p2) c',
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=C // self.dim_scale
        )

        x = self.norm(x)                          # 对最终输出归一化

        return x                                  # 返回上采样后的特征图


# VMamba
class SS2D(nn.Module):
    def __init__(
        self,
        d_model,                      # 输入特征维度（如 token/channel 数）
        d_state=16,                   # 状态空间维度（控制记忆容量）
        # d_state="auto",             # 可选：自动设置状态维度（已注释）
        d_conv=3,                     # 卷积核大小（用于深度可分离卷积）
        expand=2,                     # 通道扩展倍数（内部维度 = d_model * expand）
        dt_rank="auto",              # 动态时间建模 rank，低秩投影维度
        dt_min=0.001,                # 初始化时间尺度的最小值
        dt_max=0.1,                  # 初始化时间尺度的最大值
        dt_init="random",           # 时间尺度初始化方式（如 constant 或 random）
        dt_scale=1.0,                # 初始化缩放因子
        dt_init_floor=1e-4,          # softplus 后最小时间尺度限制
        dropout=0.,                  # 输出 dropout 比例
        conv_bias=True,              # 卷积是否带 bias
        bias=False,                  # 投影层是否带 bias
        device=None,
        dtype=None,
        **kwargs,
    ):
        # 设置构造参数
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model              # 输入维度
        self.d_state = d_state              # 状态空间维度
        self.d_conv = d_conv                # 卷积核大小
        self.expand = expand                # 通道扩展倍率
        self.d_inner = int(self.expand * self.d_model)  # 扩展后的内部维度
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank  # 低秩时间建模 rank

        # 输入投影：将输入映射为 2 * d_inner，用于后续 x/z 分支处理
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # 使用深度可分离卷积（每个通道独立卷积）
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,                 # 每通道独立卷积（Depthwise）
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,           # 保持尺寸不变
            **factory_kwargs,
        )

        self.act = nn.SiLU()  # 激活函数：Sigmoid-weighted Linear Unit

        # 为每个方向构造一个 x_proj（x -> dt、B、C 参数）
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        # 提取权重并转换为统一的参数张量（四个方向）
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # [4, 输出维度, 输入维度]
        del self.x_proj  # 删除 nn.Linear 模块，只保留权重参数

        # 构造 4 个方向的时间投影模块 dt_proj，并提取权重和 bias
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        # 将 4 个方向的时间投影权重和 bias 合并成参数张量
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # [4, 输入维度, rank]
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))      # [4, 输入维度]
        del self.dt_projs  # 删除原始 nn.Linear 模块，只保留权重与偏置参数

        # 初始化状态转移矩阵 A（对数形式），和跳跃连接参数 D
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # [4*d_inner, d_state]
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)                         # [4*d_inner]

        # 默认选择 forward_corev0 作为状态扫描函数
        # 可以根据需要替换为 forward_corev1（实现上略有不同）
        self.forward_core = self.forward_corev0

        # 输出归一化层和最终输出线性映射层
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # Dropout（可选）
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        # 创建一个线性层：将 dt_rank 映射到 d_inner，表示时间动态建模的低秩变换
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # 初始化 dt_proj 的权重，以保持初始化时的方差稳定
        dt_init_std = dt_rank ** -0.5 * dt_scale  # 使用 √d 的倒数作为标准差（类似 Transformer 中的初始化）

        if dt_init == "constant":
            # 所有权重设置为相同常数
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            # 从均匀分布中随机初始化
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            # 不支持的初始化方式
            raise NotImplementedError

        # 初始化 bias，使得 softplus(bias) 的值落在 [dt_min, dt_max] 之间
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)  # 将 bias 的 softplus 值初始化到指定范围

        # 将 softplus 的结果反推为 bias 的初始化值（softplus 的逆函数）
        # 参考：https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # softplus^{-1}(dt)

        # 将计算出的 bias 复制给 dt_proj 的偏置项
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        # 标记该 bias 参数为“不再重新初始化”（如后续统一 weight 初始化时不重置该 bias）
        dt_proj.bias._no_reinit = True

        return dt_proj  # 返回初始化好的线性层（用于投影时间尺度）

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D 真正使用的初始化方法，用于生成对数形式的状态矩阵 A_log

        # 构造一个序列 [1, 2, ..., d_state]，表示状态的指数尺度（不同时间响应速度）
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),  # shape: [d_state]
            "n -> d n",  # 扩展到 [d_inner, d_state]
            d=d_inner,
        ).contiguous()  # 保证内存连续

        # 取对数：将 A 转换为对数形式，之后会通过 exp 得到真正的状态矩阵 A
        A_log = torch.log(A)  # 保持为 float32 精度

        # 若需要多个副本（用于多方向处理），则进行复制扩展
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)  # [copies, d_inner, d_state]
            if merge:
                A_log = A_log.flatten(0, 1)  # 合并复制维度与通道维度 -> [copies*d_inner, d_state]

        # 转为可训练参数
        A_log = nn.Parameter(A_log)

        # 禁止权重衰减（weight decay），因为 A 控制动态响应特性，不应被正则化
        A_log._no_weight_decay = True

        return A_log  # 返回形如 [copies*d_inner, d_state] 的对数状态矩阵参数

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D 是状态空间模型中的“跳跃”系数（skip parameter），用于控制残差项强度

        # 创建一个全为 1 的张量，表示初始每个通道的残差权重都为 1
        D = torch.ones(d_inner, device=device)  # shape: [d_inner]

        # 如果需要多个方向副本（如上下左右），则复制 D
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)  # [copies, d_inner]
            if merge:
                D = D.flatten(0, 1)  # 合并副本维度和通道维度，得到 [copies * d_inner]

        # 转为可学习参数（将被优化器更新）
        D = nn.Parameter(D)  # 保持 float32 精度

        # 禁止权重衰减（weight decay），以免破坏残差平衡性
        D._no_weight_decay = True

        return D  # 返回形状为 [copies * d_inner] 的残差跳跃系数

    def forward_corev0(self, x: torch.Tensor):
        # 绑定 selective_scan 函数（通常外部注入）
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape  # 输入形状：[batch, channels, height, width]
        L = H * W  # 每个图像展平成序列的长度
        K = 4  # 四个方向（正序/逆序，水平/垂直）

        # 1. 构造四个方向的输入序列：
        #    - x.view(...)：水平扫描（HW 展开）
        #    - transpose(x)：垂直扫描（WH 展开）
        #    - 再拼接其逆序（flip）
        x_hwwh = torch.stack([
            x.view(B, -1, L),  # 方向 1：水平正向
            torch.transpose(x, dim0=2, dim1=3)  # 方向 2：垂直正向
            .contiguous()
            .view(B, -1, L)
        ], dim=1).view(B, 2, -1, L)

        # 构成四个方向（正→反）的序列输入：x, x_T, flip(x), flip(x_T)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # [B, 4, C, L]

        # 2. 投影：对四个方向的序列进行线性变换（提取 dt, B, C 参数）
        # einsum: [B, 4, D, L] * [4, C, D] -> [B, 4, C, L]
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl.shape = [B, 4, dt_rank + d_state*2, L]

        # 拆分出三部分参数：
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        # 3. 对 dt 参数再投影（低秩）为每个通道的时间步长 Δt
        # einsum: [B, 4, R, L] * [4, D, R] -> [B, 4, D, L]
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # 可选：+ dt bias（已注释）

        # 4. 准备 selective_scan 输入：
        xs = xs.float().view(B, -1, L)  # 输入序列展平 [B, 4*C, L]
        dts = dts.contiguous().float().view(B, -1, L)  # 时间步长 Δt [B, 4*C, L]
        Bs = Bs.float().view(B, K, -1, L)  # B 矩阵 [B, 4, d_state, L]
        Cs = Cs.float().view(B, K, -1, L)  # C 矩阵 [B, 4, d_state, L]
        Ds = self.Ds.float().view(-1)  # 残差参数 [4*C]
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # 状态转移矩阵 A = -exp(logA)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # Δt bias [4*C]

        # 5. 执行 selective_scan（核心状态空间推理）
        out_y = self.selective_scan(
            xs, dts,  # 输入序列 & 时间步长
            As, Bs, Cs, Ds,  # 状态转移相关矩阵
            z=None,
            delta_bias=dt_projs_bias,  # Δt 偏置项
            delta_softplus=True,  # 对 Δt 使用 softplus 激活
            return_last_state=False,  # 不返回最终状态，仅返回输出序列
        ).view(B, K, -1, L)  # 输出形状：[B, 4, C, L]

        assert out_y.dtype == torch.float  # 确保输出为 float32

        # 6. 构造两个翻转方向的输出
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)

        # 7. 垂直方向特征需要还原维度（WH → HW）
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        # 返回四个方向的输出：[正向横、反向横、正向纵、反向纵]
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # forward_corev1：作为 forward_corev0 的替代版本（调用 selective_scan_fn_v1）
    def forward_corev1(self, x: torch.Tensor):
        # 使用新版 selective scan 实现（可能是更快或更精简的版本）
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape  # 输入维度：[batch, channels, height, width]
        L = H * W  # 每个图像展平后的序列长度
        K = 4  # 方向数（水平正反、垂直正反）

        # 构造横向和纵向方向的序列（正向）
        x_hwwh = torch.stack([
            x.view(B, -1, L),  # 横向正向：[B, C, L]
            torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)  # 纵向正向
        ], dim=1).view(B, 2, -1, L)

        # 拼接出 4 个方向：正横、正纵、反横、反纵
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # [B, 4, C, L]

        # 对 4 个方向的序列做线性映射，得到 dt/B/C 三类参数
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # 可选：添加 bias（这里注释掉了）

        # 拆分投影结果为三部分：dt（低秩时间）、B、C
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        # 再次映射 dt 以得到时间步长向量（Delta t）
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # 可选：添加 bias（这里注释掉了）

        # 数据整理为统一的 flat 格式，准备输入 selective scan
        xs = xs.float().view(B, -1, L)  # [B, 4*C, L]
        dts = dts.contiguous().float().view(B, -1, L)  # [B, 4*C, L]
        Bs = Bs.float().view(B, K, -1, L)  # [B, 4, d_state, L]
        Cs = Cs.float().view(B, K, -1, L)  # [B, 4, d_state, L]
        Ds = self.Ds.float().view(-1)  # [4*C]，残差项
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # 状态转移矩阵 A（负指数）
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # dt 的偏置项

        # 调用 selective_scan_fn_v1 处理状态空间序列建模
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)  # 输出恢复为 [B, 4, C, L]

        assert out_y.dtype == torch.float  # 保证输出为 float32

        # 翻转方向（反横、反纵）需要重新 flip 回来
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)

        # 垂直方向需要转置回 HxW 顺序
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        # 返回四个方向：正横、反横、正纵、反纵（顺序同 v0）
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape  # 输入维度为 [batch, height, width, channels]

        # 1. 输入线性投影：将通道维映射到 2 × d_inner
        xz = self.in_proj(x)  # 输出形状为 [B, H, W, 2 × d_inner]

        # 拆分为两部分：
        # x：送入状态建模路径（state modeling）
        # z：门控路径（gate），用于后续融合（GLU-like）
        x, z = xz.chunk(2, dim=-1)  # 各自形状：[B, H, W, d_inner]

        # 2. 格式转换：[B, H, W, C] → [B, C, H, W]，用于卷积
        x = x.permute(0, 3, 1, 2).contiguous()

        # 3. 深度可分离卷积 + 激活（提取局部空间特征）
        x = self.act(self.conv2d(x))  # 输出：[B, d_inner, H, W]

        # 4. 调用 forward_core（状态空间扫描），获得四个方向的输出
        y1, y2, y3, y4 = self.forward_core(x)  # 每个为 [B, C, L]，其中 L = H × W
        assert y1.dtype == torch.float32

        # 5. 聚合四个方向的信息（加和）
        y = y1 + y2 + y3 + y4  # 输出：[B, C, L]

        # 6. 转置回 [B, H, W, C] 格式
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        # 7. 输出归一化
        y = self.out_norm(y)  # [B, H, W, C]

        # 8. 与门控分支 z 相乘（GLU-like 调制，z 经 SiLU 激活）
        y = y * F.silu(z)  # 注意：这不是加法，而是元素乘 → 类似门控机制

        # 9. 输出映射回原始维度（d_model）
        out = self.out_proj(y)  # 输出：[B, H, W, d_model]

        # 10. 可选 Dropout
        if self.dropout is not None:
            out = self.dropout(out)

        return out  # 最终输出


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    # 获取输入张量的形状：[B, H, W, C]
    batch_size, height, width, num_channels = x.size()

    # 每组通道数
    channels_per_group = num_channels // groups

    # 重塑张量形状：
    # [B, H, W, C] -> [B, H, W, groups, channels_per_group]
    x = x.view(batch_size, height, width, groups, channels_per_group)

    # 在 group 和 channel_per_group 维度上做转置（交换位置）
    # 即 [B, H, W, groups, channels_per_group] -> [B, H, W, channels_per_group, groups]
    x = torch.transpose(x, 3, 4).contiguous()

    # 展平成原来的通道数： [B, H, W, C]
    x = x.view(batch_size, height, width, -1)

    return x  # 返回混洗后的特征图


class SS_Conv_SSM(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        # 对输入右半部分特征做 LayerNorm
        self.ln_1 = norm_layer(hidden_dim // 2)

        # 空间状态建模模块（SS2D），类似注意力机制的操作
        self.self_attention = SS2D(d_model=hidden_dim // 2, dropout=attn_drop_rate, d_state=d_state, **kwargs)

        # DropPath，用于实现随机深度（Stochastic Depth）
        self.drop_path = DropPath(drop_path)

        # 左半部分使用一组卷积模块进行局部建模：3x3 -> 3x3 -> 1x1
        self.conv33conv33conv11 = nn.Sequential(
            nn.BatchNorm2d(hidden_dim // 2),  # 批归一化
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=1, stride=1),
            nn.ReLU()
        )

        # 下面是注释掉的最终 1x1 卷积（未使用）
        # self.finalconv11 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1)

    def forward(self, input: torch.Tensor):
        # 将输入在最后一维平均分为左右两部分
        input_left, input_right = input.chunk(2, dim=-1)

        # 对右半部分先做归一化，再送入状态空间注意力模块，并应用 DropPath
        x = self.drop_path(self.self_attention(self.ln_1(input_right)))

        # 对左半部分做局部卷积处理
        input_left = input_left.permute(0, 3, 1, 2).contiguous()  # 调整维度为 [B, C, H, W]
        input_left = self.conv33conv33conv11(input_left)  # 卷积序列处理
        input_left = input_left.permute(0, 2, 3, 1).contiguous()  # 调整回原始维度顺序 [B, H, W, C]

        # 将两部分重新拼接
        output = torch.cat((input_left, x), dim=-1)

        # 通道混洗（增强特征交互）
        output = channel_shuffle(output, groups=2)

        # 残差连接，返回处理结果
        return output + input



class VSSLayer(nn.Module):
    """ 一个基本的 Swin Transformer 层（用于编码器阶段）。
    参数说明:
        dim (int): 输入通道数。
        depth (int): 层的深度（包含多少个 SS_Conv_SSM 块）。
        drop (float): Dropout 比例，默认 0.0。
        attn_drop (float): 注意力 Dropout 比例，默认 0.0。
        drop_path (float 或列表): 随机深度（Stochastic Depth）比例，默认 0.0。
        norm_layer (nn.Module): 使用的归一化层类型，默认 nn.LayerNorm。
        downsample (nn.Module | None): 是否在当前阶段末尾进行下采样，默认 None。
        use_checkpoint (bool): 是否启用梯度检查点技术以节省显存，默认 False。
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim  # 当前层通道数
        self.use_checkpoint = use_checkpoint  # 是否使用 checkpoint 节省内存

        # 构建多个 SS_Conv_SSM 块（核心的注意力/状态空间块）
        self.blocks = nn.ModuleList([
            SS_Conv_SSM(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,  # 支持逐层设置 drop_path
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])  # 构建 depth 个模块

        # 初始化部分权重（是否真正起效由后续 VSSM 决定）
        if True:  # 实际会执行，但初始化会被 VSSM 中 apply 再次覆盖
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # 克隆参数，保持随机性一致
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))  # 使用 Kaiming 均匀分布初始化

            self.apply(_init_weights)

        # 如果提供了下采样模块，则实例化；否则设为 None
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        # 逐个模块前向传播
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)  # 使用 checkpoint 节省内存
            else:
                x = blk(x)

        # 如果有下采样模块，则进行下采样
        if self.downsample is not None:
            x = self.downsample(x)

        return x  # 返回输出特征


class VSSLayer_up(nn.Module):
    """ 一个基本的 Swin Transformer 升采样层（用于解码阶段）。
    参数说明:
        dim (int): 输入通道数。
        depth (int): 模块重复次数。
        drop (float): Dropout 比例。默认 0.0。
        attn_drop (float): 注意力 Dropout 比例。默认 0.0。
        drop_path (float 或列表): 随机深度的比例。默认 0.0。
        norm_layer (nn.Module): 归一化层类型。默认 nn.LayerNorm。
        upsample (nn.Module | None): 是否使用上采样模块。默认 None。
        use_checkpoint (bool): 是否使用梯度检查点来节省内存。默认 False。
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            upsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim  # 当前层的通道数
        self.use_checkpoint = use_checkpoint  # 是否使用梯度检查点

        # 构建多个 SS_Conv_SSM 块，形成模块列表
        self.blocks = nn.ModuleList([
            SS_Conv_SSM(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,  # 支持列表或单个值
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])  # 重复 depth 次构建模块

        # 这个初始化函数在 VSSM 中可能会被覆盖
        if True:  # 此处确实会被执行，只是稍后可能会被覆盖
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # 克隆权重，用于保持随机种子一致
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))  # 使用 Kaiming 均匀分布初始化

            self.apply(_init_weights)

        # 如果提供了上采样模块，则实例化；否则为 None
        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        # 如果有上采样模块，先进行上采样
        if self.upsample is not None:
            x = self.upsample(x)
        # 逐块执行前向传播
        for blk in self.blocks:
            if self.use_checkpoint:
                # 使用 checkpoint 节省内存，代价是计算时间增加
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x  # 返回处理后的特征


from sobel import *

def build_fuser(fuse_type, inc, ouc, attention):
    """
    构建特征融合模块的工厂函数

    Args:
        fuse_type (str): 融合类型，可选 'gate' / 'concat' / 'dual'
        inc (List[int]): 输入通道列表，如 [256, 256]
        ouc (int): 输出通道数，如 256

    Returns:
        nn.Module: 实例化的融合模块
    """
    if fuse_type == 'gate':
        return GatedConvEdgeFusion(inc, ouc)
    elif fuse_type == 'concat':
        return ConvEdgeFusion(inc, ouc, attention)  # 原来的 concat + conv 融合
    elif fuse_type == 'dual':
        return DualPathEdgeFusion(inc, ouc)
    else:
        raise ValueError(f"Unsupported fusion type: {fuse_type}")

class DualBranchVSSM(nn.Module):
    """
    VSSM + Input-Edge 分支：
    1) 在原始输入图像上用 Sobel 提取边缘
    2) 对边缘特征按 patch_size*2^i 下采样到各个阶段分辨率
    3) 用 1×1 卷积映射到与每个阶段通道数相同
    4) 与主干特征逐层融合（可选 attention）
    """
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        depths=[2,2,4,2],
        dims=[96,192,384,768],
        d_state=16,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,

        # 🔥 新增参数
        fusion_levels=[0,1,2],        # 要融合的阶段索引，范围 [0, num_layers-1]
        edge_attention='none',         # 融合时的 attention 类型：'none'|'se'|'cbam'
        fusion_mode='concat',
        **kwargs
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.dims = dims
        self.patch_size = patch_size
        self.fusion_levels = sorted(fusion_levels)
        # —— 主干（同 VSSM） ——
        # patch 嵌入
        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size, in_chans=in_chans,
            embed_dim=dims[0],
            norm_layer=norm_layer if patch_norm else None
        )
        # drop & 层列表
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i, depth in enumerate(depths):
            self.layers.append(
                VSSLayer(
                    dim=dims[i], depth=depth, d_state=d_state,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                    norm_layer=norm_layer,
                    downsample=PatchMerging2D if i < self.num_layers-1 else None,
                    use_checkpoint=use_checkpoint
                )
            )
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[-1], num_classes) if num_classes>0 else nn.Identity()

        # —— 输入边缘分支 ——
        # 1) Sobel 提取原图边缘
        self.edge_extractor = SobelConv(in_chans)
        # 2) 针对每个 fusion level，下采样 + 1×1 映射
        self.edge_pools = nn.ModuleList()
        self.edge_convs  = nn.ModuleList()
        for lvl in self.fusion_levels:
            stride = patch_size * (2**lvl)
            # 用最大池化将原图边缘下采样到与第 lvl 阶段相同的空间尺寸
            self.edge_pools.append(nn.MaxPool2d(kernel_size=stride, stride=stride))
            # 用 1×1 conv 将通道映射到 dims[lvl]
            self.edge_convs.append(Conv(in_chans, dims[lvl], 1))

        # 3) 融合模块：同 VSSMEdgeEnhanced 中的 ConvEdgeFusion
        self.fusers = nn.ModuleList()
        for lvl in self.fusion_levels:
            C = dims[lvl]
            # 输入特征和边缘特征通道均为 C
            self.fusers.append(build_fuser(fusion_mode, [C, C], C, attention=edge_attention))

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_backbone(self, x):
        # x: [B, C, H, W]
        # —— 先生成各阶段的边缘特征 ——
        edge_feats = []
        e = self.edge_extractor(x)  # [B, C, H, W]
        for pool, conv in zip(self.edge_pools, self.edge_convs):
            e_ds = pool(e)            # 下采样到对应阶段空间尺寸
            edge_feats.append(conv(e_ds))  # [B, dims[lvl], H/(ps*2^lvl), W/...]
        # —— 主干 forward ——
        x = self.patch_embed(x)      # [B, H/ps, W/ps, dims[0]]
        x = self.pos_drop(x)
        for i, layer in enumerate(self.layers):
            # NCHW 转换
            x_nchw = x.permute(0,3,1,2).contiguous()
            if i in self.fusion_levels:
                idx = self.fusion_levels.index(i)
                xe = edge_feats[idx]
                # 将边缘特征融合到主干特征
                x_nchw = self.fusers[idx]([x_nchw, xe])
            # NCHW -> NHWC
            x = layer(x_nchw.permute(0,2,3,1).contiguous())
        return x

    def forward(self, x):
        x = self.forward_backbone(x)      # [B, H', W', C']
        x = x.permute(0,3,1,2)            # [B, C', H', W']
        x = self.avgpool(x)               # [B, C', 1, 1]
        x = torch.flatten(x, 1)
        return self.head(x)

class DualBranchVSSMEnhanced(nn.Module):
    """
    VSSM + Input-Edge 分支：
    1) 在原始输入图像上用 Sobel 提取边缘
    2) 对边缘特征按 patch_size*2^i 下采样到各个阶段分辨率
    3) 用 1×1 卷积映射到与每个阶段通道数相同
    4) 与主干特征逐层融合（可选 attention）
    """
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        depths=[2,2,4,2],
        dims=[96,192,384,768],
        d_state=16,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,

        # 🔥 新增参数
        fusion_levels=[0,1,2],        # 要融合的阶段索引，范围 [0, num_layers-1]
        edge_attention='none',         # 融合时的 attention 类型：'none'|'se'|'cbam'
        fusion_mode='concat',
        **kwargs
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.dims = dims
        self.patch_size = patch_size
        self.fusion_levels = sorted(fusion_levels)
        # —— 主干（同 VSSM） ——
        # patch 嵌入
        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size, in_chans=in_chans,
            embed_dim=dims[0],
            norm_layer=norm_layer if patch_norm else None
        )
        # drop & 层列表
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i, depth in enumerate(depths):
            self.layers.append(
                VSSLayer(
                    dim=dims[i], depth=depth, d_state=d_state,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                    norm_layer=norm_layer,
                    downsample=PatchMerging2D if i < self.num_layers-1 else None,
                    use_checkpoint=use_checkpoint
                )
            )
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[-1], num_classes) if num_classes>0 else nn.Identity()

        # —— 输入边缘分支 ——
        self.edge_generator = MultiScaleEdgeInfoGenerator(in_chans, [dims[i] for i in self.fusion_levels])

        # 3) 融合模块：同 VSSMEdgeEnhanced 中的 ConvEdgeFusion
        self.fusers = nn.ModuleList()
        for lvl in self.fusion_levels:
            C = dims[lvl]
            # 输入特征和边缘特征通道均为 C
            self.fusers.append(build_fuser(fusion_mode, [C, C], C, attention=edge_attention))

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_backbone(self, x):
        # x: [B, C, H, W]
        # —— 先生成各阶段的边缘特征 ——
        edge_feats = self.edge_generator(x)
        for edge_feat in edge_feats:
            print(edge_feat.shape)
        # —— 主干 forward ——
        x = self.patch_embed(x)      # [B, H/ps, W/ps, dims[0]]
        x = self.pos_drop(x)
        for i, layer in enumerate(self.layers):
            # NCHW 转换
            x_nchw = x.permute(0,3,1,2).contiguous()
            if i in self.fusion_levels:
                idx = self.fusion_levels.index(i)
                xe = edge_feats[idx]
                # 将边缘特征融合到主干特征
                x_nchw = self.fusers[idx]([x_nchw, xe])
            # NCHW -> NHWC
            x = layer(x_nchw.permute(0,2,3,1).contiguous())
        return x

    def forward(self, x):
        x = self.forward_backbone(x)      # [B, H', W', C']
        x = x.permute(0,3,1,2)            # [B, C', H', W']
        x = self.avgpool(x)               # [B, C', 1, 1]
        x = torch.flatten(x, 1)
        return self.head(x)

class VSSMEdgeEnhanced(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        depths=[2, 2, 4, 2],
        depths_decoder=[2, 9, 2, 2],
        dims=[96, 192, 384, 768],
        dims_decoder=[768, 384, 192, 96],
        d_state=16,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,

        # 👇 新增参数
        edge_layer_idx=0,          # 在哪一层提取边缘特征（默认 layer0）
        fusion_levels=[1, 2],      # 要融合边缘信息的层索引
        edge_attention='none',  # 加到 __init__ 中
        fusion_mode='concat',
            **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        # Patch Embedding
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)

        self.ape = False
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Drop path
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 编码器层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # 边缘特征提取器
        self.edge_layer_idx = edge_layer_idx
        self.fusion_levels = sorted(fusion_levels)

        # 自动推断 edge_chans
        self.edge_chans = [dims[level] for level in self.fusion_levels]
        self.edge_extractor = MultiScaleEdgeInfoGenerator(self.dims[self.edge_layer_idx], self.edge_chans)


        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        # 初始化 Conv2D 权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # 边缘融合模块
        self.num_fusers = len(self.fusion_levels)
        self.fusers = nn.ModuleList()

        for fuser_idx in range(self.num_fusers):
            inc = [self.edge_chans[fuser_idx], self.edge_chans[fuser_idx]]
            ouc = self.edge_chans[fuser_idx]
            self.fusers.append(build_fuser(fusion_mode, inc, ouc, attention=edge_attention))


    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_backbone(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        saved_edges = []

        for i, layer in enumerate(self.layers):
            # NHWC -> NCHW
            x_nchw = x.permute(0, 3, 1, 2).contiguous()

            # 提取边缘特征
            if i == self.edge_layer_idx:
                with torch.no_grad():
                    edges = self.edge_extractor(x_nchw)
                saved_edges = edges

            # 是否融合边缘特征
            if i in self.fusion_levels:
                fuser_idx = self.fusion_levels.index(i)
                edge_input = saved_edges[fuser_idx]

                # 不再插值，因为 shape 已匹配
                x_nchw = self.fusers[fuser_idx]([x_nchw, edge_input])

            # NCHW -> NHWC
            x = layer(x_nchw.permute(0, 2, 3, 1).contiguous())

        return x

    def forward(self, x):
        x = self.forward_backbone(x)
        x = x.permute(0, 3, 1, 2)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 4, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96,192,384,768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes  # 分类类别数
        self.num_layers = len(depths)  # 编码器层数
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]  # 自动扩展维度
        self.embed_dim = dims[0]  # 初始嵌入维度
        self.num_features = dims[-1]  # 最后一层输出的特征维度
        self.dims = dims  # 每层的维度列表

        # Patch embedding 模块，将输入图像分割为 patch 并投影到高维空间
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        # 是否使用绝对位置编码（此处未启用）
        self.ape = False
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution  # Patch 的分辨率
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))  # 初始化绝对位置编码
            trunc_normal_(self.absolute_pos_embed, std=.02)  # 截断正态分布初始化
        self.pos_drop = nn.Dropout(p=drop_rate)  # Dropout 层

        # drop path 比例：用于随机深度技术
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]  # 解码器用的 dpr，顺序反转

        # 构建编码器各层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],  # 当前层的维度
                depth=depths[i_layer],  # 当前层的深度
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # SSM 状态维度
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # 当前层的 drop path 参数
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,  # 是否进行下采样
                use_checkpoint=use_checkpoint,  # 是否使用梯度检查点以节省内存
            )
            self.layers.append(layer)  # 添加到模块列表中

        # 平均池化与分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化到 1x1
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()  # 分类层或恒等映射

        self.apply(self._init_weights)  # 初始化所有权重

        # 额外初始化 Conv2D 层的权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _init_weights(self, m: nn.Module):
        """
        初始化权重的方法
        注意：SS_Conv_SSM 中的 out_proj.weight 初始化会被 nn.Linear 覆盖
        另外模型中未使用 nn.Embedding 或 fc.weight
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)  # 初始化全连接层权重
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 初始化偏置为0
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  # 初始化 LayerNorm 的偏置为0
            nn.init.constant_(m.weight, 1.0)  # 初始化权重为1

    @torch.jit.ignore
    def no_weight_decay(self):
        # 返回不参与权重衰减的参数名集合
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        # 返回包含不进行权重衰减关键字的参数名（模糊匹配）
        return {'relative_position_bias_table'}

    def forward_backbone(self, x):
        # 前向传播主干部分
        x = self.patch_embed(x)  # patch embedding
        if self.ape:
            x = x + self.absolute_pos_embed  # 加上绝对位置编码
        x = self.pos_drop(x)  # Dropout

        for layer in self.layers:
            x = layer(x)  # 逐层前向传播
        return x

    def forward(self, x):
        x = self.forward_backbone(x)  # 主干前向传播
        x = x.permute(0,3,1,2)  # 调整维度顺序 (B, H, W, C) -> (B, C, H, W)
        x = self.avgpool(x)  # 平均池化
        x = torch.flatten(x,start_dim=1)  # 展平为二维 (B, C)
        x = self.head(x)  # 分类头
        return x



medmamba_t = VSSM(depths=[2, 2, 4, 2],dims=[96,192,384,768],num_classes=6).to("cuda")
medmamba_s = VSSM(depths=[2, 2, 8, 2],dims=[96,192,384,768],num_classes=6).to("cuda")
medmamba_b = VSSM(depths=[2, 2, 12, 2],dims=[128,256,512,1024],num_classes=6).to("cuda")

data = torch.randn(1,3,224,224).to("cuda")

print(medmamba_t(data).shape)
