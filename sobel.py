# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np




def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))




class SobelConv(nn.Module):
    def __init__(self, channel):
        super().__init__()

        # Sobel 核心
        # sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        sobel_y = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=np.float32)
        sobel_x = sobel_y.T  # 横向

        # 转成 tensor，shape �?(1, 1, 3, 3)
        kernel_y = torch.tensor(sobel_y).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        kernel_x = torch.tensor(sobel_x).unsqueeze(0).unsqueeze(0)

        # 扩展为每个通道独立卷积�?(C, 1, 3, 3)
        kernel_y = kernel_y.repeat(channel, 1, 1, 1)  # shape: (C, 1, 3, 3)
        kernel_x = kernel_x.repeat(channel, 1, 1, 1)

        # 定义 depthwise Conv2d（每个通道单独卷积�?
        self.sobel_conv_y = nn.Conv2d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=3,
            padding=1,
            groups=channel,
            bias=False
        )
        self.sobel_conv_x = nn.Conv2d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=3,
            padding=1,
            groups=channel,
            bias=False
        )

        # 赋值并固定权重
        self.sobel_conv_y.weight.data = kernel_y.clone()
        self.sobel_conv_x.weight.data = kernel_x.clone()
        self.sobel_conv_y.weight.requires_grad = False
        self.sobel_conv_x.weight.requires_grad = False

    def forward(self, x):
        edge_x = self.sobel_conv_x(x)
        edge_y = self.sobel_conv_y(x)
        edge = edge_x + edge_y
        return edge


class MultiScaleEdgeInfoGenerator(nn.Module):
    def __init__(self, inc, oucs) -> None:
        super().__init__()

        self.sc = SobelConv(inc)
        # 下采�?
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_1x1s = nn.ModuleList(Conv(inc, ouc, 1) for ouc in oucs)

    def forward(self, x):
        outputs = [self.sc(x)]
        outputs.extend(self.maxpool(outputs[-1]) for _ in self.conv_1x1s)
        outputs = outputs[1:]
        for i in range(len(self.conv_1x1s)):
            outputs[i] = self.conv_1x1s[i](outputs[i])
        return outputs

class MultiScaleEdgeInfoGenerator_422(nn.Module):
    def __init__(self, inc, oucs) -> None:
        super().__init__()

        self.sc = SobelConv(inc)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_1x1s = nn.ModuleList(Conv(inc, ouc, 1) for ouc in oucs)

    def forward(self, x):
        edge = self.sc(x)  # 原始边缘特征 (B, C, H, W)

        outputs = []

        # 🔽 第一个输出：下采样4倍（MaxPool两次）
        x_down = self.maxpool(edge)  # H/2
        x_down = self.maxpool(x_down)  # H/4
        outputs.append(x_down)

        # 🔁 后续输出：在前一个基础上继续下采样
        for _ in range(len(self.conv_1x1s) - 1):
            x_down = self.maxpool(x_down)
            outputs.append(x_down)

        # 每个下采样特征通过对应的 1x1 卷积降维
        for i in range(len(self.conv_1x1s)):
            outputs[i] = self.conv_1x1s[i](outputs[i])

        return outputs


class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pool(x)
        y = self.fc(y)
        return x * y
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度上做最大池化和平均池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio=reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# 🔥 合并后的模块
class ConvEdgeFusion(nn.Module):
    def __init__(self, inc, ouc, attention='none', reduction=16):
        """
        Args:
            inc (List[int]): list of input channels, e.g. [256, 256]
            ouc (int): output channel after fusion
            attention (str): attention type ['none', 'se', 'cbam']
            reduction (int): reduction ratio for SE/CBAM
        """
        super().__init__()
        self.attention_type = attention.lower()

        # 特征融合过程
        self.conv_channel_fusion = Conv(sum(inc), ouc // 2, k=1)
        self.conv_3x3_feature_extract = Conv(ouc // 2, ouc // 2, 3)

        # 注意力机制（可选）
        if self.attention_type == 'se':
            self.attention = SE(sum(inc), reduction=reduction)
        elif self.attention_type == 'cbam':
            self.attention = CBAM(sum(inc), reduction_ratio=reduction)
        elif self.attention_type == 'none':
            self.attention = nn.Identity()
        else:
            raise ValueError(f"Unsupported attention type: {attention}")

        # 输出卷积
        self.conv_1x1 = Conv(ouc // 2, ouc, 1)

    def forward(self, x_list):
        x = torch.cat(x_list, dim=1)  # 拼接输入特征�?
        x = self.attention(x)
        x = self.conv_channel_fusion(x)
        x = self.conv_3x3_feature_extract(x)
        x = self.conv_1x1(x)
        return x

class ConvEdgeFusion1(nn.Module):
    def __init__(self, inc, ouc, attention='se', reduction=16):
        super().__init__()
        self.attention_type = attention.lower()

        # 可选的分支归一�?
        self.edge_proj = Conv(inc[0], inc[0], k=3)
        self.backbone_proj = Conv(inc[1], inc[1], k=3)

        # 通道融合
        self.conv_channel_fusion = Conv(sum(inc), ouc, k=1)

        # 特征提取
        self.conv_3x3 = Conv(ouc, ouc, k=3)

        # 注意力模块（放在特征提取后）
        if self.attention_type == 'se':
            self.attention = SE(ouc, reduction=reduction)
        elif self.attention_type == 'cbam':
            self.attention = CBAM(ouc, reduction_ratio=reduction)
        else:
            self.attention = nn.Identity()

        # 最终输�?
        self.final_conv = Conv(ouc, ouc, k=1)

    def forward(self, x_list):
        # 分支处理
        edge_feat = self.edge_proj(x_list[0])
        backbone_feat = self.backbone_proj(x_list[1])

        # 拼接融合
        x = torch.cat([edge_feat, backbone_feat], dim=1)

        # 融合与特征提�?
        x = self.conv_channel_fusion(x)
        x = self.conv_3x3(x)

        # 应用注意�?
        x = self.attention(x)

        # 输出
        x = self.final_conv(x)
        return x

class GatedConvEdgeFusion(nn.Module):
    def __init__(self, inc, ouc, mode='softmax', reduction=16):
        """
        Flexible fusion module between main and edge features.

        Args:
            inc (list): [main_channels, edge_channels]
            ouc (int): output channels
            mode (str): one of ['softmax', 'sigmoid', 'add', 'scalar', 'attention', 'transformer']
            reduction (int): reduction ratio for attention modules
        """
        super().__init__()
        self.c_main, self.c_edge = inc
        self.ouc = ouc
        self.mode = mode.lower()

        self.align_main = nn.Conv2d(self.c_main, ouc, kernel_size=1)
        self.align_edge = nn.Conv2d(self.c_edge, ouc, kernel_size=1)

        if self.mode == 'softmax':
            self.gate = nn.Sequential(
                nn.Conv2d(ouc * 2, 2, kernel_size=1),
                nn.Softmax(dim=1)
            )
        elif self.mode == 'sigmoid':
            self.gate = nn.Sequential(
                nn.Conv2d(ouc * 2, 2, kernel_size=1),
                nn.Sigmoid()
            )
        elif self.mode == 'scalar':
            self.alpha = nn.Parameter(torch.tensor(0.5))
        elif self.mode == 'attention':
            self.attn = CBAM(ouc * 2, reduction_ratio=reduction)
        elif self.mode == 'transformer':
            self.transformer = nn.TransformerEncoderLayer(d_model=ouc, nhead=4)
        elif self.mode == 'add':
            pass
        else:
            raise ValueError(f"Unsupported fusion mode: {mode}")

        self.output_proj = nn.Sequential(
            nn.Conv2d(ouc, ouc, kernel_size=1),
            nn.BatchNorm2d(ouc),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        main_feat, edge_feat = inputs
        # main_feat = self.align_main(main_feat)
        # edge_feat = self.align_edge(edge_feat)
        B, C, H, W = main_feat.shape

        if self.mode in ['softmax', 'sigmoid']:
            fusion_input = torch.cat([main_feat, edge_feat], dim=1)  # [B, 2C, H, W]
            weights = self.gate(fusion_input)
            out = weights[:, 0:1] * main_feat + weights[:, 1:2] * edge_feat

        elif self.mode == 'scalar':
            out = self.alpha * main_feat + (1 - self.alpha) * edge_feat

        elif self.mode == 'add':
            out = main_feat + edge_feat

        elif self.mode == 'attention':
            fusion_input = torch.cat([main_feat, edge_feat], dim=1)
            out = self.attn(fusion_input)

        elif self.mode == 'transformer':
            # Reshape to sequence: [B, C, H, W] �?[B, HW, C]
            seq = (main_feat + edge_feat).flatten(2).permute(0, 2, 1)
            seq = self.transformer(seq)
            out = seq.permute(0, 2, 1).view(B, C, H, W)

        return self.output_proj(out)


class DualPathEdgeFusion(nn.Module):
    def __init__(self, inc, ouc, mode='weighted', attention=None, reduction=16):
        """
        Improved Dual Path Fusion block.

        Args:
            inc (list): [main_channels, edge_channels]
            ouc (int): output channels
            mode (str): 'concat' | 'add' | 'weighted'
            attention (str or None): 'se' | 'cbam' | None
            reduction (int): attention reduction ratio
        """
        super().__init__()
        c_main, c_edge = inc
        self.mode = mode.lower()
        self.ouc = ouc

        # Align input dimensions
        self.main_proj = nn.Sequential(
            nn.Conv2d(c_main, ouc, kernel_size=1),
            nn.BatchNorm2d(ouc),
            nn.ReLU(inplace=True)
        )
        self.edge_proj = nn.Sequential(
            nn.Conv2d(c_edge, ouc, kernel_size=1),
            nn.BatchNorm2d(ouc),
            nn.ReLU(inplace=True)
        )

        # Optional attention
        if attention == 'se':
            self.attn = SE(ouc, reduction)
        elif attention == 'cbam':
            self.attn = CBAM(ouc, reduction)
        else:
            self.attn = nn.Identity()

        # Learnable fusion if requested
        if self.mode == 'weighted':
            self.alpha = nn.Parameter(torch.tensor(0.5))

        # Final fusion block (for concat or projection)
        if self.mode == 'concat':
            self.fusion = nn.Sequential(
                nn.Conv2d(2 * ouc, ouc, kernel_size=1),
                nn.BatchNorm2d(ouc),
                nn.ReLU(inplace=True)
            )
        elif self.mode in ['add', 'weighted']:
            self.fusion = nn.Identity()
        else:
            raise ValueError(f"Unsupported fusion mode: {self.mode}")

    def forward(self, inputs):
        # �?支持 list / tuple 输入
        if isinstance(inputs, (list, tuple)):
            main_feat, edge_feat = inputs
        # Align dimensions
        m = self.main_proj(main_feat)
        e = self.edge_proj(edge_feat)

        # Apply attention to edge
        e = self.attn(e)

        # Fusion
        if self.mode == 'concat':
            fused = self.fusion(torch.cat([m, e], dim=1))
        elif self.mode == 'add':
            fused = self.fusion(m + e)
        elif self.mode == 'weighted':
            fused = self.fusion(self.alpha * m + (1 - self.alpha) * e)

        return fused
