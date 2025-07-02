import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.maxpool_conv(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpSample, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Padding if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerBlock(dim, heads, dim_head, mlp_dim, dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class BoundaryModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 使用Sobel滤波器检测边界
        self.sobel_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels)
        
        # 初始化Sobel滤波器权重
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        sobel_kernel_x = sobel_kernel_x.repeat(in_channels, 1, 1, 1)
        sobel_kernel_y = sobel_kernel_y.repeat(in_channels, 1, 1, 1)
        
        with torch.no_grad():
            self.sobel_x.weight = nn.Parameter(sobel_kernel_x)
            self.sobel_y.weight = nn.Parameter(sobel_kernel_y)
            
            # 冻结Sobel滤波器权重
            self.sobel_x.weight.requires_grad = False
            self.sobel_y.weight.requires_grad = False
            
        # 融合原始特征和边界特征
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # 提取边界信息
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        
        # 合并原始特征和边界特征
        fusion = torch.cat([x, edge_x, edge_y], dim=1)
        return self.fusion(fusion)


class XBoundFormer(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(XBoundFormer, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # 特征维度配置
        feature_dims = [64, 128, 256, 512, 1024 // (2 if bilinear else 1)]
        transformer_dim = feature_dims[-1]
        
        # CNN编码器
        self.inc = ConvBlock(n_channels, feature_dims[0])
        self.down1 = DownSample(feature_dims[0], feature_dims[1])
        self.down2 = DownSample(feature_dims[1], feature_dims[2])
        self.down3 = DownSample(feature_dims[2], feature_dims[3])
        self.down4 = DownSample(feature_dims[3], feature_dims[4])
        
        # 边界感知模块
        self.boundary1 = BoundaryModule(feature_dims[0], feature_dims[0])
        self.boundary2 = BoundaryModule(feature_dims[1], feature_dims[1])
        self.boundary3 = BoundaryModule(feature_dims[2], feature_dims[2])
        self.boundary4 = BoundaryModule(feature_dims[3], feature_dims[3])
        
        # Transformer模块
        # 线性投影用于调整维度
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(feature_dims[4], transformer_dim, kernel_size=1),
            nn.BatchNorm2d(transformer_dim),
            nn.ReLU(inplace=True)
        )
        
        # Transformer编码器
        self.transformer = Transformer(
            dim=transformer_dim,
            depth=3,
            heads=8,
            dim_head=64,
            mlp_dim=transformer_dim * 4,
            dropout=0.1
        )
        
        # 将Transformer输出转回空间特征
        self.from_patch_embedding = nn.Sequential(
            nn.Conv2d(transformer_dim, feature_dims[4], kernel_size=1),
            nn.BatchNorm2d(feature_dims[4]),
            nn.ReLU(inplace=True)
        )
        
        # 解码器部分
        self.up1 = UpSample(feature_dims[4] + feature_dims[3], feature_dims[3], bilinear)
        self.up2 = UpSample(feature_dims[3] + feature_dims[2], feature_dims[2], bilinear)
        self.up3 = UpSample(feature_dims[2] + feature_dims[1], feature_dims[1], bilinear)
        self.up4 = UpSample(feature_dims[1] + feature_dims[0], feature_dims[0], bilinear)
        
        # 输出层
        self.outc = nn.Sequential(
            nn.Conv2d(feature_dims[0], n_classes, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        try:
            # 编码器路径
            x1 = self.inc(x)
            b1 = self.boundary1(x1)
            
            x2 = self.down1(x1)
            b2 = self.boundary2(x2)
            
            x3 = self.down2(x2)
            b3 = self.boundary3(x3)
            
            x4 = self.down3(x3)
            b4 = self.boundary4(x4)
            
            x5 = self.down4(x4)
            
            # Transformer处理
            bottleneck = self.to_patch_embedding(x5)
            
            # 将特征图转换为序列
            b, c, h, w = bottleneck.shape
            tokens = bottleneck.flatten(2).transpose(1, 2)  # [b, h*w, c]
            
            # 应用Transformer
            transformed = self.transformer(tokens)
            
            # 将序列转回特征图
            bottleneck_out = transformed.transpose(1, 2).reshape(b, c, h, w)
            bottleneck_out = self.from_patch_embedding(bottleneck_out)
            
            # 解码器路径，融合边界特征
            x = self.up1(bottleneck_out, b4)
            x = self.up2(x, b3)
            x = self.up3(x, b2)
            x = self.up4(x, b1)
            
            # 最终预测
            output = self.outc(x)
            
            # 为了与现有代码结构兼容
            return output, output, output, output
            
        except Exception as e:
            import traceback
            print(f"XBoundFormer forward error: {str(e)}")
            traceback.print_exc()
            raise e


if __name__ == '__main__':
    # 测试前向传播
    model = XBoundFormer(n_channels=3, n_classes=1)
    x = torch.randn(2, 3, 256, 256)
    output = model(x)
    print(output[0].shape)  # 应为 [2, 1, 256, 256] 