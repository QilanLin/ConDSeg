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
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = ConvBlock(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Padding if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class LayerNorm2d(nn.Module):
    """2D版本的LayerNorm，能处理4D特征图[B, C, H, W]"""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # 输入 [B, C, H, W]，转置为 [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # 转回 [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dim * mlp_ratio, dropout=dropout)
        
    def forward(self, x):
        """只接受序列形式的输入 [B, N, C]"""
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class BoundaryAwareModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BoundaryAwareModule, self).__init__()
        
        # Sobel filters for boundary detection
        self.sobel_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels)
        
        # Initialize Sobel filters
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        sobel_kernel_x = sobel_kernel_x.expand(in_channels, 1, 3, 3)
        sobel_kernel_y = sobel_kernel_y.expand(in_channels, 1, 3, 3)
        
        with torch.no_grad():
            self.sobel_x.weight = nn.Parameter(sobel_kernel_x)
            self.sobel_y.weight = nn.Parameter(sobel_kernel_y)
            
            # Freeze the sobel kernels
            self.sobel_x.weight.requires_grad = False
            self.sobel_y.weight.requires_grad = False
            
        # Boundary enhancement
        self.enhance = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Extract boundary information
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        
        # Combine original and edge features
        boundary_features = torch.cat([x, edge_x, edge_y], dim=1)
        enhanced = self.enhance(boundary_features)
        return enhanced


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0.):
        super(CrossAttentionBlock, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x, context):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        kv = self.kv(context).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class XBoundFormer(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(XBoundFormer, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # CNN Encoder
        self.inc = ConvBlock(n_channels, 64)
        self.down1 = DownSample(64, 128)
        self.down2 = DownSample(128, 256)
        self.down3 = DownSample(256, 512)
        
        factor = 2 if bilinear else 1
        self.down4 = DownSample(512, 1024 // factor)
        
        # Transformer bottleneck
        self.patch_size = 2  # For tokenization at bottleneck
        self.embed_dim = 1024 // factor
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim=self.embed_dim, num_heads=8, dropout=0.1),
            TransformerBlock(dim=self.embed_dim, num_heads=8, dropout=0.1),
            TransformerBlock(dim=self.embed_dim, num_heads=8, dropout=0.1),
        ])
        
        # Boundary-aware modules
        self.boundary1 = BoundaryAwareModule(64, 64)
        self.boundary2 = BoundaryAwareModule(128, 128)
        self.boundary3 = BoundaryAwareModule(256, 256)
        self.boundary4 = BoundaryAwareModule(512, 512)
        
        # Cross-attention modules for integrating boundary information
        self.cross_attn1 = CrossAttentionBlock(64, num_heads=4)
        self.cross_attn2 = CrossAttentionBlock(128, num_heads=4)
        self.cross_attn3 = CrossAttentionBlock(256, num_heads=4)
        self.cross_attn4 = CrossAttentionBlock(512, num_heads=4)
        
        # CNN Decoder with skip connections
        self.up1 = UpSample(1024, 512 // factor)
        self.up2 = UpSample(512, 256 // factor)
        self.up3 = UpSample(256, 128 // factor)
        self.up4 = UpSample(128, 64)
        
        # Output layer
        self.outc = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.Sigmoid()
        )
        
    def tokenize(self, x):
        """将特征图转换为token序列"""
        b, c, h, w = x.shape
        # 确保h和w能被patch_size整除
        p = self.patch_size
        padding_h = (p - h % p) % p
        padding_w = (p - w % p) % p
        if padding_h > 0 or padding_w > 0:
            x = F.pad(x, [0, padding_w, 0, padding_h])
            h, w = h + padding_h, w + padding_w
            
        # 重塑为token序列
        x = x.reshape(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # [b, h//p, w//p, p, p, c]
        x = x.reshape(b, (h // p) * (w // p), p * p * c)  # [b, hw//p^2, p^2*c]
        return x, h, w
        
    def untokenize(self, x, h, w):
        """将token序列转换回特征图"""
        b, hw, d = x.shape
        p = self.patch_size
        
        # 计算padding后的尺寸
        h_padded = (h + p - 1) // p * p
        w_padded = (w + p - 1) // p * p
        
        # 重塑回特征图
        x = x.reshape(b, h_padded // p, w_padded // p, p, p, d // (p*p))
        x = x.permute(0, 5, 1, 3, 2, 4)  # [b, c, h//p, p, w//p, p]
        x = x.reshape(b, d // (p*p), h_padded, w_padded)
        
        # 如果之前有padding，现在去除
        if h_padded > h or w_padded > w:
            x = x[:, :, :h, :w]
            
        return x
    
    def forward(self, x):
        try:
            # Encoder path with boundary-aware modules
            x1 = self.inc(x)
            b1 = self.boundary1(x1)
            
            x2 = self.down1(x1)
            b2 = self.boundary2(x2)
            
            x3 = self.down2(x2)
            b3 = self.boundary3(x3)
            
            x4 = self.down3(x3)
            b4 = self.boundary4(x4)
            
            x5 = self.down4(x4)
            
            # Apply transformer at bottleneck
            _, c, h, w = x5.shape
            x5_tokens, orig_h, orig_w = self.tokenize(x5)
            
            for block in self.transformer_blocks:
                x5_tokens = block(x5_tokens)
            
            x5 = self.untokenize(x5_tokens, orig_h, orig_w)
            
            # Decoder path with cross-attention for boundary integration
            x = self.up1(x5, x4)
            
            # Apply cross-attention between decoder features and boundary features
            # Convert spatial features to sequence for cross-attention
            b, c, h, w = x.shape
            x_seq = x.reshape(b, c, h*w).permute(0, 2, 1)  # [b, hw, c]
            b4_seq = b4.reshape(b, b4.size(1), -1).permute(0, 2, 1)  # [b, hw, c]
            x_seq = self.cross_attn4(x_seq, b4_seq)
            x = x_seq.permute(0, 2, 1).reshape(b, c, h, w)
            
            x = self.up2(x, x3)
            b, c, h, w = x.shape
            x_seq = x.reshape(b, c, h*w).permute(0, 2, 1)  # [b, hw, c]
            b3_seq = b3.reshape(b, b3.size(1), -1).permute(0, 2, 1)  # [b, hw, c]
            x_seq = self.cross_attn3(x_seq, b3_seq)
            x = x_seq.permute(0, 2, 1).reshape(b, c, h, w)
            
            x = self.up3(x, x2)
            b, c, h, w = x.shape
            x_seq = x.reshape(b, c, h*w).permute(0, 2, 1)  # [b, hw, c]
            b2_seq = b2.reshape(b, b2.size(1), -1).permute(0, 2, 1)  # [b, hw, c]
            x_seq = self.cross_attn2(x_seq, b2_seq)
            x = x_seq.permute(0, 2, 1).reshape(b, c, h, w)
            
            x = self.up4(x, x1)
            b, c, h, w = x.shape
            x_seq = x.reshape(b, c, h*w).permute(0, 2, 1)  # [b, hw, c]
            b1_seq = b1.reshape(b, b1.size(1), -1).permute(0, 2, 1)  # [b, hw, c]
            x_seq = self.cross_attn1(x_seq, b1_seq)
            x = x_seq.permute(0, 2, 1).reshape(b, c, h, w)
            
            # Final prediction
            output = self.outc(x)
            
            # For compatibility with existing code structure
            return output, output, output, output
            
        except Exception as e:
            print(f"XBoundFormer forward error: {str(e)}")
            # 重新抛出异常以便跟踪
            raise e


if __name__ == '__main__':
    # Test forward pass
    model = XBoundFormer(n_channels=3, n_classes=1)
    x = torch.randn(2, 3, 256, 256)
    output = model(x)
    print(output[0].shape)  # Should be [2, 1, 256, 256] 