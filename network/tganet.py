import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
    
    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpBlock, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Padding if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, 
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class GlobalAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(GlobalAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average pooling
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_out).view(b, c, 1, 1)
        
        # Max pooling
        max_out = self.max_pool(x).view(b, c)
        max_out = self.fc(max_out).view(b, c, 1, 1)
        
        # Combine both features
        y = self.sigmoid(avg_out + max_out)
        
        return x * y.expand_as(x)


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Generate attention along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and process
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        
        return x * self.sigmoid(y)


class TripletAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(TripletAttentionModule, self).__init__()
        self.channel_attention = GlobalAttentionModule(in_channels)
        self.spatial_attention = SpatialAttentionModule()
        self.fusion = ConvBlock(in_channels * 2, in_channels)
    
    def forward(self, x):
        # Channel attention path
        channel_out = self.channel_attention(x)
        
        # Spatial attention path
        spatial_out = self.spatial_attention(x)
        
        # Fusion of both paths
        combined = torch.cat([channel_out, spatial_out], dim=1)
        out = self.fusion(combined)
        
        return out


class TGANet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(TGANet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = nn.Sequential(
            ConvBlock(n_channels, 64),
            ConvBlock(64, 64)
        )
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(512, 1024 // factor)
        
        # Bottleneck Triplet Attention
        self.triplet_attn = TripletAttentionModule(1024 // factor)
        
        # Decoder with skip connections
        self.up1 = UpBlock(1024, 512 // factor)
        self.up2 = UpBlock(512, 256 // factor)
        self.up3 = UpBlock(256, 128 // factor)
        self.up4 = UpBlock(128, 64)
        
        # Final output
        self.outc = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Apply triplet attention at bottleneck
        x5 = self.triplet_attn(x5)
        
        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Final output
        output = self.outc(x)
        
        # For compatibility with existing code structure
        return output, output, output, output


if __name__ == '__main__':
    # Test forward pass
    model = TGANet(n_channels=3, n_classes=1)
    x = torch.randn(2, 3, 256, 256)
    output = model(x)
    print(output[0].shape)  # Should be [2, 1, 256, 256] 