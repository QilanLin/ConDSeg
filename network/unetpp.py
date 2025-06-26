import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class NestedUp(nn.Module):
    """Upscaling for UNet++ with nested skip connections"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, *xlist):
        x1 = self.up(x1)
        # Padding for each of the skip connections
        x_all = [x1]  # Starting with the upsampled feature
        
        for x2 in xlist:
            if x2 is not None:
                # Apply padding to x1 to match x2's dimensions
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]
                
                # Only pad if there's a difference
                if diffX > 0 or diffY > 0:
                    x1_padded = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                       diffY // 2, diffY - diffY // 2])
                else:
                    x1_padded = x1
                
                x_all.append(x2)  # Add the skip connection
        
        # Concatenate all features
        try:
            x = torch.cat(x_all, dim=1)
            return self.conv(x)
        except RuntimeError as e:
            # Print debug info
            print("Error in NestedUp forward:")
            print(f"x1 shape: {x1.shape}")
            for i, x in enumerate(xlist):
                print(f"xlist[{i}] shape: {x.shape}")
            raise e


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, deep_supervision=False, bilinear=True):
        super(UNetPlusPlus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.deep_supervision = deep_supervision
        self.bilinear = bilinear

        # Initial convolution block
        self.inc = DoubleConv(n_channels, 64)
        
        # Downsampling path
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # First layer of skip connections (X^1_j)
        self.x01 = NestedUp(64 + 128, 64)  # Connects X00->X01
        self.x11 = NestedUp(128 + 256, 128)  # Connects X10->X11
        self.x21 = NestedUp(256 + 512, 256)  # Connects X20->X21
        self.x31 = NestedUp(512 + 1024//factor, 512)  # Connects X30->X31
        
        # Second layer of skip connections (X^2_j)
        self.x02 = NestedUp(128 + 64 + 64, 64)  # x11(128) + x00(64) + x01(64) 
        self.x12 = NestedUp(256 + 128 + 128, 128)  # x21(256) + x10(128) + x11(128)
        self.x22 = NestedUp(512 + 256 + 256, 256)  # x31(512) + x20(256) + x21(256)
        
        # Third layer of skip connections (X^3_j)
        self.x03 = NestedUp(128 + 64 + 64 + 64, 64)  # x12(128) + x00(64) + x01(64) + x02(64)
        self.x13 = NestedUp(256 + 128 + 128 + 128, 128)  # x22(256) + x10(128) + x11(128) + x12(128)
        
        # Fourth (last) layer of skip connections (X^4_j)
        self.x04 = NestedUp(128 + 64 + 64 + 64 + 64, 64)  # x13(128) + x00(64) + x01(64) + x02(64) + x03(64)
        
        # Output convolutions for deep supervision
        self.out1 = OutConv(64, n_classes)
        if deep_supervision:
            self.out2 = OutConv(64, n_classes)
            self.out3 = OutConv(64, n_classes)
            self.out4 = OutConv(64, n_classes)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder (downsampling) path
        x00 = self.inc(x)       # Level 0 features
        x10 = self.down1(x00)   # Level 1 features
        x20 = self.down2(x10)   # Level 2 features
        x30 = self.down3(x20)   # Level 3 features
        x40 = self.down4(x30)   # Level 4 features (bottom)
        
        # Decoder path with nested skip connections
        
        # First layer of skip connections
        x01 = self.x01(x10, x00)
        x11 = self.x11(x20, x10)
        x21 = self.x21(x30, x20)
        x31 = self.x31(x40, x30)
        
        # Make sure all inputs are padded correctly before concatenation
        # Second layer of skip connections
        try:
            x02 = self.x02(x11, x00, x01)
            x12 = self.x12(x21, x10, x11)
            x22 = self.x22(x31, x20, x21)
            
            # Third layer of skip connections
            x03 = self.x03(x12, x00, x01, x02)
            x13 = self.x13(x22, x10, x11, x12)
            
            # Fourth layer of skip connections
            x04 = self.x04(x13, x00, x01, x02, x03)
        except RuntimeError as e:
            print("Error in UNetPlusPlus forward:")
            print(f"x00: {x00.shape}, x01: {x01.shape}")
            print(f"x10: {x10.shape}, x11: {x11.shape}")
            print(f"x20: {x20.shape}, x21: {x21.shape}")
            print(f"x30: {x30.shape}, x31: {x31.shape}")
            print(f"x40: {x40.shape}")
            raise e
        
        # Output prediction
        logits = self.out1(x04)
        mask_pred = self.sigmoid(logits)
        
        # For compatibility with existing code structure
        return mask_pred, mask_pred, mask_pred, mask_pred


if __name__ == "__main__":
    # Test forward pass
    model = UNetPlusPlus(n_channels=3, n_classes=1)
    x = torch.randn(2, 3, 256, 256)
    output = model(x)
    print(output[0].shape)  # Should be [2, 1, 256, 256] 