import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple Residual Block with fewer parameters"""
    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        return out + x


class BlindSR(nn.Module):
    """Lightweight Blind Super-Resolution Network for 4x upscaling.
    
    Takes 3×128×128 LQ images and outputs 3×512×512 HQ images.
    Uses fewer parameters than the original version.
    """
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16):
        super(BlindSR, self).__init__()
        
        # First convolution layer
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        
        # Residual blocks
        res_blocks = []
        for _ in range(nb):
            res_blocks.append(ResidualBlock(nf))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Global skip connection conv
        self.conv_after_body = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # Upsampling layers (4x = 2x + 2x)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # Final output layer
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # First feature extraction
        feat = self.conv_first(x)
        body_feat = feat
        
        # Residual blocks
        body_feat = self.res_blocks(body_feat)
        
        # Global skip connection
        body_feat = self.conv_after_body(body_feat)
        feat = feat + body_feat
        
        # Upsampling (4x)
        feat = self.lrelu(self.upconv1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.upconv2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        
        # Final convolution
        out = self.conv_last(feat)
        
        return out






if __name__ == "__main__":
    max_parms = 2276356

    model = BlindSR()
    parms = sum(p.numel() for p in model.parameters())

    print(parms)
    print("/")
    print(max_parms)
