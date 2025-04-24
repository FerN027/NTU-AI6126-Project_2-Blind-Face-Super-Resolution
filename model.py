import torch
import torch.nn as nn
import torch.nn.functional as F


class CALayer(nn.Module):
    """Channel Attention (squeeze & excite)"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.avg_pool(x)       # [B, C, 1, 1]
        w = self.fc(w)             # [B, C, 1, 1]
        return x * w

class RCAB(nn.Module):
    """Residual Channel-Attention Block"""
    def __init__(self, nf, reduction=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            CALayer(nf, reduction)
        )
    def forward(self, x):
        return x + self.body(x)

class RCAN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=20, scale=4):
        super().__init__()
        # head
        self.head = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        # body: nb RCABs + one conv
        body = [RCAB(nf) for _ in range(nb)]
        body.append(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
        self.body = nn.Sequential(*body)
        # tail (upsampling)
        tail = []
        for _ in range(int(scale // 2)):
            tail += [
                nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            ]
        tail.append(nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True))
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        x = self.tail(x)
        return x


if __name__ == "__main__":
    max_parms = 2276356

    model = RCAN()
    parms = sum(p.numel() for p in model.parameters())

    print(parms)
    print("/")
    print(max_parms)
