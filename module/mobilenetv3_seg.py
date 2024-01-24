"""MobileNet3 for Semantic Segmentation"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchstat import stat

from .base import BaseModel

__all__ = ['MobileNetV3Seg']


class MobileNetV3Seg(BaseModel):
    def __init__(self, nclass, aux=False, backbone='mobilenetv3_large', **kwargs):
        super(MobileNetV3Seg, self).__init__(nclass, aux, backbone, **kwargs)
        mode = backbone.split('_')[-1]
        self.head = _Head(nclass, mode, **kwargs)
        if aux:
            inter_channels = 40 if mode == 'large' else 24
            self.auxlayer = nn.Conv2d(inter_channels, nclass, 1)

    def forward(self, x):
        size = x.size()[2:]
        _, c2, _, c4 = self.base_forward(x)
        outputs = list()
        x = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)

        # if self.aux:
        #     auxout = self.auxlayer(c2)
        #     auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
        #     outputs.append(auxout)

        return tuple(outputs)


class _Head(nn.Module):
    def __init__(self, nclass, mode='small', norm_layer=nn.BatchNorm2d, **kwargs):
        super(_Head, self).__init__()
        in_channels = 960 if mode == 'large' else 576
        self.lr_aspp = LRASPP(in_channels, norm_layer, **kwargs)
        self.project = nn.Conv2d(128, nclass, 1)

    def forward(self, x):
        x = self.lr_aspp(x)
        return self.project(x)


class LRASPP(nn.Module):
    """Lite R-ASPP"""

    def __init__(self, in_channels, norm_layer, **kwargs):
        super(LRASPP, self).__init__()
        out_channels = 128
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )
        self.b1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat2 = F.interpolate(feat2, size, mode='bilinear', align_corners=True)
        x = feat1 * feat2
        return x


if __name__ == '__main__':
    NUM_CLASS = 10
    crop = 768
    model = MobileNetV3Seg(NUM_CLASS, backbone='mobilenetv3_large')
    x = torch.rand(1, 3, crop, crop)
    y = model(x)
    print(y[0].size())
    # save as onnx
    model.eval()
    torch.onnx.export(model, x, "mobilenetv3_large.onnx")
