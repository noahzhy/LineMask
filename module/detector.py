import torch
import torch.nn as nn

from shufflenetv2 import ShuffleNetV2
from custom_layers import *


class Detector(nn.Module):
    def __init__(self, category_num=3, max_lines=10, load_param=False):
        super(Detector, self).__init__()

        self.stage_repeats = [4, 8, 4]
        self.outs = [-1, 24, 48, 96, 192]
        self.backbone = ShuffleNetV2(self.stage_repeats, self.outs, load_param)

        self.upsample3to2 = upsample(self.outs[-1], self.outs[-2])
        self.upsample2to1 = upsample(self.outs[-2], self.outs[-3])
        self.conv1x1_p2 = nn.Conv2d(self.outs[-2], self.outs[-2], kernel_size=1, stride=1, padding=0)
        self.conv1x1_p1 = nn.Conv2d(self.outs[-3], self.outs[-3], kernel_size=1, stride=1, padding=0)
        self.last_conv = dw_conv3(self.outs[-3], self.outs[-3])

        self.detect_head = DetectHead(self.outs[-3], category_num)
        self.seg = nn.Sequential(
            # 8x upsample
            nn.UpsamplingBilinear2d(scale_factor=8),
            nn.Conv2d(self.outs[-3], max_lines, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        # 1x1 conv
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(3, max_lines, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        P1, P2, P3 = self.backbone(inputs)
        x = self.upsample3to2(P3) + self.conv1x1_p2(P2)
        x = self.upsample2to1(x) + self.conv1x1_p1(P1)
        x = self.last_conv(x)
        seg = self.seg(x) + self.conv1x1(inputs)
        return seg


if __name__ == "__main__":
    model = Detector()
    x = torch.rand(1, 3, 352, 352)
    torch.onnx.export(model, x, "./test.onnx",
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True)  # whether to execute constant folding for optimization
    model.eval()
    # inference
    torch_out = model(x)
    for h in torch_out:
        # check min and max value
        print(h.min(), h.max())
