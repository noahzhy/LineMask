import torch
import torch.nn as nn

from .shufflenetv2 import ShuffleNetV2
from .custom_layers import *


class Detector(nn.Module):
    def __init__(self, category_num=3, max_lines=10, load_param=False):
        super(Detector, self).__init__()

        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = [-1, 24, 48, 96, 192]
        self.backbone = ShuffleNetV2(
            self.stage_repeats, self.stage_out_channels, load_param)

        self.upsample3to2 = upsample(
            self.stage_out_channels[-1], self.stage_out_channels[-2])
        self.upsample2to1 = upsample(
            self.stage_out_channels[-2], self.stage_out_channels[-3])
        self.conv1x1_p2 = nn.Conv2d(
            self.stage_out_channels[-2], self.stage_out_channels[-2], kernel_size=1, stride=1, padding=0)
        self.conv1x1_p1 = nn.Conv2d(
            self.stage_out_channels[-3], self.stage_out_channels[-3], kernel_size=1, stride=1, padding=0)
        self.last_conv = dw_conv3(
            self.stage_out_channels[-3], self.stage_out_channels[-3])

        self.detect_head = DetectHead(
            self.stage_out_channels[-3], category_num)
        self.seg = nn.Sequential(
            # 8x upsample
            nn.UpsamplingBilinear2d(scale_factor=8),
            nn.Conv2d(
                self.stage_out_channels[-3], max_lines, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        # 1x1 conv
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(3, max_lines, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        P1, P2, P3 = self.backbone(inputs)
        skip = self.conv1x1(inputs)
        x = P3
        x = self.upsample3to2(x) + self.conv1x1_p2(P2)
        x = self.upsample2to1(x) + self.conv1x1_p1(P1)
        x = self.last_conv(x)
        det = self.detect_head(x)
        seg = self.seg(x) + skip
        return det, seg


if __name__ == "__main__":
    model = Detector()
    test_data = torch.rand(1, 3, 352, 352)
    torch.onnx.export(model,                    # model being run
                      # model input (or a tuple for multiple inputs)
                      test_data,
                      # where to save the model (can be a file or file-like object)
                      "./test.onnx",
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True)  # whether to execute constant folding for optimization
    model.eval()
    # inference
    torch_out = model(test_data)
    for h in torch_out:
        # check min and max value
        print(h.min(), h.max())
