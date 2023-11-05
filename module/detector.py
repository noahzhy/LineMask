import torch
import torch.nn as nn

from shufflenetv2 import ShuffleNetV2
from custom_layers import *


class Detector(nn.Module):
    def __init__(self, category_num, line_max=10, load_param=False):
        super(Detector, self).__init__()

        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = [-1, 24, 48, 96, 192]
        self.backbone = ShuffleNetV2(self.stage_repeats, self.stage_out_channels, load_param)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.SPP = SPP(sum(self.stage_out_channels[-3:]), self.stage_out_channels[-2])

        # 1x1 conv2d, unit_out is line_max
        self.conv_1x1 = Conv1x1(self.stage_out_channels[-2], line_max)
        # same output channel as conv_1x1
        self.skip_conv = Conv1x1(self.stage_out_channels[-3], line_max)

        self.detect_head = DetectHead(self.stage_out_channels[-2], category_num)
        # attn
        self.attn = Attention(self.stage_out_channels[-2], self.stage_out_channels[-2])

        # 8x up
        self.up_8x = nn.Upsample(scale_factor=8, mode='nearest')

    def forward(self, x):
        P1, P2, P3 = self.backbone(x)
        P = torch.cat((self.avg_pool(P1), P2, self.upsample(P3)), dim=1)
        y = self.SPP(P)
        det = self.detect_head(y)
        # det shape: [batch_size, 6, 22, 22]
        attn = self.attn(y)
        conv1x1 = self.conv_1x1(attn)
        conv1x1 = conv1x1 + self.skip_conv(P1)
        seg = self.up_8x(conv1x1)
        # seg shape: [batch_size, line_max, 352, 352]
        return det, seg


if __name__ == "__main__":
    model = Detector(10, 10, False)
    test_data = torch.rand(1, 3, 352, 352)
    # torch.onnx.export(model,                    # model being run
    #                  test_data,                 # model input (or a tuple for multiple inputs)
    #                  "./test.onnx",             # where to save the model (can be a file or file-like object)
    #                  export_params=True,        # store the trained parameter weights inside the model file
    #                  opset_version=11,          # the ONNX version to export the model to
    #                  do_constant_folding=True)  # whether to execute constant folding for optimization
    y = model(test_data)
    for idx, i in enumerate(y):
        print(idx, i.shape)

    # calculate the flops
    from thop import profile
    from thop import clever_format

    flops, params = profile(model, inputs=(test_data,))
    flops, params = clever_format([flops, params], "%.2f")
    print(flops, params)
