"""Base Model for Semantic Segmentation"""
import sys
sys.path.append('..')

import torch.nn as nn

from .mobilenetv3 import mobilenet_v3_small_1_0, mobilenet_v3_large_1_0


class BaseModel(nn.Module):
    def __init__(self, nclass, aux=False, backbone='mobilenetv3_small', pretrained_base=False, **kwargs):
        super(BaseModel, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.backbone = backbone

        if backbone == 'mobilenetv3_small':
            self.pretrained = mobilenet_v3_small_1_0(dilated=True, pretrained=pretrained_base, **kwargs)
        elif backbone == 'mobilenetv3_large':
            self.pretrained = mobilenet_v3_large_1_0(dilated=True, pretrained=pretrained_base, **kwargs)
        else:
            raise RuntimeError("Unknown backnone: {}".format(backbone))

    def base_forward(self, x):
        """forwarding pre-trained network"""
        if self.backbone in ['mobilenetv3_small', 'mobilenetv3_large']:
            x = self.pretrained.conv1(x)
            x = self.pretrained.layer1(x)
            c1 = self.pretrained.layer2(x)
            c2 = self.pretrained.layer3(c1)
            c3 = self.pretrained.layer4(c2)
            c4 = self.pretrained.layer5(c3)
        else:
            raise ValueError

        return c1, c2, c3, c4


if __name__ == '__main__':
    model = BaseModel(19)
