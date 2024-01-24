import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class AddCoordinates(object):

    r"""Coordinate Adder Module as defined in 'An Intriguing Failing of
    Convolutional Neural Networks and the CoordConv Solution'
    (https://arxiv.org/pdf/1807.03247.pdf).

    This module concatenates coordinate information (`x`, `y`, and `r`) with
    given input tensor.

    `x` and `y` coordinates are scaled to `[-1, 1]` range where origin is the
    center. `r` is the Euclidean distance from the center and is scaled to
    `[0, 1]`.

    Args:
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, (C_{in} + 2) or (C_{in} + 3), H_{in}, W_{in})`

    Examples:
        >>> coord_adder = AddCoordinates(True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = coord_adder(input)

        >>> coord_adder = AddCoordinates(True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = coord_adder(input)
    """

    def __init__(self, with_r=True, usegpu=True):
        '''Allows conv filters to know where they are in Cartesian space by adding extra,
        hard coded input channels that contatin coordinates of the data seen by the conv filter'''
        self.with_r = with_r
        self.usegpu = usegpu

    def __call__(self, image):
        batch_size, _, image_height, image_width = image.size()

        # Coordinates informations with range [-1, 1]
        # hard-coded (constant, untrained)
        y_coords = 2.0 * torch.arange(image_height).unsqueeze(
            1).expand(image_height, image_width) / (image_height - 1.0) - 1.0
        x_coords = 2.0 * torch.arange(image_width).unsqueeze(
            0).expand(image_height, image_width) / (image_width - 1.0) - 1.0

        # channel-wise concat (2 channels are added)
        coords = torch.stack((y_coords, x_coords), dim=0)

        # use r-coordinates (3 channels are added)
        if self.with_r:
            rs = ((y_coords ** 2) + (x_coords ** 2)) ** 0.5
            rs = rs / torch.max(rs)
            rs = torch.unsqueeze(rs, dim=0)
            coords = torch.cat((coords, rs), dim=0)

        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1)
        coords = Variable(coords)

        if self.usegpu:
            coords = coords.cuda()

        image = torch.cat((coords, image), dim=1)

        return image


class CoordConvNet(nn.Module):

    r"""Improves 2D Convolutions inside a ConvNet by processing extra
    coordinate information as defined in 'An Intriguing Failing of
    Convolutional Neural Networks and the CoordConv Solution'
    (https://arxiv.org/pdf/1807.03247.pdf).

    This module adds coordinate information to inputs of each 2D convolution
    module (`torch.nn.Conv2d`).

    Assumption: ConvNet Model must contain single `Sequential` container
    (`torch.nn.modules.container.Sequential`).

    Args:
        cnn_model: A ConvNet model that must contain single `Sequential`
            container (`torch.nn.modules.container.Sequential`).
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: Same as the input of the model.
        - Output: A list that contains all outputs (including
            intermediate outputs) of the model.

    Examples:
        >>> cnn_model = ...
        >>> cnn_model = CoordConvNet(cnn_model, True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> outputs = cnn_model(input)

        >>> cnn_model = ...
        >>> cnn_model = CoordConvNet(cnn_model, True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> outputs = cnn_model(input)
    """

    def __init__(self, cnn_model, with_r=True, usegpu=True):
        super(CoordConvNet, self).__init__()

        self.with_r = with_r

        self.cnn_model = cnn_model
        self.__get_model()
        self.__update_weights()

        self.coord_adder = AddCoordinates(self.with_r, usegpu)

    def __get_model(self):
        for module in list(self.cnn_model.modules()):
            if module.__class__ == torch.nn.modules.container.Sequential:
                self.cnn_model = module
                break

    def __update_weights(self):
        coord_channels = 2
        if self.with_r:
            coord_channels += 1

        for l in list(self.cnn_model.modules()):
            if l.__str__().startswith('Conv2d'):
                weights = l.weight.data  # conv filter weights

                out_channels, in_channels, k_height, k_width = weights.size()

                # Coord conv with weights connected to input coordinates
                # set by initialization or learning to zero
                # if weights are non-zero -> function will contain translation dependence
                coord_weights = torch.zeros(out_channels, coord_channels,
                                            k_height, k_width)

                # coordinate informations
                weights = torch.cat((coord_weights, weights), dim=1)
                weights = nn.Parameter(weights)

                l.weight = weights
                l.in_channels += coord_channels

    def __get_outputs(self, x):
        outputs = []
        for layer_name, layer in self.cnn_model._modules.items():
            if layer.__str__().startswith('Conv2d'):
                '''adds coordinate information to inputs of each 2D conv module'''
                x = self.coord_adder(x)
            x = layer(x)
            outputs.append(x)

        return outputs

    def forward(self, x):
        return self.__get_outputs(x)


class InstanceCounter(nn.Module):
    r"""Instance Counter Module. Basically, it is a convolutional network
    to count instances for a given feature map.

    Args:
        input_n_filters (int): Number of channels in the input image
        use_coordinates (bool, optional): If `True`, adds coordinate
            information to input image and hidden state. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, 1)`
    """

    def __init__(self, input_n_filters, use_coordinates=False, usegpu=True):
        super(InstanceCounter, self).__init__()

        self.input_n_filters = input_n_filters
        self.n_filters = 32
        self.use_coordinates = use_coordinates
        self.usegpu = usegpu

        self.__generate_cnn()

        self.output = nn.Sequential()
        self.output.add_module('linear', nn.Linear(self.n_filters,
                                                   1))
        self.output.add_module('sigmoid', nn.Sigmoid())

    def __generate_cnn(self):

        self.cnn = nn.Sequential()
        self.cnn.add_module('pool1', nn.MaxPool2d(2, stride=2))
        self.cnn.add_module('conv1', nn.Conv2d(self.input_n_filters,
                                               self.n_filters,
                                               kernel_size=(3, 3),
                                               stride=(1, 1),
                                               padding=(1, 1)))

        self.cnn.add_module('relu1', nn.ReLU())
        self.cnn.add_module('conv2', nn.Conv2d(self.n_filters,
                                               self.n_filters,
                                               kernel_size=(3, 3),
                                               stride=(1, 1),
                                               padding=(1, 1)))

        self.cnn.add_module('relu2', nn.ReLU())
        self.cnn.add_module('pool2', nn.MaxPool2d(2, stride=2))
        self.cnn.add_module('conv3', nn.Conv2d(self.n_filters,
                                               self.n_filters,
                                               kernel_size=(3, 3),
                                               stride=(1, 1),
                                               padding=(1, 1)))

        self.cnn.add_module('relu3', nn.ReLU())
        self.cnn.add_module('conv4', nn.Conv2d(self.n_filters,
                                               self.n_filters,
                                               kernel_size=(3, 3),
                                               stride=(1, 1),
                                               padding=(1, 1)))
        self.cnn.add_module('relu4', nn.ReLU())
        self.cnn.add_module('pool3', nn.AdaptiveAvgPool2d((1, 1)))
        # b, nf, 1, 1

        if self.use_coordinates:
            self.cnn = CoordConvNet(self.cnn, with_r=True, usegpu=self.usegpu)

    def forward(self, x):

        x = self.cnn(x)
        if self.use_coordinates:
            x = x[-1]
        x = x.squeeze(3).squeeze(2)
        x = self.output(x)

        return x


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

    
class ChannelGate(nn.Module):
    '''Generate 2 different(avg, pool) spatial context descriptors to refine input feature'''
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels

        # Shared MLP
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),  # reduction
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)  # restoration
            )
        
        self.pool_types = pool_types

    def forward(self, x):
        '''x: Input feature  (N, C, h, w)
           kernel_size of pooling operation = (h, w) -> to squeeze the spatial dimension'''
        channel_att_sum = None  # It should be MLP(AvgPool(x)) + MLP(MaxPool(x))
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # (N, C, 1, 1)
                channel_att_raw = self.mlp(avg_pool)  # (N, C)

            elif pool_type=='max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # (N, C, 1, 1)
                channel_att_raw = self.mlp(max_pool)  # (N, C)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw

            else:
                channel_att_sum = channel_att_sum + channel_att_raw  # (N, C) - Channel Attention Map
        
        # Sigmoid & Broad-casting (N, C) -> (N, C, 1) -> (N, C, 1, 1) -> (N, C, h, w)
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)

        # Feature Refinement
        return x * scale  # (N, C, h, w)

    
class ChannelPool(nn.Module):
    '''Apply max pooling & avg pooling along the channel axis and concatenate them
       to generate an efficient feature descriptor'''
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

    
class SpatialGate(nn.Module):
    '''Produce 2D spatial attention map to refine channel-refined feature (sequential)'''
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        '''x: channel-refined feature (sequential)'''
        x_compress = self.compress(x)  # (N, 2, h, w)
        x_out = self.spatial(x_compress)  # (N, 1, h, w) - Spatial Attention Map
        scale = torch.sigmoid(x_out)  # broadcasting

        return x * scale
    

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        '''x: Input feature (N, C, h, w)
           x_out: (N, c, h, w)'''
        x_out = self.ChannelGate(x)  # Channel-refinement
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)  # Spatial-refinement

        return x_out  # Refined feature


class double_conv(nn.Module):
    '''(Conv + B.N + ReLU) x 2 with Attention'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()

        # Conv block
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        # Attention module
        self.cbam = CBAM(out_ch)

    def forward(self, input):
        input = self.conv(input)
        output = self.cbam(input)  # feature-refinement

        return output


class inconv(nn.Module):
    '''Initial block of UNet with Attention'''
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)

        return x


class down(nn.Module):
    '''Down-sampling block with Attention'''
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)

        return x


class up(nn.Module):
    '''Up-sampling block with Attention'''
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        # bilinear=True -> up-sample with bilinear interpolation (rule-based)
        # bilinear=False -> up-sample with conv (trained, requires sufficient memory)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        '''x1 should be up-sampled'''
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x


class outconv(nn.Module):
    '''Last block of UNet with Attention'''
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        # self.conv = nn.Conv2d(in_ch, out_ch, 1)

        #self.cbam = CBAM(in_ch)

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, out_ch, 1),
        )


    def forward(self, x):
        #x = self.cbam(x)
        x = self.conv(x)

        return x


# class UNet_CBAM(nn.Module):
#     def __init__(self):
#         super(UNet_CBAM, self).__init__()
#         self.inc = inconv(3, 64)

#         self.down1 = down(64, 128)
#         self.down2 = down(128, 256)
#         self.down3 = down(256, 512)
#         self.down4 = down(512, 512)

#         self.up1 = up(1024, 256)
#         self.up2 = up(512, 128)
#         self.up3 = up(256, 64)
#         self.up4 = up(128, 64)

#         # Segmentation heads (parallel)
#         self.sem_out = outconv(64, 2)  # semantic classes=2 (f.g / b.g)
#         self.ins_out = outconv(64, 32)  # instance classes=32 (it should be enough large to represent all instances)
#         self.ins_cls_cnn = InstanceCounter(64, use_coordinates=True,
#                                            usegpu=True)

#     def forward(self, x):
#         x1 = self.inc(x)  # (N, 64, h, w)

#         x2 = self.down1(x1)  # (N, 128, h/2, w/2)
#         x3 = self.down2(x2)  # (N, 256, h/4, w/4)
#         x4 = self.down3(x3)  # (N, 512, h/8, w/8)
#         x5 = self.down4(x4)  # (N, 512, h/16, w/16)

#         x = self.up1(x5, x4)  # (N, 256, h/8, w/8)
#         x = self.up2(x, x3)  # (N, 128, h/4, w/4)
#         x = self.up3(x, x2)  # (N, 64, h/2, w/2)
#         x = self.up4(x, x1)  # (N, 64, h, w)

#         # heads
#         sem = self.sem_out(x)  # (N, 2, h, w)
#         ins = self.ins_out(x)  # (N, 32, h, w) - 32 dims embedding space
#         cnt = self.ins_cls_cnn(x)

#         return sem, ins, cnt


class UNet_CBAM_Deeper(nn.Module):
    '''Only added 1 down-path & corresponding up-path'''
    def __init__(self, max_objects=10, usegpu=True):
        super(UNet_CBAM_Deeper, self).__init__()
        self.inc = inconv(3, 64)

        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        self.down5 = down(1024, 1024)

        self.up1 = up(2048, 512)
        self.up2 = up(1024, 256)
        self.up3 = up(512, 128)
        self.up4 = up(256, 64)
        self.up5 = up(128, 64)

        # Segmentation heads
        self.sem_out = outconv(64, 2)  # semantic classes=2 (f.g / b.g)
        self.ins_out = outconv(64, max_objects)  # instance classes=32 (enoguh large to represent all instances)
        self.ins_cls_cnn = InstanceCounter(64, use_coordinates=True, usegpu=usegpu)

    def forward(self, x):
        x1 = self.inc(x)  # (N, 64, h, w)

        x2 = self.down1(x1)  # (N, 128, h/2, w/2)
        x3 = self.down2(x2)  # (N, 256, h/4, w/4)
        x4 = self.down3(x3)  # (N, 512, h/8, w/8)
        x5 = self.down4(x4)  # (N, 1024, h/16, w/16)
        x6 = self.down5(x5)  # (N, 1024, h/32, w/32)

        x = self.up1(x6, x5)  # (N, 512, h/16, w/16)
        x = self.up2(x, x4)  # (N, 256, h/8, w/8)
        x = self.up3(x, x3)  # (N, 128, h/4, w/4)
        x = self.up4(x, x2)  # (N, 64, h/2, w/2)
        x = self.up5(x, x1)  # (N, 64, h, w)

        sem = self.sem_out(x)  # (N, 2, h, w)
        ins = self.ins_out(x)  # (N, 32, h, w)
        cnt = self.ins_cls_cnn(x)

        return sem, ins, cnt


if __name__ == '__main__':
    # Test
    import torch
    from torchsummary import summary

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # cpu mode
    device = torch.device("cpu")
    model = UNet_CBAM_Deeper().to(device)
    x = torch.randn(1, 3, 512, 512).to(device)
    y = model(x)
    for i in y:
        print(i.size())

    summary(model, (3, 512, 512), device='cpu')
