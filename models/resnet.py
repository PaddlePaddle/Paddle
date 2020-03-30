from __future__ import division
from __future__ import print_function

import math
import paddle.fluid as fluid

from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.container import Sequential

from model import Model
from .download import get_weights_path

__all__ = ['ResNet', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet50': ('https://paddle-hapi.bj.bcebos.com/models/resnet50.pdparams',
                 '0884c9087266496c41c60d14a96f8530')
}


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False)

        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        x = self._conv(inputs)
        x = self._batch_norm(x)

        return x


class BasicBlock(fluid.dygraph.Layer):
    expansion = 1

    def __init__(self, num_channels, num_filters, stride, shortcut=True):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters, stride, shortcut=True):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu')
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        x = self.conv0(inputs)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        x = fluid.layers.elementwise_add(x=short, y=conv2)

        layer_helper = LayerHelper(self.full_name(), act='relu')
        return layer_helper.append_activation(x)
        # return fluid.layers.relu(x)


class ResNet(Model):
    def __init__(self, Block, depth=50, num_classes=1000):
        super(ResNet, self).__init__()

        layer_config = {
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }
        assert depth in layer_config.keys(), \
            "supported depth are {} but input layer is {}".format(
                layer_config.keys(), depth)

        layers = layer_config[depth]
        num_in = [64, 256, 512, 1024]
        num_out = [64, 128, 256, 512]

        self.conv = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu')
        self.pool = Pool2D(
            pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

        self.layers = []
        for idx, num_blocks in enumerate(layers):
            blocks = []
            shortcut = False
            for b in range(num_blocks):
                block = Block(
                    num_channels=num_in[idx] if b == 0 else num_out[idx] * 4,
                    num_filters=num_out[idx],
                    stride=2 if b == 0 and idx != 0 else 1,
                    shortcut=shortcut)
                blocks.append(block)
                shortcut = True
            layer = self.add_sublayer("layer_{}".format(idx),
                                      Sequential(*blocks))
            self.layers.append(layer)

        self.global_pool = Pool2D(
            pool_size=7, pool_type='avg', global_pooling=True)

        stdv = 1.0 / math.sqrt(2048 * 1.0)
        self.fc_input_dim = num_out[-1] * 4 * 1 * 1
        self.fc = Linear(
            self.fc_input_dim,
            num_classes,
            act='softmax',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv)))

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.pool(x)
        for layer in self.layers:
            x = layer(x)
        x = self.global_pool(x)
        x = fluid.layers.reshape(x, shape=[-1, self.fc_input_dim])
        x = self.fc(x)
        return x


def _resnet(arch, Block, depth, pretrained):
    model = ResNet(Block, depth)
    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path(model_urls[arch][0],
                                       model_urls[arch][1])
        assert weight_path.endswith(
            '.pdparams'), "suffix of weight must be .pdparams"
        model.load(weight_path[:-9])
    return model


def resnet50(pretrained=False):
    return _resnet('resnet50', BottleneckBlock, 50, pretrained)


def resnet101(pretrained=False):
    return _resnet('resnet101', BottleneckBlock, 101, pretrained)


def resnet152(pretrained=False):
    return _resnet('resnet152', BottleneckBlock, 152, pretrained)
