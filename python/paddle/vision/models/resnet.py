# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle import nn
from paddle.utils.download import get_weights_path_from_url

__all__ = []

model_urls = {
    'resnet18': (
        'https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams',
        'cf548f46534aa3560945be4b95cd11c4',
    ),
    'resnet34': (
        'https://paddle-hapi.bj.bcebos.com/models/resnet34.pdparams',
        '8d2275cf8706028345f78ac0e1d31969',
    ),
    'resnet50': (
        'https://paddle-hapi.bj.bcebos.com/models/resnet50.pdparams',
        'ca6f485ee1ab0492d38f323885b0ad80',
    ),
    'resnet101': (
        'https://paddle-hapi.bj.bcebos.com/models/resnet101.pdparams',
        '02f35f034ca3858e1e54d4036443c92d',
    ),
    'resnet152': (
        'https://paddle-hapi.bj.bcebos.com/models/resnet152.pdparams',
        '7ad16a2f1e7333859ff986138630fd7a',
    ),
    'resnext50_32x4d': (
        'https://paddle-hapi.bj.bcebos.com/models/resnext50_32x4d.pdparams',
        'dc47483169be7d6f018fcbb7baf8775d',
    ),
    "resnext50_64x4d": (
        'https://paddle-hapi.bj.bcebos.com/models/resnext50_64x4d.pdparams',
        '063d4b483e12b06388529450ad7576db',
    ),
    'resnext101_32x4d': (
        'https://paddle-hapi.bj.bcebos.com/models/resnext101_32x4d.pdparams',
        '967b090039f9de2c8d06fe994fb9095f',
    ),
    'resnext101_64x4d': (
        'https://paddle-hapi.bj.bcebos.com/models/resnext101_64x4d.pdparams',
        '98e04e7ca616a066699230d769d03008',
    ),
    'resnext152_32x4d': (
        'https://paddle-hapi.bj.bcebos.com/models/resnext152_32x4d.pdparams',
        '18ff0beee21f2efc99c4b31786107121',
    ),
    'resnext152_64x4d': (
        'https://paddle-hapi.bj.bcebos.com/models/resnext152_64x4d.pdparams',
        '77c4af00ca42c405fa7f841841959379',
    ),
    'wide_resnet50_2': (
        'https://paddle-hapi.bj.bcebos.com/models/wide_resnet50_2.pdparams',
        '0282f804d73debdab289bd9fea3fa6dc',
    ),
    'wide_resnet101_2': (
        'https://paddle-hapi.bj.bcebos.com/models/wide_resnet101_2.pdparams',
        'd4360a2d23657f059216f5d5a1a9ac93',
    ),
}


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D

        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock"
            )

        self.conv1 = nn.Conv2D(
            inplanes, planes, 3, padding=1, stride=stride, bias_attr=False
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(planes, planes, 3, padding=1, bias_attr=False)
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


class BottleneckBlock(nn.Layer):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2D(inplanes, width, 1, bias_attr=False)
        self.bn1 = norm_layer(width)

        self.conv2 = nn.Conv2D(
            width,
            width,
            3,
            padding=dilation,
            stride=stride,
            groups=groups,
            dilation=dilation,
            bias_attr=False,
        )
        self.bn2 = norm_layer(width)

        self.conv3 = nn.Conv2D(
            width, planes * self.expansion, 1, bias_attr=False
        )
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Layer):
    """ResNet model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        Block (BasicBlock|BottleneckBlock): Block module of model.
        depth (int, optional): Layers of ResNet, Default: 50.
        width (int, optional): Base width per convolution group for each convolution block, Default: 64.
        num_classes (int, optional): Output dim of last fc layer. If num_classes <= 0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool, optional): Use pool before the last fc layer or not. Default: True.
        groups (int, optional): Number of groups for each convolution block, Default: 1.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNet model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import ResNet
            >>> from paddle.vision.models.resnet import BottleneckBlock, BasicBlock

            >>> # build ResNet with 18 layers
            >>> resnet18 = ResNet(BasicBlock, 18)

            >>> # build ResNet with 50 layers
            >>> resnet50 = ResNet(BottleneckBlock, 50)

            >>> # build Wide ResNet model
            >>> wide_resnet50_2 = ResNet(BottleneckBlock, 50, width=64*2)

            >>> # build ResNeXt model
            >>> resnext50_32x4d = ResNet(BottleneckBlock, 50, width=4, groups=32)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = resnet18(x)

            >>> print(out.shape)
            [1, 1000]
    """

    def __init__(
        self,
        block,
        depth=50,
        width=64,
        num_classes=1000,
        with_pool=True,
        groups=1,
    ):
        super().__init__()
        layer_cfg = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }
        layers = layer_cfg[depth]
        self.groups = groups
        self.base_width = width
        self.num_classes = num_classes
        self.with_pool = with_pool
        self._norm_layer = nn.BatchNorm2D

        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2D(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias_attr=False,
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1, 1))

        if num_classes > 0:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    1,
                    stride=stride,
                    bias_attr=False,
                ),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.with_pool:
            x = self.avgpool(x)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x)

        return x


def _resnet(arch, Block, depth, pretrained, **kwargs):
    model = ResNet(Block, depth, **kwargs)
    if pretrained:
        assert (
            arch in model_urls
        ), f"{arch} model do not have a pretrained model now, you should set pretrained=False"
        weight_path = get_weights_path_from_url(
            model_urls[arch][0], model_urls[arch][1]
        )

        param = paddle.load(weight_path)
        model.set_dict(param)

    return model


def resnet18(pretrained=False, **kwargs):
    """ResNet 18-layer model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_models_ResNet>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNet 18-layer model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import resnet18

            >>> # build model
            >>> model = resnet18()

            >>> # build model and load imagenet pretrained weight
            >>> # model = resnet18(pretrained=True)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    return _resnet('resnet18', BasicBlock, 18, pretrained, **kwargs)


def resnet34(pretrained=False, **kwargs):
    """ResNet 34-layer model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_models_ResNet>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNet 34-layer model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import resnet34

            >>> # build model
            >>> model = resnet34()

            >>> # build model and load imagenet pretrained weight
            >>> # model = resnet34(pretrained=True)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    return _resnet('resnet34', BasicBlock, 34, pretrained, **kwargs)


def resnet50(pretrained=False, **kwargs):
    """ResNet 50-layer model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_models_ResNet>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNet 50-layer model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import resnet50

            >>> # build model
            >>> model = resnet50()

            >>> # build model and load imagenet pretrained weight
            >>> # model = resnet50(pretrained=True)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    return _resnet('resnet50', BottleneckBlock, 50, pretrained, **kwargs)


def resnet101(pretrained=False, **kwargs):
    """ResNet 101-layer model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_models_ResNet>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNet 101-layer.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import resnet101

            >>> # build model
            >>> model = resnet101()

            >>> # build model and load imagenet pretrained weight
            >>> # model = resnet101(pretrained=True)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    return _resnet('resnet101', BottleneckBlock, 101, pretrained, **kwargs)


def resnet152(pretrained=False, **kwargs):
    """ResNet 152-layer model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_models_ResNet>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNet 152-layer model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import resnet152

            >>> # build model
            >>> model = resnet152()

            >>> # build model and load imagenet pretrained weight
            >>> # model = resnet152(pretrained=True)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    return _resnet('resnet152', BottleneckBlock, 152, pretrained, **kwargs)


def resnext50_32x4d(pretrained=False, **kwargs):
    """ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_models_ResNet>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNeXt-50 32x4d model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import resnext50_32x4d

            >>> # build model
            >>> model = resnext50_32x4d()

            >>> # build model and load imagenet pretrained weight
            >>> # model = resnext50_32x4d(pretrained=True)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    kwargs['groups'] = 32
    kwargs['width'] = 4
    return _resnet('resnext50_32x4d', BottleneckBlock, 50, pretrained, **kwargs)


def resnext50_64x4d(pretrained=False, **kwargs):
    """ResNeXt-50 64x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_models_ResNet>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNeXt-50 64x4d model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import resnext50_64x4d

            >>> # build model
            >>> model = resnext50_64x4d()

            >>> # build model and load imagenet pretrained weight
            >>> # model = resnext50_64x4d(pretrained=True)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    kwargs['groups'] = 64
    kwargs['width'] = 4
    return _resnet('resnext50_64x4d', BottleneckBlock, 50, pretrained, **kwargs)


def resnext101_32x4d(pretrained=False, **kwargs):
    """ResNeXt-101 32x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_models_ResNet>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNeXt-101 32x4d model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import resnext101_32x4d

            >>> # build model
            >>> model = resnext101_32x4d()

            >>> # build model and load imagenet pretrained weight
            >>> # model = resnext101_32x4d(pretrained=True)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    kwargs['groups'] = 32
    kwargs['width'] = 4
    return _resnet(
        'resnext101_32x4d', BottleneckBlock, 101, pretrained, **kwargs
    )


def resnext101_64x4d(pretrained=False, **kwargs):
    """ResNeXt-101 64x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_models_ResNet>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNeXt-101 64x4d model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import resnext101_64x4d

            >>> # build model
            >>> model = resnext101_64x4d()

            >>> # build model and load imagenet pretrained weight
            >>> # model = resnext101_64x4d(pretrained=True)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    kwargs['groups'] = 64
    kwargs['width'] = 4
    return _resnet(
        'resnext101_64x4d', BottleneckBlock, 101, pretrained, **kwargs
    )


def resnext152_32x4d(pretrained=False, **kwargs):
    """ResNeXt-152 32x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_models_ResNet>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNeXt-152 32x4d model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import resnext152_32x4d

            >>> # build model
            >>> model = resnext152_32x4d()

            >>> # build model and load imagenet pretrained weight
            >>> # model = resnext152_32x4d(pretrained=True)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    kwargs['groups'] = 32
    kwargs['width'] = 4
    return _resnet(
        'resnext152_32x4d', BottleneckBlock, 152, pretrained, **kwargs
    )


def resnext152_64x4d(pretrained=False, **kwargs):
    """ResNeXt-152 64x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_models_ResNet>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNeXt-152 64x4d model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import resnext152_64x4d

            >>> # build model
            >>> model = resnext152_64x4d()

            >>> # build model and load imagenet pretrained weight
            >>> # model = resnext152_64x4d(pretrained=True)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    kwargs['groups'] = 64
    kwargs['width'] = 4
    return _resnet(
        'resnext152_64x4d', BottleneckBlock, 152, pretrained, **kwargs
    )


def wide_resnet50_2(pretrained=False, **kwargs):
    """Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_models_ResNet>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of Wide ResNet-50-2 model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import wide_resnet50_2

            >>> # build model
            >>> model = wide_resnet50_2()

            >>> # build model and load imagenet pretrained weight
            >>> # model = wide_resnet50_2(pretrained=True)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    kwargs['width'] = 64 * 2
    return _resnet('wide_resnet50_2', BottleneckBlock, 50, pretrained, **kwargs)


def wide_resnet101_2(pretrained=False, **kwargs):
    """Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_models_ResNet>`.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of Wide ResNet-101-2 model.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.models import wide_resnet101_2

            >>> # build model
            >>> model = wide_resnet101_2()

            >>> # build model and load imagenet pretrained weight
            >>> # model = wide_resnet101_2(pretrained=True)

            >>> x = paddle.rand([1, 3, 224, 224])
            >>> out = model(x)

            >>> print(out.shape)
            [1, 1000]
    """
    kwargs['width'] = 64 * 2
    return _resnet(
        'wide_resnet101_2', BottleneckBlock, 101, pretrained, **kwargs
    )
