# encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.utils.download import get_weights_path_from_url

model_urls = {
    "wideresnet": (
        'https://bj.bcebos.com/v1/ai-studio-online/83c42c90785543eaa6bc7c37b91bd002a988b049d1664d80a6ed85bfe797a8cc?responseContentDisposition=attachment%3B%20filename%3Dcheckpoint.pdparams',
        '3238a4fcac05ebdc4f4c9c4cd3176d96',)
}


class BasicBlock(nn.Layer):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2D(in_planes)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2D(in_planes,
                               out_planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               weight_attr=nn.initializer.KaimingNormal())

        self.bn2 = nn.BatchNorm2D(out_planes)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2D(out_planes,
                               out_planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               weight_attr=nn.initializer.KaimingNormal())

        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0,
                                                                weight_attr=nn.initializer.KaimingNormal()) or None

    def forward(self, x):
        out = None
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return paddle.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Layer):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Layer):
    """WideResNet model from
    `"ImageNet Classification with Deep Convolutional Neural Networks"
    <https://arxiv.org/pdf/1605.07146.pdf>`_
    Args:
        num_classes (int): Output dim of last fc layer. Default: 10.

    Examples:

        .. code-block:: python

            from paddle.vision.models import WideResNet

            wideresnet = WideResNet()
    """

    def __init__(self, num_classes=10):
        super(WideResNet, self).__init__()

        depth = 28
        widen_factor = 20
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2D(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, weight_attr=nn.initializer.KaimingNormal())
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, 0.3)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, 0.3)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, 0.3)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2D(nChannels[3])
        self.relu = nn.ReLU()
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = paddle.reshape(out, shape=(-1, self.nChannels))
        return self.fc(out)


def _wideresnet(arch, pretrained=False, **kwargs):
    model = WideResNet(**kwargs)
    if pretrained:
        assert (
                arch in model_urls
        ), "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])
        param = paddle.load(weight_path)
        model.set_dict(param)
    return model


def wideresnet(pretrained=False, **kwargs):
    """WideResNet model
    Args:
        pretrained (bool): If True, returns a model pre-trained on cifar-10. Default: False.
    Examples:
        .. code-block:: python
            from paddle.vision.models import wideresnet
            # build model
            model = wideresnet()
            # build model and load imagenet pretrained weight
            # model = wideresnet(pretrained=True)
    """
    return _wideresnet('wideresnet', pretrained, **kwargs)