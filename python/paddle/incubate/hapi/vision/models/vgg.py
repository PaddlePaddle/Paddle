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

import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.container import Sequential

from ...model import Model
from ...download import get_weights_path_from_url

__all__ = [
    'VGG',
    'vgg11',
    'vgg13',
    'vgg16',
    'vgg19',
]

model_urls = {
    'vgg16': ('https://paddle-hapi.bj.bcebos.com/models/vgg16.pdparams',
              'c788f453a3b999063e8da043456281ee')
}


class Classifier(fluid.dygraph.Layer):
    def __init__(self, num_classes, classifier_activation='softmax'):
        super(Classifier, self).__init__()
        self.linear1 = Linear(512 * 7 * 7, 4096)
        self.linear2 = Linear(4096, 4096)
        self.linear3 = Linear(4096, num_classes, act=classifier_activation)

    def forward(self, x):
        x = self.linear1(x)
        x = fluid.layers.relu(x)
        x = fluid.layers.dropout(x, 0.5)
        x = self.linear2(x)
        x = fluid.layers.relu(x)
        x = fluid.layers.dropout(x, 0.5)
        out = self.linear3(x)
        return out


class VGG(Model):
    """VGG model from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        features (fluid.dygraph.Layer): vgg features create by function make_layers.
        num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer 
                            will not be defined. Default: 1000.
        classifier_activation (str): activation for the last fc layer. Default: 'softmax'.

    Examples:
        .. code-block:: python

            from paddle.incubate.hapi.vision.models import VGG
            from paddle.incubate.hapi.vision.models.vgg import make_layers

            vgg11_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

            features = make_layers(vgg11_cfg)

            vgg11 = VGG(features)

    """

    def __init__(self,
                 features,
                 num_classes=1000,
                 classifier_activation='softmax'):
        super(VGG, self).__init__()
        self.features = features
        self.num_classes = num_classes

        if num_classes > 0:
            classifier = Classifier(num_classes, classifier_activation)
            self.classifier = self.add_sublayer("classifier",
                                                Sequential(classifier))

    def forward(self, x):
        x = self.features(x)

        if self.num_classes > 0:
            x = fluid.layers.flatten(x, 1)
            x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3

    for v in cfg:
        if v == 'M':
            layers += [Pool2D(pool_size=2, pool_stride=2)]
        else:
            if batch_norm:
                conv2d = Conv2D(in_channels, v, filter_size=3, padding=1)
                layers += [conv2d, BatchNorm(v, act='relu')]
            else:
                conv2d = Conv2D(
                    in_channels, v, filter_size=3, padding=1, act='relu')
                layers += [conv2d]
            in_channels = v
    return Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B':
    [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512,
        512, 512, 'M'
    ],
    'E': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512,
        'M', 512, 512, 512, 512, 'M'
    ],
}


def _vgg(arch, cfg, batch_norm, pretrained, **kwargs):
    model = VGG(make_layers(
        cfgs[cfg], batch_norm=batch_norm),
                num_classes=1000,
                **kwargs)

    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])
        assert weight_path.endswith(
            '.pdparams'), "suffix of weight must be .pdparams"
        model.load(weight_path)

    return model


def vgg11(pretrained=False, batch_norm=False, **kwargs):
    """VGG 11-layer model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.
        batch_norm (bool): If True, returns a model with batch_norm layer. Default: False.

    Examples:
        .. code-block:: python

            from paddle.incubate.hapi.vision.models import vgg11

            # build model
            model = vgg11()

            # build vgg11 model with batch_norm
            model = vgg11(batch_norm=True)
    """
    model_name = 'vgg11'
    if batch_norm:
        model_name += ('_bn')
    return _vgg(model_name, 'A', batch_norm, pretrained, **kwargs)


def vgg13(pretrained=False, batch_norm=False, **kwargs):
    """VGG 13-layer model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.
        batch_norm (bool): If True, returns a model with batch_norm layer. Default: False.

    Examples:
        .. code-block:: python

            from paddle.incubate.hapi.vision.models import vgg13

            # build model
            model = vgg13()

            # build vgg13 model with batch_norm
            model = vgg13(batch_norm=True)
    """
    model_name = 'vgg13'
    if batch_norm:
        model_name += ('_bn')
    return _vgg(model_name, 'B', batch_norm, pretrained, **kwargs)


def vgg16(pretrained=False, batch_norm=False, **kwargs):
    """VGG 16-layer model 
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.
        batch_norm (bool): If True, returns a model with batch_norm layer. Default: False.

    Examples:
        .. code-block:: python

            from paddle.incubate.hapi.vision.models import vgg16

            # build model
            model = vgg16()

            # build vgg16 model with batch_norm
            model = vgg16(batch_norm=True)
    """
    model_name = 'vgg16'
    if batch_norm:
        model_name += ('_bn')
    return _vgg(model_name, 'D', batch_norm, pretrained, **kwargs)


def vgg19(pretrained=False, batch_norm=False, **kwargs):
    """VGG 19-layer model 
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.
        batch_norm (bool): If True, returns a model with batch_norm layer. Default: False.

    Examples:
        .. code-block:: python

            from paddle.incubate.hapi.vision.models import vgg19

            # build model
            model = vgg19()

            # build vgg19 model with batch_norm
            model = vgg19(batch_norm=True)
    """
    model_name = 'vgg19'
    if batch_norm:
        model_name += ('_bn')
    return _vgg(model_name, 'E', batch_norm, pretrained, **kwargs)
