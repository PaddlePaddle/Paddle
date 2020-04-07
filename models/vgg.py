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
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.container import Sequential

from model import Model
from .download import get_weights_path

__all__ = [
    'VGG',
    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19_bn',
    'vgg19',
]

model_urls = {
    'vgg16': ('https://paddle-hapi.bj.bcebos.com/models/vgg16.pdparams',
              'c788f453a3b999063e8da043456281ee')
}


class Classifier(fluid.dygraph.Layer):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.linear1 = Linear(512 * 7 * 7, 4096)
        self.linear2 = Linear(4096, 4096)
        self.linear3 = Linear(4096, num_classes, act='softmax')

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
        num_classes (int): output dim of last fc layer. Default: 1000.
    """

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        classifier = Classifier(num_classes)
        self.classifier = self.add_sublayer("classifier",
                                            Sequential(classifier))

    def forward(self, x):
        x = self.features(x)
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
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'E': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
        512, 'M', 512, 512, 512, 512, 'M'
    ],
}


def _vgg(arch, cfg, batch_norm, pretrained, **kwargs):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)

    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path(model_urls[arch][0],
                                       model_urls[arch][1])
        assert weight_path.endswith(
            '.pdparams'), "suffix of weight must be .pdparams"
        model.load(weight_path[:-9])

    return model


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg('vgg11', 'A', False, pretrained, **kwargs)


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model with batch normalization
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, **kwargs)


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg('vgg13', 'B', False, pretrained, **kwargs)


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model with batch normalization
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, **kwargs)


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model 
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg('vgg16', 'D', False, pretrained, **kwargs)


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer with batch normalization
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, **kwargs)


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model 
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg('vgg19', 'E', False, pretrained, **kwargs)


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model with batch normalization
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, **kwargs)
