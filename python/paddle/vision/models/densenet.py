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
import paddle.nn as nn

from paddle.utils.download import get_weights_path_from_url

__all__ = ['densenet121','densenet161','densenet169',
'densenet201']

model_urls = {
    'DenseNet121' : (''),
    'DenseNet161' : (''),
    'DenseNet169' : (''),
    'DenseNet201' : ('')
}

class DenseLayer(nn.Layer):
    
    def __init__(self, in_c, growth_rate, bn_size):
        super().__init__()
        out_c = growth_rate * bn_size
        self.layers = nn.Sequential(
            nn.BatchNorm2D(in_c),
            nn.ReLU(),nn.Conv2D(in_c,out_c,1),
            nn.BatchNorm2D(out_c),
            nn.ReLU(),
            nn.Conv2D(out_c,growth_rate,3,padding=1))

    def forward(self, x):
        y = self.layers(x)
        return y

class DenseBlock(nn.Layer):

    def __init__(self, num_layers, in_c, growth_rate, bn_size):
        super().__init__()
        self.layers = nn.LayerList()
        for ind in range(num_layers):
            self.layers.append(
                DenseLayer(
                    in_c=in_c+ind*growth_rate,
                    growth_rate=growth_rate,
                    bn_size=bn_size)
            )
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_x = layer(paddle.concat(features,axis=1))
            features.append(new_x)
        return paddle.concat(features,axis=1)

class Transition(nn.Layer):

    def __init__(self, in_c, out_c):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2D(in_c),
            nn.ReLU(),nn.Conv2D(in_c,out_c,1),nn.AvgPool2D(2,2))
    
    def forward(self, x):
        return self.layers(x)


class DenseNet(nn.Layer):
    
    def __init__(self,
                num_classes=1000,
                growth_rate=32,
                block=(6,12,24,16),
                bn_size=4,
                out_c=64):
        super().__init__()
        self.conv_pool = nn.Sequential(
                            nn.Conv2D(3, out_c, 7, stride=2, padding=3), nn.MaxPool2D(3, 2))
        self.blocks = nn.LayerList()
        in_c = out_c
        for ind, n in enumerate(block):
            self.blocks.append(DenseBlock(n, in_c, growth_rate, bn_size))
            in_c += growth_rate * n
            if ind != len(block) - 1:
                self.blocks.append(Transition(in_c, in_c // 2))
                in_c //= 2
        self.blocks.append(
            nn.Sequential(
                nn.BatchNorm2D(in_c),
                nn.ReLU(),nn.AdaptiveAvgPool2D((1, 1)),nn.Flatten()))
        self.cls = nn.Linear(in_c, num_classes)
    
    def forward(self, x):
        x = self.conv_pool(x)
        for layer in self.blocks:
            x = layer(x)
        x = self.cls(x)
        return x

def _DenseNet(arch, block_cfg, batch_norm, pretrained, **kwargs):
    model = DenseNet(block=block_cfg, **kwargs)

    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])

        param = paddle.load(weight_path)
        model.load_dict(param)

    return model


def densenet121(pretrained=False, batch_norm=False, **kwargs):
    model_name = 'DenseNet121'
    if batch_norm:
        model_name += ('_bn')
    return _DenseNet(model_name, 
        (6,12,24,16), batch_norm, pretrained, **kwargs)


def densenet161(pretrained=False, batch_norm=False, **kwargs):
    model_name = 'DenseNet161'
    if batch_norm:
        model_name += ('_bn')
    return _DenseNet(model_name, 
        (6,12,32,32), batch_norm, pretrained, **kwargs)


def densenet169(pretrained=False, batch_norm=False, **kwargs):
    model_name = 'DenseNet169'
    if batch_norm:
        model_name += ('_bn')
    return _DenseNet(model_name, 
        (6,12,48,32), batch_norm, pretrained, **kwargs)


def densenet201(pretrained=False, batch_norm=False, **kwargs):
    model_name = 'DenseNet201'
    if batch_norm:
        model_name += ('_bn')
    return _DenseNet(model_name, 
        (6,12,64,48), batch_norm, pretrained, **kwargs)
