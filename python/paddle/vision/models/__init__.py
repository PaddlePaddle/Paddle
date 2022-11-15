#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .resnet import ResNet  # noqa: F401
from .resnet import resnet18  # noqa: F401
from .resnet import resnet34  # noqa: F401
from .resnet import resnet50  # noqa: F401
from .resnet import resnet101  # noqa: F401
from .resnet import resnet152  # noqa: F401
from .resnet import resnext50_32x4d  # noqa: F401
from .resnet import resnext50_64x4d  # noqa: F401
from .resnet import resnext101_32x4d  # noqa: F401
from .resnet import resnext101_64x4d  # noqa: F401
from .resnet import resnext152_32x4d  # noqa: F401
from .resnet import resnext152_64x4d  # noqa: F401
from .resnet import wide_resnet50_2  # noqa: F401
from .resnet import wide_resnet101_2  # noqa: F401
from .mobilenetv1 import MobileNetV1  # noqa: F401
from .mobilenetv1 import mobilenet_v1  # noqa: F401
from .mobilenetv2 import MobileNetV2  # noqa: F401
from .mobilenetv2 import mobilenet_v2  # noqa: F401
from .mobilenetv3 import MobileNetV3Small  # noqa: F401
from .mobilenetv3 import MobileNetV3Large  # noqa: F401
from .mobilenetv3 import mobilenet_v3_small  # noqa: F401
from .mobilenetv3 import mobilenet_v3_large  # noqa: F401
from .vgg import VGG  # noqa: F401
from .vgg import vgg11  # noqa: F401
from .vgg import vgg13  # noqa: F401
from .vgg import vgg16  # noqa: F401
from .vgg import vgg19  # noqa: F401
from .lenet import LeNet  # noqa: F401
from .densenet import DenseNet  # noqa: F401
from .densenet import densenet121  # noqa: F401
from .densenet import densenet161  # noqa: F401
from .densenet import densenet169  # noqa: F401
from .densenet import densenet201  # noqa: F401
from .densenet import densenet264  # noqa: F401
from .alexnet import AlexNet  # noqa: F401
from .alexnet import alexnet  # noqa: F401
from .inceptionv3 import InceptionV3  # noqa: F401
from .inceptionv3 import inception_v3  # noqa: F401
from .squeezenet import SqueezeNet  # noqa: F401
from .squeezenet import squeezenet1_0  # noqa: F401
from .squeezenet import squeezenet1_1  # noqa: F401
from .googlenet import GoogLeNet  # noqa: F401
from .googlenet import googlenet  # noqa: F401
from .shufflenetv2 import ShuffleNetV2  # noqa: F401
from .shufflenetv2 import shufflenet_v2_x0_25  # noqa: F401
from .shufflenetv2 import shufflenet_v2_x0_33  # noqa: F401
from .shufflenetv2 import shufflenet_v2_x0_5  # noqa: F401
from .shufflenetv2 import shufflenet_v2_x1_0  # noqa: F401
from .shufflenetv2 import shufflenet_v2_x1_5  # noqa: F401
from .shufflenetv2 import shufflenet_v2_x2_0  # noqa: F401
from .shufflenetv2 import shufflenet_v2_swish  # noqa: F401

__all__ = [  # noqa
    'ResNet',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'resnext50_32x4d',
    'resnext50_64x4d',
    'resnext101_32x4d',
    'resnext101_64x4d',
    'resnext152_32x4d',
    'resnext152_64x4d',
    'wide_resnet50_2',
    'wide_resnet101_2',
    'VGG',
    'vgg11',
    'vgg13',
    'vgg16',
    'vgg19',
    'MobileNetV1',
    'mobilenet_v1',
    'MobileNetV2',
    'mobilenet_v2',
    'MobileNetV3Small',
    'MobileNetV3Large',
    'mobilenet_v3_small',
    'mobilenet_v3_large',
    'LeNet',
    'DenseNet',
    'densenet121',
    'densenet161',
    'densenet169',
    'densenet201',
    'densenet264',
    'AlexNet',
    'alexnet',
    'InceptionV3',
    'inception_v3',
    'SqueezeNet',
    'squeezenet1_0',
    'squeezenet1_1',
    'GoogLeNet',
    'googlenet',
    'ShuffleNetV2',
    'shufflenet_v2_x0_25',
    'shufflenet_v2_x0_33',
    'shufflenet_v2_x0_5',
    'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5',
    'shufflenet_v2_x2_0',
    'shufflenet_v2_swish',
]
