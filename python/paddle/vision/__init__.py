#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from . import models  # noqa: F401
from . import transforms  # noqa: F401
from . import datasets  # noqa: F401
from . import ops  # noqa: F401
from .image import set_image_backend  # noqa: F401
from .image import get_image_backend  # noqa: F401
from .image import image_load  # noqa: F401
from .datasets import DatasetFolder  # noqa: F401
from .datasets import ImageFolder  # noqa: F401
from .datasets import MNIST  # noqa: F401
from .datasets import FashionMNIST  # noqa: F401
from .datasets import Flowers  # noqa: F401
from .datasets import Cifar10  # noqa: F401
from .datasets import Cifar100  # noqa: F401
from .datasets import VOC2012  # noqa: F401
from .models import ResNet  # noqa: F401
from .models import resnet18  # noqa: F401
from .models import resnet34  # noqa: F401
from .models import resnet50  # noqa: F401
from .models import resnet101  # noqa: F401
from .models import resnet152  # noqa: F401
from .models import wide_resnet50_2  # noqa: F401
from .models import wide_resnet101_2  # noqa: F401
from .models import MobileNetV1  # noqa: F401
from .models import mobilenet_v1  # noqa: F401
from .models import MobileNetV2  # noqa: F401
from .models import mobilenet_v2  # noqa: F401
from .models import MobileNetV3Small  # noqa: F401
from .models import MobileNetV3Large  # noqa: F401
from .models import mobilenet_v3_small  # noqa: F401
from .models import mobilenet_v3_large  # noqa: F401
from .models import SqueezeNet  # noqa: F401
from .models import squeezenet1_0  # noqa: F401
from .models import squeezenet1_1  # noqa: F401
from .models import VGG  # noqa: F401
from .models import vgg11  # noqa: F401
from .models import vgg13  # noqa: F401
from .models import vgg16  # noqa: F401
from .models import vgg19  # noqa: F401
from .models import LeNet  # noqa: F401
from .models import DenseNet  # noqa: F401
from .models import densenet121  # noqa: F401
from .models import densenet161  # noqa: F401
from .models import densenet169  # noqa: F401
from .models import densenet201  # noqa: F401
from .models import densenet264  # noqa: F401
from .models import AlexNet  # noqa: F401
from .models import alexnet  # noqa: F401
from .models import ResNeXt  # noqa: F401
from .models import resnext50_32x4d  # noqa: F401
from .models import resnext50_64x4d  # noqa: F401
from .models import resnext101_32x4d  # noqa: F401
from .models import resnext101_64x4d  # noqa: F401
from .models import resnext152_32x4d  # noqa: F401
from .models import resnext152_64x4d  # noqa: F401
from .models import InceptionV3  # noqa: F401
from .models import inception_v3  # noqa: F401
from .models import GoogLeNet  # noqa: F401
from .models import googlenet  # noqa: F401
from .models import ShuffleNetV2  # noqa: F401
from .models import shufflenet_v2_x0_25  # noqa: F401
from .models import shufflenet_v2_x0_33  # noqa: F401
from .models import shufflenet_v2_x0_5  # noqa: F401
from .models import shufflenet_v2_x1_0  # noqa: F401
from .models import shufflenet_v2_x1_5  # noqa: F401
from .models import shufflenet_v2_x2_0  # noqa: F401
from .models import shufflenet_v2_swish  # noqa: F401
from .transforms import BaseTransform  # noqa: F401
from .transforms import Compose  # noqa: F401
from .transforms import Resize  # noqa: F401
from .transforms import RandomResizedCrop  # noqa: F401
from .transforms import CenterCrop  # noqa: F401
from .transforms import RandomHorizontalFlip  # noqa: F401
from .transforms import RandomVerticalFlip  # noqa: F401
from .transforms import Transpose  # noqa: F401
from .transforms import Normalize  # noqa: F401
from .transforms import BrightnessTransform  # noqa: F401
from .transforms import SaturationTransform  # noqa: F401
from .transforms import ContrastTransform  # noqa: F401
from .transforms import HueTransform  # noqa: F401
from .transforms import ColorJitter  # noqa: F401
from .transforms import RandomCrop  # noqa: F401
from .transforms import Pad  # noqa: F401
from .transforms import RandomRotation  # noqa: F401
from .transforms import Grayscale  # noqa: F401
from .transforms import ToTensor  # noqa: F401
from .transforms import to_tensor  # noqa: F401
from .transforms import hflip  # noqa: F401
from .transforms import vflip  # noqa: F401
from .transforms import resize  # noqa: F401
from .transforms import pad  # noqa: F401
from .transforms import rotate  # noqa: F401
from .transforms import to_grayscale  # noqa: F401
from .transforms import crop  # noqa: F401
from .transforms import center_crop  # noqa: F401
from .transforms import adjust_brightness  # noqa: F401
from .transforms import adjust_contrast  # noqa: F401
from .transforms import adjust_hue  # noqa: F401
from .transforms import normalize  # noqa: F401

__all__ = [  #noqa
    'set_image_backend', 'get_image_backend', 'image_load'
]
