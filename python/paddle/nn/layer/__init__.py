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

# TODO: define activation functions of neural network

from . import activation
from . import loss
from . import conv
from . import extension
from . import activation
from . import norm

from .activation import *
from .loss import *
from .conv import *
from .extension import *
from .activation import *
from .norm import *
# from .activation import PReLU        #DEFINE_ALIAS
from .activation import ReLU  #DEFINE_ALIAS
from .activation import Sigmoid  #DEFINE_ALIAS
# from .activation import Softmax        #DEFINE_ALIAS
from .activation import LogSoftmax  #DEFINE_ALIAS
from .activation import HSigmoid  #DEFINE_ALIAS
from .common import BilinearTensorProduct  #DEFINE_ALIAS
from .common import Pool2D  #DEFINE_ALIAS
from .common import Embedding  #DEFINE_ALIAS
from .common import Linear  #DEFINE_ALIAS
from .common import UpSample  #DEFINE_ALIAS
from .conv import Conv2D  #DEFINE_ALIAS
from .conv import Conv2DTranspose  #DEFINE_ALIAS
from .conv import Conv3D  #DEFINE_ALIAS
from .conv import Conv3DTranspose  #DEFINE_ALIAS
# from .conv import TreeConv        #DEFINE_ALIAS
# from .conv import Conv1D        #DEFINE_ALIAS
from .extension import RowConv  #DEFINE_ALIAS
# from .learning_rate import CosineDecay        #DEFINE_ALIAS
# from .learning_rate import ExponentialDecay        #DEFINE_ALIAS
# from .learning_rate import InverseTimeDecay        #DEFINE_ALIAS
# from .learning_rate import NaturalExpDecay        #DEFINE_ALIAS
# from .learning_rate import NoamDecay        #DEFINE_ALIAS
# from .learning_rate import PiecewiseDecay        #DEFINE_ALIAS
# from .learning_rate import PolynomialDecay        #DEFINE_ALIAS
# from .loss import NCELoss        #DEFINE_ALIAS
from .loss import CrossEntropyLoss  #DEFINE_ALIAS
from .loss import MSELoss  #DEFINE_ALIAS
from .loss import L1Loss  #DEFINE_ALIAS
from .loss import NLLLoss  #DEFINE_ALIAS
from .loss import BCELoss  #DEFINE_ALIAS
from .norm import BatchNorm  #DEFINE_ALIAS
from .norm import GroupNorm  #DEFINE_ALIAS
from .norm import LayerNorm  #DEFINE_ALIAS
from .norm import SpectralNorm  #DEFINE_ALIAS
from .norm import InstanceNorm  #DEFINE_ALIAS
# from .rnn import RNNCell        #DEFINE_ALIAS
# from .rnn import GRUCell        #DEFINE_ALIAS
# from .rnn import LSTMCell        #DEFINE_ALIAS
