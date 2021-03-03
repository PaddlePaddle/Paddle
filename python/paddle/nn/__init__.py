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

# TODO: import all neural network related api under this directory,
# including layers, linear, conv, rnn etc.

from .layer import norm
from .functional import extension
from .layer import common
from .layer import rnn
from .utils import weight_norm_hook

from . import initializer

__all__ = []
__all__ += norm.__all__
__all__ += extension.__all__
__all__ += common.__all__
__all__ += rnn.__all__
__all__ += weight_norm_hook.__all__

# TODO: define alias in nn directory
from .clip import ClipGradByGlobalNorm  #DEFINE_ALIAS
from .clip import ClipGradByNorm  #DEFINE_ALIAS
from .clip import ClipGradByValue  #DEFINE_ALIAS
# from .control_flow import cond  #DEFINE_ALIAS
# from .control_flow import DynamicRNN        #DEFINE_ALIAS
# from .control_flow import StaticRNN        #DEFINE_ALIAS
# from .control_flow import while_loop  #DEFINE_ALIAS
# from .control_flow import rnn        #DEFINE_ALIAS
from .decode import BeamSearchDecoder  #DEFINE_ALIAS
from .decode import dynamic_decode  #DEFINE_ALIAS
# from .decode import Decoder        #DEFINE_ALIAS
# from .decode import crf_decoding        #DEFINE_ALIAS
# from .decode import ctc_greedy_decoder        #DEFINE_ALIAS
# from .input import Input        #DEFINE_ALIAS
from .layer.activation import ELU  #DEFINE_ALIAS
from .layer.activation import GELU  #DEFINE_ALIAS
from .layer.activation import Tanh  #DEFINE_ALIAS
from .layer.activation import Hardshrink  #DEFINE_ALIAS
from .layer.activation import Hardswish  #DEFINE_ALIAS
from .layer.activation import Hardtanh  #DEFINE_ALIAS
from .layer.activation import PReLU  #DEFINE_ALIAS
from .layer.activation import ReLU  #DEFINE_ALIAS
from .layer.activation import ReLU6  #DEFINE_ALIAS
from .layer.activation import SELU  #DEFINE_ALIAS
from .layer.activation import LeakyReLU  #DEFINE_ALIAS
from .layer.activation import Sigmoid  #DEFINE_ALIAS
from .layer.activation import Hardsigmoid  #DEFINE_ALIAS
from .layer.activation import LogSigmoid  #DEFINE_ALIAS
from .layer.activation import Softmax  #DEFINE_ALIAS
from .layer.activation import Softplus  #DEFINE_ALIAS
from .layer.activation import Softshrink  #DEFINE_ALIAS
from .layer.activation import Softsign  #DEFINE_ALIAS
from .layer.activation import Swish  #DEFINE_ALIAS
from .layer.activation import Tanhshrink  #DEFINE_ALIAS
from .layer.activation import ThresholdedReLU  #DEFINE_ALIAS
from .layer.activation import LogSoftmax  #DEFINE_ALIAS
from .layer.activation import Maxout  #DEFINE_ALIAS
from .layer.common import Pad1D  #DEFINE_ALIAS
from .layer.common import Pad2D  #DEFINE_ALIAS
from .layer.common import Pad3D  #DEFINE_ALIAS
from .layer.common import CosineSimilarity  #DEFINE_ALIAS
from .layer.common import Embedding  #DEFINE_ALIAS
from .layer.common import Linear  #DEFINE_ALIAS
from .layer.common import Flatten  #DEFINE_ALIAS
from .layer.common import Upsample  #DEFINE_ALIAS
from .layer.common import UpsamplingNearest2D  #DEFINE_ALIAS
from .layer.common import UpsamplingBilinear2D  #DEFINE_ALIAS
from .layer.common import Bilinear  #DEFINE_ALIAS
from .layer.common import Dropout  #DEFINE_ALIAS
from .layer.common import Dropout2D  #DEFINE_ALIAS
from .layer.common import Dropout3D  #DEFINE_ALIAS
from .layer.common import AlphaDropout  #DEFINE_ALIAS

from .layer.pooling import AvgPool1D  #DEFINE_ALIAS
from .layer.pooling import AvgPool2D  #DEFINE_ALIAS
from .layer.pooling import AvgPool3D  #DEFINE_ALIAS
from .layer.pooling import MaxPool1D  #DEFINE_ALIAS
from .layer.pooling import MaxPool2D  #DEFINE_ALIAS
from .layer.pooling import MaxPool3D  #DEFINE_ALIAS
from .layer.pooling import AdaptiveAvgPool1D  #DEFINE_ALIAS
from .layer.pooling import AdaptiveAvgPool2D  #DEFINE_ALIAS
from .layer.pooling import AdaptiveAvgPool3D  #DEFINE_ALIAS

from .layer.pooling import AdaptiveMaxPool1D  #DEFINE_ALIAS
from .layer.pooling import AdaptiveMaxPool2D  #DEFINE_ALIAS
from .layer.pooling import AdaptiveMaxPool3D  #DEFINE_ALIAS
from .layer.conv import Conv1D  #DEFINE_ALIAS
from .layer.conv import Conv2D  #DEFINE_ALIAS
from .layer.conv import Conv3D  #DEFINE_ALIAS
from .layer.conv import Conv1DTranspose  #DEFINE_ALIAS
from .layer.conv import Conv2DTranspose  #DEFINE_ALIAS
from .layer.conv import Conv3DTranspose  #DEFINE_ALIAS
# from .layer.conv import TreeConv        #DEFINE_ALIAS
# from .layer.conv import Conv1D        #DEFINE_ALIAS
from .layer.common import Linear
# from .layer.loss import NCELoss        #DEFINE_ALIAS
from .layer.loss import BCEWithLogitsLoss  #DEFINE_ALIAS
from .layer.loss import CrossEntropyLoss  #DEFINE_ALIAS
from .layer.loss import HSigmoidLoss  #DEFINE_ALIAS
from .layer.loss import MSELoss  #DEFINE_ALIAS
from .layer.loss import L1Loss  #DEFINE_ALIAS
from .layer.loss import NLLLoss  #DEFINE_ALIAS
from .layer.loss import BCELoss  #DEFINE_ALIAS
from .layer.loss import KLDivLoss  #DEFINE_ALIAS
from .layer.loss import MarginRankingLoss  #DEFINE_ALIAS
from .layer.loss import CTCLoss  #DEFINE_ALIAS
from .layer.loss import SmoothL1Loss  #DEFINE_ALIAS
from .layer.norm import BatchNorm  #DEFINE_ALIAS
from .layer.norm import SyncBatchNorm  #DEFINE_ALIAS
from .layer.norm import GroupNorm  #DEFINE_ALIAS
from .layer.norm import LayerNorm  #DEFINE_ALIAS
from .layer.norm import SpectralNorm  #DEFINE_ALIAS
from .layer.norm import InstanceNorm1D  #DEFINE_ALIAS
from .layer.norm import InstanceNorm2D  #DEFINE_ALIAS
from .layer.norm import InstanceNorm3D  #DEFINE_ALIAS
from .layer.norm import BatchNorm1D  #DEFINE_ALIAS
from .layer.norm import BatchNorm2D  #DEFINE_ALIAS
from .layer.norm import BatchNorm3D  #DEFINE_ALIAS
from .layer.norm import LocalResponseNorm  #DEFINE_ALIAS

from .layer.rnn import RNNCellBase  #DEFINE_ALIAS
from .layer.rnn import SimpleRNNCell  #DEFINE_ALIAS
from .layer.rnn import LSTMCell  #DEFINE_ALIAS
from .layer.rnn import GRUCell  #DEFINE_ALIAS
from .layer.rnn import RNN  #DEFINE_ALIAS
from .layer.rnn import BiRNN  #DEFINE_ALIAS
from .layer.rnn import SimpleRNN  #DEFINE_ALIAS
from .layer.rnn import LSTM  #DEFINE_ALIAS
from .layer.rnn import GRU  #DEFINE_ALIAS

from .layer.transformer import MultiHeadAttention
from .layer.transformer import TransformerEncoderLayer
from .layer.transformer import TransformerEncoder
from .layer.transformer import TransformerDecoderLayer
from .layer.transformer import TransformerDecoder
from .layer.transformer import Transformer
from .layer.distance import PairwiseDistance  #DEFINE_ALIAS

from .layer.vision import PixelShuffle

from .layer import loss  #DEFINE_ALIAS
from .layer import conv  #DEFINE_ALIAS
from .layer import vision  #DEFINE_ALIAS
from ..fluid.dygraph.layers import Layer  #DEFINE_ALIAS
from ..fluid.dygraph.container import LayerList, ParameterList, Sequential  #DEFINE_ALIAS
