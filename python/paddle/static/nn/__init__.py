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

__all__ = [     #noqa
           'fc', 
           'batch_norm', 
           'embedding', 
           'bilinear_tensor_product', 
           'case', 
           'cond', 
           'conv2d', 
           'conv2d_transpose', 
           'conv3d', 
           'conv3d_transpose', 
           'crf_decoding', 
           'data_norm', 
           'deform_conv2d', 
           'group_norm', 
           'instance_norm', 
           'layer_norm', 
           'multi_box_head', 
           'nce', 
           'prelu', 
           'row_conv', 
           'spectral_norm', 
           'switch_case', 
           'while_loop'
]

from .common import fc  # noqa: F401
from .common import deform_conv2d  # noqa: F401

from ...fluid.layers import batch_norm  # noqa: F401
from ...fluid.layers import bilinear_tensor_product  # noqa: F401
from ...fluid.layers import case  # noqa: F401
from ...fluid.layers import cond  # noqa: F401
from ...fluid.layers import conv2d  # noqa: F401
from ...fluid.layers import conv2d_transpose  # noqa: F401
from ...fluid.layers import conv3d  # noqa: F401
from ...fluid.layers import conv3d_transpose  # noqa: F401
from ...fluid.layers import create_parameter  # noqa: F401
from ...fluid.layers import crf_decoding  # noqa: F401
from ...fluid.layers import data_norm  # noqa: F401
from ...fluid.layers import group_norm  # noqa: F401
from ...fluid.layers import instance_norm  # noqa: F401
from ...fluid.layers import layer_norm  # noqa: F401
from ...fluid.layers import multi_box_head  # noqa: F401
from ...fluid.layers import nce  # noqa: F401
from ...fluid.layers import prelu  # noqa: F401
from ...fluid.layers import py_func  # noqa: F401
from ...fluid.layers import row_conv  # noqa: F401
from ...fluid.layers import spectral_norm  # noqa: F401
from ...fluid.layers import switch_case  # noqa: F401
from ...fluid.layers import while_loop  # noqa: F401

from ...fluid.input import embedding  # noqa: F401
from ...fluid.contrib.layers import sparse_embedding  # noqa: F401
