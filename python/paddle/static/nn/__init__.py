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

from ...tensor.creation import create_parameter  # noqa: F401
from .common import (
    batch_norm,
    bilinear_tensor_product,
    continuous_value_model,  # noqa: F401
    conv2d,
    conv2d_transpose,
    conv3d,
    conv3d_transpose,
    data_norm,
    deform_conv2d,
    embedding,
    fc,
    group_norm,
    instance_norm,
    layer_norm,
    prelu,
    py_func,
    row_conv,
    sparse_embedding,
    spectral_norm,
)
from .control_flow import case, cond, switch_case, while_loop
from .loss import nce
from .sequence_lod import (
    sequence_conv,
    sequence_enumerate,
    sequence_expand,
    sequence_expand_as,
    sequence_first_step,
    sequence_last_step,
    sequence_pad,
    sequence_pool,
    sequence_reshape,
    sequence_scatter,
    sequence_slice,
    sequence_softmax,
    sequence_unpad,
)
from .static_pylayer import static_pylayer

__all__ = [
    'fc',
    'batch_norm',
    'bilinear_tensor_product',
    'embedding',
    'case',
    'cond',
    'static_pylayer',
    'conv2d',
    'conv2d_transpose',
    'conv3d',
    'conv3d_transpose',
    'data_norm',
    'deform_conv2d',
    'group_norm',
    'instance_norm',
    'layer_norm',
    'nce',
    'prelu',
    'py_func',
    'row_conv',
    'spectral_norm',
    'switch_case',
    'while_loop',
    'sparse_embedding',
    'sequence_conv',
    'sequence_softmax',
    'sequence_pool',
    'sequence_first_step',
    'sequence_last_step',
    'sequence_slice',
    'sequence_expand',
    'sequence_expand_as',
    'sequence_pad',
    'sequence_unpad',
    'sequence_reshape',
    'sequence_scatter',
    'sequence_enumerate',
    'prelu',
]
