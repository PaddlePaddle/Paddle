#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

_channelwise_quant_axis1_ops = [
    'conv2d_transpose', 'mul', 'matmul', 'matmul_v2'
]


def load_variable_data(scope, var_name):
    '''
    Load variable value from scope
    '''
    var_node = scope.find_var(var_name)
    assert var_node is not None, \
        "Cannot find " + var_name + " in scope."
    return np.array(var_node.get_tensor())


def set_variable_data(scope, place, var_name, np_value):
    '''
    Set the value of var node by name, if the node exits,
    '''
    assert isinstance(np_value, np.ndarray), \
       'The type of value should be numpy array.'
    var_node = scope.find_var(var_name)
    if var_node != None:
        tensor = var_node.get_tensor()
        tensor.set(np_value, place)


def quant_tensor(x, scale, quant_axis=0, weight_bits=8):
    # symmetry quant
    def _clip(x, scale):
        x[x > scale] = scale
        x[x < -scale] = -scale
        return x

    assert quant_axis in [0, 1], 'quant_axis should be 0 or 1 for now.'
    bnt = (1 << (weight_bits - 1)) - 1
    if isinstance(scale, list):
        for i, s in enumerate(scale):
            if s == 0.0:
                s = 1e-8
            if quant_axis == 0:
                x[i] = _clip(x[i], s)
                x[i] = x[i] / s * bnt
            else:
                x[:, i] = _clip(x[:, i], s)
                x[:, i] = x[:, i] / s * bnt
    else:
        scale = 1e-8 if scale == 0.0 else scale
        x = _clip(x, scale)
        x = x / scale * bnt
    return x


def dequant_tensor(x, scale, quant_axis=0, weight_bits=8):
    assert quant_axis in [0, 1], 'quant_axis should be 0 or 1 for now.'
    bnt = (1 << (weight_bits - 1)) - 1
    if isinstance(scale, list):
        for i, s in enumerate(scale):
            if s == 0.0:
                s = 1e-8
            if quant_axis == 0:
                x[i] = x[i] * s / bnt
            else:
                x[:, i] = x[:, i] * s / bnt
    else:
        scale = 1e-8 if scale == 0.0 else scale
        x = x * scale / bnt
    return x


def stable_sigmoid(x):
    sig = np.where(x < 0, np.exp(x) / (1 + np.exp(x)), 1 / (1 + np.exp(-x)))
    return sig


def calculate_quant_cos_error(orig_tensor, qdq_tensor):
    cos_sim = np.inner(orig_tensor.flatten(), qdq_tensor.flatten()) \
              / (np.linalg.norm(orig_tensor.flatten()) * np.linalg.norm(qdq_tensor.flatten()))
    return cos_sim
