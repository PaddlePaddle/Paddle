# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from functools import partial, reduce
import paddle
from paddle.utils import deprecated
from . import nn
from .layer_function_generator import templatedoc
from ..layer_helper import LayerHelper
from ..framework import (
    Variable,
    _non_static_mode,
    static_only,
    _in_legacy_dygraph,
    in_dygraph_mode,
)
from .. import core
from ..data_feeder import check_variable_and_dtype, check_type
from ..param_attr import ParamAttr
from ..initializer import NumpyArrayInitializer, Constant
from .. import core
import warnings
from paddle import _C_ops, _legacy_C_ops

__all__ = [
    'cross_entropy',
]

kIgnoreIndex = -100


def cross_entropy(input, label, soft_label=False, ignore_index=kIgnoreIndex):
    r"""
    :alias_main: paddle.nn.functional.cross_entropy
        :alias: paddle.nn.functional.cross_entropy,paddle.nn.functional.loss.cross_entropy
        :old_api: paddle.fluid.layers.cross_entropy

    This operator computes the cross entropy between input and label. It
    supports both hard-label and and soft-label cross entropy computation.

    1. Hard-label cross entropy: if soft_label=False, :math:`label[i_1, i_2, ..., i_k]`
       is the hard label of each sample.

        .. math::

           output[i_1, i_2, ..., i_k]=-log(input[i_1, i_2, ..., i_k, j]), label[i_1, i_2, ..., i_k] = j, j != ignore\_index

    2. Soft-label cross entropy: if soft_label=True,  :math:`label[i_1, i_2, ..., i_k, j]`
       is the soft label of each sample corresponding to the j-th class.

        .. math::

           output[i_1, i_2, ..., i_k]= -\sum_{j}label[i_1,i_2,...,i_k,j]*log(input[i_1, i_2, ..., i_k,j])

    Args:
        input (Variable): a multidimensional Tensor with shape
                :math:`[N_1, N_2, ..., N_k, D]`, where the last dimension D is
                the class number. The data type should be float32 or float64.
        label (Variable): label value corresponding to input. If
                soft_label=False, the dimension of label should be :math:`[N_1, N_2, ..., N_k]`
                or :math:`[N_1, N_2, ..., N_k, 1]` , and its data type should be int64,
                and the value must be inside [0, D). If soft_label=True, the shape,
                data type of label should be the same with input, and the sum of
                soft label value of each sample should be 1.
        soft_label (bool): indicate whether label is soft. Default False, meaning that
                the label is hard. If soft_label=True, the label is soft.
        ignore_index (int): specify an ignorable label value. The ignored label would be
                omitted when computing. If it is a negative integer, no label would
                be ignored. Only valid when soft_label=False. Default -100.

    Returns:
         A Variable holding Tensor representing the cross entropy, whose data type is the same with input.
         If soft_label=False, the shape of output is the same with label.
         If soft_label=True, the shape of output is :math:`[N_1, N_2, ..., N_k, 1]` .

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            class_num = 7
            x = fluid.data(name='x', shape=[None, 3, 10], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            predict = fluid.layers.fc(input=x, size=class_num, act='softmax')
            cost = fluid.layers.cross_entropy(input=predict, label=label)
    """
    if not soft_label:
        return cross_entropy2(input, label, ignore_index)

    if _non_static_mode():
        return _legacy_C_ops.cross_entropy(
            input, label, "soft_label", soft_label, "ignore_index", ignore_index
        )

    inputs = {'X': [input], 'Label': [label]}
    attrs = {"soft_label": soft_label, "ignore_index": ignore_index}

    check_variable_and_dtype(
        input, 'input', ['float16', 'float32', 'float64'], 'cross_entropy'
    )
    helper = LayerHelper('cross_entropy', **locals())
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='cross_entropy', inputs=inputs, outputs={'Y': [out]}, attrs=attrs
    )
    return out


def cross_entropy2(input, label, ignore_index=kIgnoreIndex):
    if _non_static_mode():
        loss, _, _ = _legacy_C_ops.cross_entropy2(
            input, label, 'ignore_index', ignore_index
        )
        return loss

    inputs = {'X': [input], 'Label': [label]}
    attrs = {'ignore_index': ignore_index}
    check_variable_and_dtype(
        input, 'input', ['float16', 'float32', 'float64'], 'cross_entropy2'
    )
    helper = LayerHelper('cross_entropy2', **locals())
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    xshape = helper.create_variable_for_type_inference(dtype=input.dtype)
    match_x = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='cross_entropy2',
        inputs=inputs,
        outputs={'Y': [out], 'MatchX': [match_x], 'XShape': [xshape]},
        attrs=attrs,
    )
    return out
