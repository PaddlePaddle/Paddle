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
import paddle
import paddle.fluid as fluid
from paddle.nn import Layer, LayerList
from paddle.fluid.layer_helper import LayerHelper
from paddle.nn import initializer as I
from paddle.fluid.param_attr import ParamAttr
from paddle import _C_ops


def fused_bn_act(x,
                 scale,
                 bias,
                 mean,
                 var,
                 momentum,
                 epsilon,
                 act_type,
                 use_global_stats=None,
                 training=False,
                 trainable_statistics=False):

    if fluid.framework.in_dygraph_mode():
        attrs = ("epsilon", epsilon, 'momentum', momentum, 'act_type', act_type,
                 'use_global_stats', use_global_stats, 'is_test', not training,
                 "trainable_statistics", trainable_statistics)
        out, _, _, _, _, _ = \
                              getattr(_C_ops, "fused_batch_norm_act")(x, scale, bias, mean, var, mean, var, *attrs)
        return out

    helper = LayerHelper('fused_batch_norm_act', **locals())
    bn_param_dtype = fluid.core.VarDesc.VarType.FP32

    x_shape = x.shape
    channel_num = x_shape[1]
    param_shape = [channel_num]

    # create output
    # mean and mean_out share the same memory
    mean_out = mean
    # variance and variance out share the same memory
    variance_out = var
    saved_mean = helper.create_variable_for_type_inference(
        dtype=bn_param_dtype, stop_gradient=True)
    saved_variance = helper.create_variable_for_type_inference(
        dtype=bn_param_dtype, stop_gradient=True)
    reserve_space = helper.create_variable_for_type_inference(
        dtype=fluid.core.VarDesc.VarType.FP16, stop_gradient=True)
    batch_norm_out = helper.create_variable_for_type_inference(
        fluid.core.VarDesc.VarType.FP16)

    inputs = {
        "X": x,
        "Scale": scale,
        "Bias": bias,
        "Mean": mean,
        "Variance": var,
    }
    attrs = {
        "epsilon": epsilon,
        'momentum': momentum,
        'use_global_stats': use_global_stats,
        'is_test': not training,
        "trainable_statistics": trainable_statistics
    }

    outputs = {
        "Y": batch_norm_out,
        "MeanOut": mean_out,
        "VarianceOut": variance_out,
        "SavedMean": saved_mean,
        "SavedVariance": saved_variance,
        "ReserveSpace": reserve_space,
    }

    helper.append_op(
        type="fused_batch_norm_act",
        inputs=inputs,
        outputs=outputs,
        attrs=attrs)

    return batch_norm_out


class FusedBNAct(Layer):
    def __init__(self,
                 param_shape,
                 momentum=0.9,
                 epsilon=1e-05,
                 param_attr=None,
                 bias_attr=None,
                 moving_mean_name=None,
                 moving_variance_name=None,
                 act=None,
                 name=None,
                 is_test=False,
                 use_global_stats=False,
                 trainable_statistics=False):
        super(FusedBNAct, self).__init__()
        self._momentum = momentum
        self._eps = epsilon
        self._act = "linear" if act is None else act
        self._use_global_stats = use_global_stats
        self._is_test = is_test
        self._trainable_statistics = trainable_statistics

        # init filter
        bn_param_dtype = fluid.core.VarDesc.VarType.FP32

        # create parameter
        if param_attr == False:
            self.scale = self.create_parameter(
                attr=None,
                shape=param_shape,
                dtype=bn_param_dtype,
                default_initializer=I.Constant(1.0))
            self.scale.stop_gradient = True
        else:
            self.scale = self.create_parameter(
                attr=param_attr,
                shape=param_shape,
                dtype=bn_param_dtype,
                default_initializer=I.Constant(1.0))
            self.scale.stop_gradient = param_attr != None and param_attr.learning_rate == 0.

        if bias_attr == False:
            self.bias = self.create_parameter(
                attr=None,
                shape=param_shape,
                dtype=bn_param_dtype,
                default_initializer=I.Constant(0.0),
                is_bias=True)
            self.bias.stop_gradient = True
        else:
            self.bias = self.create_parameter(
                attr=bias_attr,
                shape=param_shape,
                dtype=bn_param_dtype,
                is_bias=True)
            self.bias.stop_gradient = bias_attr != None and bias_attr.learning_rate == 0.

        self.mean = self.create_parameter(
            attr=ParamAttr(
                name=moving_mean_name,
                initializer=I.Constant(0.0),
                trainable=False),
            shape=param_shape,
            dtype=bn_param_dtype)
        self.mean.stop_gradient = True
        self.variance = self.create_parameter(
            attr=ParamAttr(
                name=moving_variance_name,
                initializer=I.Constant(1.0),
                trainable=False),
            shape=param_shape,
            dtype=bn_param_dtype)
        self.variance.stop_gradient = True

    def forward(self, x):
        out = fused_bn_act(
            x,
            self.scale,
            self.bias,
            self.mean,
            self.variance,
            self._momentum,
            self._eps,
            self._act,
            use_global_stats=self._use_global_stats,
            training=self.training,
            trainable_statistics=self._trainable_statistics)
        return out
