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

import warnings
from ...fluid.layer_helper import LayerHelper
from ...fluid.framework import in_dygraph_mode, convert_np_dtype_to_dtype_
from ...fluid import core
from ...fluid.data_feeder import check_variable_and_dtype, check_dtype
import paddle
from paddle import _C_ops

__all__ = []


def fused_ffn(x,
              linear1_weight,
              linear2_weight,
              seed1_data=None,
              seed2_data=None,
              linear1_bias=None,
              linear2_bias=None,
              ln1_scale=None,
              ln1_bias=None,
              ln2_scale=None,
              ln2_bias=None,
              dropout_prob1=0.5,
              dropout_prob2=0.5,
              act_method="relu",
              epsilon1=1e-5,
              epsilon2=1e-5,
              dropout_implementation1='upscale_in_train',
              dropout_implementation2='upscale_in_train',
              is_test1=False,
              is_test2=False,
              fix_seed1=False,
              fix_seed2=False,
              seed1=0,
              seed2=0,
              normalize_pre_or_post=False,
              name=None):

    if in_dygraph_mode():
        out, _, _, _, _, _, _, _, _, _, _ = _C_ops.fused_ffn(
            x,
            seed1_data,
            seed2_data,
            linear1_weight,
            linear1_bias,
            linear2_weight,
            linear2_bias,
            ln1_scale,
            ln1_bias,
            ln2_scale,
            ln2_bias,
            'dropout_prob1',
            dropout_prob1,
            'dropout_prob2',
            dropout_prob2,
            'normalize_pre_or_post',
            normalize_pre_or_post,
            'act_method',
            "relu",
            'epsilon1',
            epsilon1,
            'epsilon2',
            epsilon2,
            'dropout_implementation1',
            dropout_implementation1,
            'dropout_implementation2',
            dropout_implementation2,
            'is_test1',
            is_test1,
            'is_test2',
            is_test2,
            'fix_seed1',
            fix_seed1,
            'fix_seed2',
            fix_seed2,
            'seed1',
            seed1,
            'seed2',
            seed1, )
        return out

    helper = LayerHelper("fused_ffn", **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    dropout1_mask = helper.create_variable_for_type_inference(
        'uint8', stop_gradient=True)
    dropout2_mask = helper.create_variable_for_type_inference(
        'uint8', stop_gradient=True)
    ln1_mean = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln1_variance = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln2_mean = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln2_variance = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    linear1_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln1_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    dropout1_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    dropout2_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)

    helper.append_op(
        type='fused_ffn',
        inputs={
            'X': x,
            'seed1': seed1,
            'seed2': seed2,
            'Linear1Weight': linear1_weight,
            'Linear1Bias': linear1_bias,
            'Linear2Weight': linear2_weight,
            'Linear2Bias': linear2_bias,
            'Ln1Scale': ln1_scale,
            'Ln1Bias': ln1_bias,
            'Ln2Scale': ln2_scale,
            'Ln2Bias': ln2_bias,
        },
        outputs={
            'Out': out,
            'Dropout1Mask': dropout1_mask,
            'Dropout2Mask': dropout2_mask,
            'Ln1Mean': ln1_mean,
            'Ln1Variance': ln1_variance,
            'Ln2Mean': ln2_mean,
            'Ln2Variance': ln2_variance,
            'Linear1Out': linear1_out,
            'Ln1Out': ln1_out,
            'Dropout1Out': dropout1_out,
            'Dropout2Out': dropout2_out,
        },
        attrs={
            'dropout_prob1': dropout_prob1,
            'dropout_prob2': dropout_prob2,
            'act_method': act_method,
            'normalize_pre_or_post': normalize_pre_or_post,
            'epsilon1': epsilon1,
            'epsilon2': epsilon2,
            'dropout_implementation1': dropout_implementation1,
            'dropout_implementation2': dropout_implementation2,
            'is_test1': is_test1,
            'is_test2': is_test2,
            'fix_seed1': fix_seed1,
            'fix_seed2': fix_seed2,
            'seed1': seed1,
            'seed2': seed1,
        })

    return out
