# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import _non_static_mode
from paddle.tensor.linalg import matmul
from paddle import _C_ops, _legacy_C_ops

__all__ = ['custom_fused_dense']

# const phi::DenseTensor* x = ctx.Input<phi::DenseTensor>("X");
# const phi::DenseTensor* y = ctx.Input<phi::DenseTensor>("Y");
# const phi::DenseTensor* bias = ctx.Input<phi::DenseTensor>("Bias");

# phi::DenseTensor* out = ctx.Output<phi::DenseTensor>("Out");
# phi::DenseTensor* reserve_space =
#     ctx.Output<phi::DenseTensor>("ReserveSpace");

# bool trans_x = ctx.Attr<bool>("trans_x");
# bool trans_y = ctx.Attr<bool>("trans_y");

# std::string activation = ctx.Attr<std::string>("activation");


def custom_fused_gelu_dense(x, y, bias, transx, transy, activation):
    if _non_static_mode():
        return _legacy_C_ops.custom_fused_dense(x, y, bias,
                                                transx, transy, activation)

    helper = LayerHelper('custom_fused_dense', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    gelu_in = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type='custom_fused_dense',
                     inputs={
                         'X': x,
                         'Y': y,
                         'Bias': bias,
                     },
                     outputs={
                         'Out': out,
                         'GeluIn': gelu_in,
                     },
                     attrs={
                         'transx': transx,
                         'transy': transy,
                         'activation': activation,
                     })
    return out

def custom_fused_dense(x, y, bias, transx, transy, activation):
    if _non_static_mode():
        return _legacy_C_ops.custom_fused_dense(x, y, bias,
                                                transx, transy, activation)

    helper = LayerHelper('custom_fused_dense', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type='custom_fused_dense',
                     inputs={
                         'X': x,
                         'Y': y,
                         'Bias': bias,
                     },
                     outputs={
                         'Out': out,
                     },
                     attrs={
                         'transx': transx,
                         'transy': transy,
                         'activation': activation,
                     })
    return out


def custom_fmha(qkv, cu_seq_len, host_seq_len, is_test, dropout_rate,
                zero_tensors, use_fmha_mke_opt):
    if _non_static_mode():
        return _legacy_C_ops.custom_fmha(qkv, cu_seq_len, host_seq_len, is_test,
                                         dropout_rate, zero_tensors,
                                         use_fmha_mke_opt)

    helper = LayerHelper('custom_fmha', **locals())
    ctx_out = helper.create_variable_for_type_inference(dtype=qkv.dtype)
    s_out = helper.create_variable_for_type_inference(dtype=qkv.dtype)
    dropout_mask = helper.create_variable_for_type_inference(dtype=qkv.dtype)
    dropout_out = helper.create_variable_for_type_inference(dtype=qkv.dtype)
    helper.append_op(type='custom_fmha',
                     inputs={
                         'QKV': qkv,
                         'CuSeqLen': cu_seq_len,
                         'HostSeqLen': host_seq_len
                     },
                     outputs={
                         'CtxOut': ctx_out,
                         'SOut': s_out,
                         'DropoutMask': dropout_mask,
                         'DropoutOut': dropout_out
                     },
                     attrs={
                         'is_test': is_test,
                         'dropout_rate': dropout_rate,
                         'zero_tensors': zero_tensors,
                         'use_fmha_mke_opt': use_fmha_mke_opt
                     })
    return ctx_out, s_out, dropout_mask, dropout_out


def custom_fused_dropout_residual_ln(hidden_states, input_tensor, weight, bias,
                                     epsilon, is_test, fix_seed, seed_val,
                                     is_upscale_in_train, hidden_dropout_prob):
    if _non_static_mode():
        return _legacy_C_ops.custom_fused_dropout_residual_ln(
            hidden_states, input_tensor, weight, bias, epsilon, is_test,
            fix_seed, seed_val, is_upscale_in_train, hidden_dropout_prob)

    helper = LayerHelper('custom_fused_dropout_residual_ln', **locals())
    out = helper.create_variable_for_type_inference(dtype=hidden_states.dtype)
    dropout_mask = helper.create_variable_for_type_inference(
        dtype=hidden_states.dtype)
    ln_mean = helper.create_variable_for_type_inference(dtype="float32")
    ln_var = helper.create_variable_for_type_inference(dtype="float32")
    dropout_residual_out = helper.create_variable_for_type_inference(
        dtype=hidden_states.dtype)

    helper.append_op(type='custom_fused_dropout_residual_ln',
                     inputs={
                         'X': hidden_states,
                         'Residual': input_tensor,
                         'LnScale': weight,
                         'LnBias': bias,
                     },
                     outputs={
                         'Out': out,
                         'DropoutMask': dropout_mask,
                         'LnMean': ln_mean,
                         'LnVar': ln_var,
                         'DropoutResidualOut': dropout_residual_out,
                     },
                     attrs={
                         'ln_epsilon': epsilon,
                         'is_test': is_test,
                         'fix_seed': fix_seed,
                         'seed_val': seed_val,
                         'is_upscale_in_train': is_upscale_in_train,
                         'dropout_rate': hidden_dropout_prob,
                     })
    return out, dropout_mask, ln_mean, ln_var, dropout_residual_out


def acc_merge(acc, total, out, step):
    if _non_static_mode():
        return _legacy_C_ops.acc_merge(acc, total, out, step)

    helper = LayerHelper('acc_merge', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type='acc_merge',
                     inputs={
                         'Acc': acc,
                         'Total': total
                     },
                     outputs={
                         'Out': out,
                         'Step': step
                     })
    return out


def custom_lr(x, out, base_lr, max_step):
    if _non_static_mode():
        return _legacy_C_ops.custom_lr(x, out, base_lr, max_step)

    helper = LayerHelper('custom_lr', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type='custom_lr',
                     inputs={'X': x},
                     outputs={'Out': out},
                     attrs={
                         'base_lr': base_lr,
                         'max_step': max_step
                     })
    return out
