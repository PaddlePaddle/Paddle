# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# File: python/paddle/incubate/nn/functional/layer_norm_cuda.py
from paddle import _C_ops
from paddle.base.data_feeder import convert_dtype
from paddle.base.framework import in_dynamic_or_pir_mode
from paddle.base.layer_helper import LayerHelper


def fused_rms_norm_ext(x, scale, epsilon=1e-5, name=None):
    """
    Applies Layer Normalization over the last dimension of the input tensor using CUDA implementation.
    Args:
        x (Tensor): Input tensor of shape [rows, cols] or higher dimensions (flattened to 2D).
        scale (Tensor): Scale tensor of shape [cols].
        bias (Tensor, optional): Bias tensor of shape [cols]. If None, no bias is added.
        epsilon (float): Small constant to avoid division by zero.
        name (str, optional): Name of the operator.
    Returns:
        y (Tensor): Normalized tensor of same shape as x.
        mean (Tensor): Tensor of shape [rows], the mean of each row.
        invvar (Tensor): Tensor of shape [rows], the inverse standard deviation of each row.
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.fused_rms_norm_ext(x, scale, epsilon)
    helper = LayerHelper('fused_rms_norm_ext', **locals())
    dtype = convert_dtype(x.dtype)
    y = helper.create_variable_for_type_inference(dtype)
    invvar = helper.create_variable_for_type_inference('float32')

    inputs = {'x': x, 'scale': scale}

    helper.append_op(
        type='fused_rms_norm_ext',
        inputs=inputs,
        outputs={'y': y, 'invvar': invvar},
        attrs={'epsilon': epsilon},
    )
    return y, invvar

def rms_norm_paddle(x, scale, bias=None, epsilon=1e-5):
    # 计算均方根
    variance = paddle.mean(paddle.square(x), axis=-1, keepdim=True)
    # 计算 RMS
    rms = paddle.sqrt(variance + epsilon)
    # 归一化
    y = x / rms
    # 应用缩放
    y = y * scale.reshape([1, -1])
    # 应用偏置（如果有）
    if bias is not None:
        y = y + bias.reshape([1, -1])

    # 返回归一化后的张量、均值（RMS Norm 中为0）和逆标准差
    return y, (1.0 / rms).squeeze(-1)

import paddle
import numpy as np
paddle.seed(42)

# 生成测试数据
batch_size, seq_len, hidden_size = 2, 3, 4
x_np = np.random.randn(batch_size, seq_len, hidden_size).astype('float32')
weight_np = np.random.randn(hidden_size).astype('float32')



# 转换为各框架的 tensor
x_paddle = paddle.to_tensor(x_np, stop_gradient=False)
weight_paddle = paddle.to_tensor(weight_np, stop_gradient=False)
x_ops = x_paddle.clone()
x_ops.stop_gradient = False
weight_ops = weight_paddle.clone()
weight_ops.stop_gradient = False

# 前向计算
y_paddle, invvar_paddle = rms_norm_paddle(x_paddle, weight_paddle)
y_ops, invvar_ops = fused_rms_norm_ext(x_ops, weight_ops)
loss_paddle = paddle.mean(y_paddle)+paddle.mean(invvar_paddle)
y_paddle.backward()
loss_ops = paddle.mean(y_ops)+paddle.mean(invvar_ops)
y_ops.backward()
paddle.device.synchronize()

# print(y_paddle.numpy())
# print(".......")
# print(y_ops.numpy())


print(x_paddle.grad)
print(".......")
print(x_ops.grad)

print(paddle.allclose(y_paddle, y_ops, atol=1e-6))
print(paddle.allclose(x_ops.grad, x_paddle.grad, atol=1e-5))