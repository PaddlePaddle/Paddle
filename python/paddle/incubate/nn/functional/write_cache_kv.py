# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


from paddle import _C_ops
from paddle.framework import LayerHelper, in_dynamic_or_pir_mode


def write_cache_kv(
    input_k,
    input_v,
    cache_kv,
    sequence_lengths,
):
    r"""
    Apply WriteCacheKVKernel kernel.

    Args:
        input_k (Tensor): the input k Tensor.
        input_v (Tensor): the input v Tensor.
        cache_kv (Tensor): the input cache_kv Tensor.
        sequence_lengths (Tensor): the input sequence_lengths Tensor.

    Returns:
        Tensor: the output Tensor.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')

            >>> paddle_x = paddle.cast(paddle.randn(shape=[32, 256]), dtype=paddle.float16)
            >>> paddle_weight = paddle.cast(paddle.randn(shape=[256]), dtype=paddle.float16)
            >>> paddle_bias = paddle.cast(paddle.randn(shape=[256]), dtype=paddle.float16)
            >>> epsilon = 1e-6
            >>> paddle_rmsnorm = paddle.incubate.nn.functional.fused_rms_norm(paddle_x, paddle_weight, paddle_bias, epsilon, 1)
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.write_cache_kv(
            input_k,
            input_v,
            cache_kv,
            sequence_lengths,
        )

    helper = LayerHelper('write_cache_kv', **locals())

    inputs = {
        'input_k': input_k,
        'input_v': input_v,
        'cache_kv': cache_kv,
        'sequence_lengths': sequence_lengths,
    }

    cache_kv_out = helper.create_variable_for_type_inference(
        dtype=cache_kv.dtype
    )

    outputs_dict = {'cache_kv_out': cache_kv_out}

    helper.append_op(
        type='write_cache_kv',
        inputs=inputs,
        outputs=outputs_dict,
    )

    return cache_kv_out
