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
from paddle import core
from paddle.framework import LayerHelper, in_dynamic_mode


def fused_get_rotary_embedding(
    input_ids,
    position_ids,
    head_dim_shape_tensor,
    prompt_num,
    use_neox
):
    r"""
    Apply FusedGetRotaryEmbeddingKernel kernel.
    Args:
        input_ids (Tensor): the input input_ids Tensor.
        position_ids (Tensor): the position_ids v Tensor.
        head_dim_shape_tensor (Tensor): the input head_dim_shape_tensor Tensor.
        prompt_num (int): the prompt_num.
        use_neox (bool): The use_neox.
    Returns:
        Tensor: the output Tensor.
    Examples:
        .. code-block:: python
            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')
    """
    if in_dynamic_mode():
        return _C_ops.fused_get_rotary_embedding(
            input_ids,
            position_ids,
            head_dim_shape_tensor,
            prompt_num,
            use_neox
        )

    helper = LayerHelper('fused_get_rotary_embedding', **locals())

    inputs = {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'head_dim_shape_tensor': head_dim_shape_tensor,
    }

    rotary_embedding = helper.create_variable_for_type_inference(
        dtype=core.VarDesc.VarType.FLOAT32
    )

    outputs_dict = {
        'rotary_embedding': rotary_embedding,
        }

    helper.append_op(
        type='fused_get_rotary_embedding',
        inputs=inputs,
        outputs=outputs_dict,
        attrs={
            'prompt_num': prompt_num,
            'use_neox': use_neox,
        },
    )

    return (rotary_embedding)