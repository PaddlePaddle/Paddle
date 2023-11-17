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
from paddle.framework import LayerHelper, in_dynamic_mode


def qkv_transpose_split(
    qkv,
    padding_offset,
    seq_lens,
    input_ids,
    num_head,
    head_size
):
    r"""
    Apply QkvTransposeSplitKernel kernel.
    Args:
        qkv (Tensor): the input qkv Tensor.
        padding_offset (Tensor): the padding_offset v Tensor.
        seq_lens (Tensor): the input seq_lens Tensor.
        input_ids (Tensor): the input input_ids Tensor.
        num_head (int): The num_head, Default 1.
    Returns:
        Tensor: the output Tensor.
    Examples:
        .. code-block:: python
            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')
    """
    if in_dynamic_mode():
        return _C_ops.qkv_transpose_split(
            qkv,
            padding_offset,
            seq_lens,
            input_ids,
            num_head,
            head_size,
        )

    helper = LayerHelper('qkv_transpose_split', **locals())

    inputs = {
        'qkv': qkv,
        'padding_offset': padding_offset,
        'seq_lens': seq_lens,
        'input_ids': input_ids,
    }

    q_out = helper.create_variable_for_type_inference(
        dtype=qkv.dtype
    )
    k_out = helper.create_variable_for_type_inference(
        dtype=qkv.dtype
    )
    v_out = helper.create_variable_for_type_inference(
        dtype=qkv.dtype
    )

    outputs_dict = {
        'q_out': q_out,
        'k_out': k_out,
        'v_out': v_out,
        }

    helper.append_op(
        type='qkv_transpose_split',
        inputs=inputs,
        outputs=outputs_dict,
        attrs={
            'num_head': num_head,
            'head_size': head_size,
        },
    )

    return (q_out, k_out, v_out)