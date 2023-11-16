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


def rebuild_padding(tmp_out, padding_offset, seq_lens, input_ids, name=None):
    """
    Apply rebuild_padding kernel.
    Args:
        tmp_out (Tensor): the input tmp_out Tensor.
        padding_offset (Tensor): the input padding_offset Tensor.
        seq_lens (Tensor): the input seq_lens Tensor.
        input_ids (Tensor): the input input_ids Tensor.
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.rebuild_padding(
            tmp_out, padding_offset, seq_lens, input_ids
        )

    helper = LayerHelper("rebuild_padding", **locals())

    inputs = {
        'tmp_out': tmp_out,
        'padding_offset': padding_offset,
        'seq_lens': seq_lens,
        'in_ids': input_ids,
    }
    out = helper.create_variable_for_type_inference(dtype=tmp_out.dtype)
    output_dict = {'out': out}
    helper.append_op(
        type='rebuild_padding',
        input=inputs,
        outputs=output_dict,
    )
    return out
