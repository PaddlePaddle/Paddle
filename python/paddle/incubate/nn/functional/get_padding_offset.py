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


def get_padding_offset(
    input_ids,
    cum_offsets,
    token_num,
    seq_len,
):
    if in_dynamic_or_pir_mode():
        return _C_ops.get_padding_offset(
            input_ids,
            cum_offsets,
            token_num,
            seq_len,
        )

    helper = LayerHelper("get_padding_offset", **locals())
    inputs = {
        'input_ids': input_ids,
        'cum_offsets': cum_offsets,
        'token_num': token_num,
        'seq_len': seq_len,
    }
    x_remove_padding = helper.create_variable_for_type_inference(
        dtype=input_ids.dtype,
    )
    cum_offsets_out = helper.create_variable_for_type_inference(
        dtype=seq_len.dtype,
    )
    padding_offset = helper.create_variable_for_type_inference(
        dtype=seq_len.dtype,
    )
    outputs_dict = {
        'x_remove_padding': x_remove_padding,
        'cum_offsets_out': cum_offsets_out,
        'padding_offset': padding_offset,
    }
    helper.append_op(
        type='get_padding_offset',
        inputs=inputs,
        outputs=outputs_dict,
    )
    return x_remove_padding, cum_offsets_out, padding_offset
