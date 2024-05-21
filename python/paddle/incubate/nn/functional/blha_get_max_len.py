# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


def blha_get_max_len(seq_lens_encoder, seq_lens_decoder, batch_size):
    """
    Apply Fused BlhaGetMaxLen kernel. Typically used before the block_multihead_attention operator.

    Args:
        seq_lens_encoder (Tensor): Sentence length of the encoder.
        seq_lens_decoder (Tensor): Sentence length of the decoder.
        batch_size (Tensor): the batch size.

    Returns:
        Tensor|(max_enc_len_this_time, max_dec_len_this_time)

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')

            >>> seq_lens_encoder = paddle.cast(paddle.randn(shape=[10]), dtype=paddle.int32)
            >>> seq_lens_decoder = paddle.cast(paddle.randn(shape=[10]), dtype=paddle.int32)
            >>> bsz = 10
            >>> batch_size = paddle.ones(shape=[bsz])
            >>> max_enc_len_this_time, max_dec_len_this_time = paddle.incubate.nn.functional.blha_get_max_len(seq_lens_encoder, seq_lens_decoder, batch_size)
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.blha_get_max_len(
            seq_lens_encoder, seq_lens_decoder, batch_size
        )

    helper = LayerHelper('blha_get_max_len', **locals())
    max_enc_len_this_time = helper.create_variable_for_type_inference(
        dtype="int32"
    )
    max_dec_len_this_time = helper.create_variable_for_type_inference(
        dtype="int32"
    )

    inputs = {}
    inputs['seq_lens_encoder'] = seq_lens_encoder
    inputs['seq_lens_decoder'] = seq_lens_decoder
    inputs['batch_size'] = batch_size

    outputs = {
        'max_enc_len_this_time': max_enc_len_this_time,
        'max_dec_len_this_time': max_dec_len_this_time,
    }
    helper.append_op(
        type='blha_get_max_len',
        inputs=inputs,
        outputs=outputs,
    )
    return max_enc_len_this_time, max_dec_len_this_time
