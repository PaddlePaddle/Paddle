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


def encode_rotary_qk(
     q,
     kv,
     rotary_emb,
     seq_lens,
     rotary_emb_dims,
     use_neox,
 ):

    r"""
     Apply EncodeRotaryQkKernel kernel.

     Args:
         input_q (Tensor): the input q Tensor.
         input_kv (Tensor): the input kv Tensor.
         rotary_emb (Tensor): the input cache_kv Tensor.
         seq_lens (Tensor): the input sequence_lengths Tensor.
         rotary_emb_dims (int): the input rotary_emb_dims.
         use_neox (bool): the input use_neox.

     Returns:
         

     Examples:
         .. code-block:: python

             >>> # doctest: +REQUIRES(env:GPU)
             >>> import paddle
             >>> paddle.device.set_device('gpu')

     """
    if in_dynamic_or_pir_mode():

        return _C_ops.encode_rotary_qk(
             q,
             kv,
             rotary_emb,
             seq_lens,
             rotary_emb_dims,
             use_neox,
        )

    helper = LayerHelper('encode_rotary_qk', **locals())

    inputs = {
         'q': q,
         'kv': kv,
         'rotary_emb': rotary_emb,
         'seq_lens': seq_lens,
         'rotary_emb_dims': rotary_emb_dims,
         'use_neox': use_neox,
     }

    rotary_q_out = helper.create_variable_for_type_inference(
         dtype=q.dtype
    )
    rotary_kv_out = helper.create_variable_for_type_inference(
         dtype=kv.dtype
    )

    outputs_dict = {
        'rotary_q_out': rotary_q_out, 
        'rotary_kv_out': rotary_kv_out
    }
     # "rotary_q_out", "rotary_kv_out"
    helper.append_op(
         type='encode_rotary_qk',
         inputs=inputs,
         outputs=outputs_dict,
    )

    return rotary_q_out, rotary_kv_out