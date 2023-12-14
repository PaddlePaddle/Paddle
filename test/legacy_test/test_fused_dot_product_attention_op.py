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

import unittest

import numpy as np
from op_test import (
    OpTest,
    convert_float_to_uint16,
    convert_uint16_to_float,
)

import paddle
import paddle.nn.functional as F
from paddle.incubate.nn.functional import fused_dot_product_attention

np.random.seed(2023)


def skip_unit_test():
    return (
        not paddle.is_compiled_with_cuda()
        or paddle.device.cuda.get_device_capability() != (8, 0)
        or paddle.get_cudnn_version() < 8906
    )


skip_msg = (
    "only support with cuda and CUDNN 8.9.6 or later,"
    " and only Ampere and later GPU is supported."
)


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedAttentionOpFP16(OpTest):
    def _set_shape(self):
        self.batch_size = 8
        self.q_seqlen = 128
        self.kv_seqlen = 128
        self.num_heads = 12
        self.head_size = 64

    def _set_config(self):
        self.has_attn_mask = False
        self.is_causal_masking = False
        self.dropout_prob = 0.0
        self.dtype = "float16"
        self.rtol = 5e-4
        self.atol = 5e-4

    def setUp(self):
        self._set_shape()
        self._set_config()
        # has_attn_mask and is_causal_masking can't be True at the same time
        assert not (self.has_attn_mask and self.is_causal_masking)
        self.training = True
        self.scaling_factor = self.head_size**-0.5
        self.q_shape = (
            self.batch_size,
            self.q_seqlen,
            self.num_heads,
            self.head_size,
        )
        self.kv_shape = (
            self.batch_size,
            self.kv_seqlen,
            self.num_heads,
            self.head_size,
        )
        self._generate_input_data()
        self.__class__.op_type = "fused_dot_product_attention"
        # use autograd to check grad in this unittest.
        self.__class__.no_need_check_grad = True

    def _generate_input_data(self):
        def _random(shape, mask=None):
            if self.dtype == "bfloat16":
                data = np.random.normal(loc=0.0, scale=0.02, size=shape).astype(
                    "float32"
                )
                # mask has the same shape as data, if the mask value is 0, the
                # corresponding data value will be set to 0.
                if mask is not None:
                    data = data * mask
                return convert_float_to_uint16(data)
            else:
                data = np.random.random(shape).astype("float32")
                if mask is not None:
                    data = data * mask
                return data.astype(self.dtype)

        self.q = _random(self.q_shape)
        self.k = _random(self.kv_shape)
        self.v = _random(self.kv_shape)

        self.attn_mask = np.ones(
            shape=(self.batch_size, 1, self.q_seqlen, self.kv_seqlen),
            dtype=np.int32,
        )
        self.q_actual_seqlen = np.full(
            shape=(self.batch_size,), fill_value=self.q_seqlen, dtype=np.int32
        )
        self.kv_actual_seqlen = np.full(
            shape=(self.batch_size,), fill_value=self.kv_seqlen, dtype=np.int32
        )
        self.attn_mask = np.ones(
            shape=(self.batch_size, 1, self.q_seqlen, self.kv_seqlen),
            dtype=np.int32,
        )
        if self.has_attn_mask:
            self.q_actual_seqlen = np.random.randint(
                low=20,
                high=self.q_seqlen,
                size=(self.batch_size,),
                dtype=np.int32,
            )
            self.kv_actual_seqlen = np.random.randint(
                low=20,
                high=self.kv_seqlen,
                size=(self.batch_size,),
                dtype=np.int32,
            )
            self.attn_mask = np.zeros(
                shape=(self.batch_size, 1, self.q_seqlen, self.kv_seqlen),
                dtype=np.int32,
            )
            for i in range(0, self.batch_size):
                self.attn_mask[
                    i,
                    0,
                    0 : self.q_actual_seqlen[i],
                    0 : self.kv_actual_seqlen[i],
                ] = 1

        # need to set invalid position of dout to 0
        dout_shape = (
            self.batch_size,
            self.q_seqlen,
            self.num_heads,
            self.head_size,
        )
        dout_mask = None
        if self.has_attn_mask:
            dout_mask = np.ones(shape=dout_shape, dtype=np.int32)
            for i in range(0, self.batch_size):
                dout_mask[i, self.q_actual_seqlen[i] :, :, :] = 0
        self.dout = _random(dout_shape, dout_mask)

    def _get_reference_out(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        q_tensor = paddle.to_tensor(self.q, stop_gradient=False)
        k_tensor = paddle.to_tensor(self.k, stop_gradient=False)
        v_tensor = paddle.to_tensor(self.v, stop_gradient=False)

        q_out = paddle.transpose(
            x=q_tensor, perm=[0, 2, 1, 3]
        )  # [b, s, h, d] -> [b, h, s, d]
        k_out = paddle.transpose(
            x=k_tensor, perm=[0, 2, 1, 3]
        )  # [b, s, h, d] -> [b, h, s, d]
        v_out = paddle.transpose(
            x=v_tensor, perm=[0, 2, 1, 3]
        )  # [b, s, h, d] -> [b, h, s, d]

        qk_out = paddle.matmul(
            x=q_out * self.scaling_factor,
            y=k_out,
            transpose_x=False,
            transpose_y=True,
        )

        if self.is_causal_masking:
            self.attn_mask = np.tril(self.attn_mask, k=0)

        if self.has_attn_mask or self.is_causal_masking:
            attn_mask = paddle.to_tensor(self.attn_mask, stop_gradient=True)
            attn_mask = (paddle.cast(attn_mask, self.dtype) - 1.0) * 1e4
            attn_mask_out = qk_out + attn_mask
            softmax_out = F.softmax(attn_mask_out)
        else:
            softmax_out = F.softmax(qk_out)

        if self.dropout_prob:
            dropout_out = F.dropout(
                softmax_out,
                self.dropout_prob,
                training=self.training,
                mode="upscale_in_train",
            )
            qkv_out = paddle.matmul(dropout_out, v_out)
        else:
            qkv_out = paddle.matmul(softmax_out, v_out)

        mha_out = paddle.transpose(
            qkv_out, perm=[0, 2, 1, 3]
        )  # [b, h, s, d] -> [b, s, h, d]

        paddle.autograd.backward(
            [mha_out],
            [paddle.to_tensor(self.dout, dtype=self.dtype)],
            retain_graph=True,
        )

        # need to set invalid position of output to 0
        valid_mha_out = paddle.full_like(mha_out, 0)
        for i in range(0, self.batch_size):
            valid_mha_out[i, 0 : self.q_actual_seqlen[i], :, :] = mha_out[
                i, 0 : self.q_actual_seqlen[i], :, :
            ]

        return (
            valid_mha_out,
            q_tensor.grad,
            k_tensor.grad,
            v_tensor.grad,
            softmax_out,
        )

    def _get_fused_attn_out(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        q_tensor = paddle.to_tensor(self.q, stop_gradient=False)
        k_tensor = paddle.to_tensor(self.k, stop_gradient=False)
        v_tensor = paddle.to_tensor(self.v, stop_gradient=False)

        attn_mask = paddle.to_tensor(self.attn_mask, stop_gradient=True)

        (
            fmha_out,
            softmax_out,
        ) = fused_dot_product_attention(
            q_tensor,
            k_tensor,
            v_tensor,
            attn_mask,
            self.scaling_factor,
            self.dropout_prob,
            True,
            self.is_causal_masking,
            True,
        )

        paddle.autograd.backward(
            [fmha_out], [paddle.to_tensor(self.dout)], retain_graph=True
        )

        return (
            fmha_out,
            q_tensor.grad,
            k_tensor.grad,
            v_tensor.grad,
            softmax_out,
        )

    def _compare_output(self):
        def _convert(value):
            if self.dtype == "bfloat16":
                return convert_uint16_to_float(value)
            return value

        output_names = [
            "fmha_out",
            "q_grad",
            "k_grad",
            "v_grad",
        ]

        outputs_ref = self._get_reference_out()
        outputs_fused = self._get_fused_attn_out()

        for i in range(len(output_names)):
            ref_res = outputs_ref[i]
            fused_res = outputs_fused[i]
            np.testing.assert_allclose(
                _convert(ref_res.numpy()),
                _convert(fused_res.numpy()),
                atol=self.atol,
                rtol=self.rtol,
                err_msg=f"Checking < {output_names[i]} > failed",
            )

    def test_output(self):
        self._compare_output()


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedAttentionOpFP16WithPaddingMask(TestFusedAttentionOpFP16):
    def _set_config(self):
        self.has_attn_mask = True
        self.is_causal_masking = False
        self.dropout_prob = 0.0
        self.dtype = "float16"
        self.rtol = 5e-3
        self.atol = 5e-3


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedAttentionOpFP16WithCausalMask(TestFusedAttentionOpFP16):
    def _set_config(self):
        self.has_attn_mask = False
        self.is_causal_masking = True
        self.dropout_prob = 0.0
        self.dtype = "float16"
        self.rtol = 5e-3
        self.atol = 5e-3


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedAttentionOpBF16(TestFusedAttentionOpFP16):
    def _set_config(self):
        self.has_attn_mask = False
        self.is_causal_masking = False
        self.dropout_prob = 0.0
        self.dtype = "bfloat16"
        self.rtol = 5e-3
        self.atol = 5e-3


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedAttentionOpBF16WithPaddingMask(TestFusedAttentionOpFP16):
    def _set_config(self):
        self.has_attn_mask = True
        self.is_causal_masking = False
        self.dropout_prob = 0.0
        self.dtype = "bfloat16"
        self.rtol = 5e-3
        self.atol = 5e-3


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedAttentionOpBF16WithCausalMask(TestFusedAttentionOpFP16):
    def _set_config(self):
        self.has_attn_mask = False
        self.is_causal_masking = True
        self.dropout_prob = 0.0
        self.dtype = "bfloat16"
        self.rtol = 5e-3
        self.atol = 5e-3


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedAttentionOpBF16WithPaddingMaskCase2(
    TestFusedAttentionOpBF16WithPaddingMask
):
    def _set_shape(self):
        self.batch_size = 2
        self.q_seqlen = 1024
        self.kv_seqlen = 1024
        self.num_heads = 4
        self.head_size = 64


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedAttentionOpBF16WithPaddingMaskCase3(
    TestFusedAttentionOpBF16WithPaddingMask
):
    def _set_shape(self):
        self.batch_size = 1
        self.q_seqlen = 2048
        self.kv_seqlen = 2048
        self.num_heads = 2
        self.head_size = 128


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedAttentionOpBF16WithCausalMaskCase2(
    TestFusedAttentionOpBF16WithCausalMask
):
    def _set_shape(self):
        self.batch_size = 2
        self.q_seqlen = 1024
        self.kv_seqlen = 1024
        self.num_heads = 4
        self.head_size = 128


if __name__ == "__main__":
    unittest.main()
