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
from op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float

import paddle
import paddle.nn.functional as F
from paddle import _legacy_C_ops

np.random.seed(2023)


def skip_unit_test():
    return (
        not paddle.is_compiled_with_cuda()
        or paddle.device.cuda.get_device_capability() != (8, 0)
        or paddle.get_cudnn_version() < 8901
    )


skip_msg = (
    "only support with cuda and CUDNN 8.9.1 or later,"
    " and only Ampere devices are supported"
)


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedSelfAttentionOp(OpTest):
    def _set_shape(self):
        self.batch_size = 8
        self.q_seqlen = 128
        self.kv_seqlen = 128
        self.num_heads = 16
        self.head_size = 64

    def _set_config(self):
        self.has_attn_mask = False
        self.attn_mode = "self_attn"  # self_attn or cross_attn
        self.dropout_prob = 0.0
        self.dtype = "float16"
        self.rtol = 5e-3
        self.atol = 5e-3

    def setUp(self):
        self._set_shape()
        self._set_config()
        assert self.attn_mode in [
            "self_attn",
            "cross_attn",
        ], "attn_mode should be self_attn or cross_attn"
        self.training = True
        self.scaling_factor = self.head_size**-0.5
        self.embed_dim = self.head_size * self.num_heads
        self.q_size = self.batch_size * self.q_seqlen * self.embed_dim
        self.kv_size = self.batch_size * self.kv_seqlen * self.embed_dim
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
        def _random(shape):
            if self.dtype == "bfloat16":
                data = np.random.normal(loc=0.0, scale=0.02, size=shape).astype(
                    "float32"
                )
                return convert_float_to_uint16(data)
            else:
                return np.random.random(shape).astype(self.dtype)

        self.q = _random(self.q_shape)

        if self.attn_mode == "self_attn":
            self.kv = self.q
        else:
            self.kv = _random(self.kv_shape)

        self.q_actual_seqlen = (
            np.ones(shape=(self.batch_size,), dtype=np.int32) * self.q_seqlen
        )
        self.kv_actual_seqlen = (
            np.ones(shape=(self.batch_size,), dtype=np.int32) * self.kv_seqlen
        )
        self.attn_mask = np.ones(
            shape=(self.batch_size, 1, self.q_seqlen, self.kv_seqlen),
            dtype=np.int32,
        )
        if self.has_attn_mask:
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
        self.dout = _random(
            (self.batch_size, self.q_seqlen, self.num_heads, self.head_size)
        )

    def _get_reference_out(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        q_tensor = paddle.to_tensor(self.q, stop_gradient=False)
        k_tensor = paddle.to_tensor(self.kv, stop_gradient=False)
        v_tensor = paddle.to_tensor(self.kv, stop_gradient=False)

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

        if self.has_attn_mask:
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
        return mha_out, q_tensor.grad, k_tensor.grad, v_tensor.grad, softmax_out

    def _get_cudnn_fmha_out(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))

        if self.attn_mode == "self_attn":
            qkv = np.stack(
                [self.q, self.kv, self.kv], axis=2
            )  # [b, s, 3, h, d]
            qkv_tensor = paddle.to_tensor(qkv, stop_gradient=False)
        else:
            q_tensor = paddle.to_tensor(self.q, stop_gradient=False)
            kv = np.stack([self.kv, self.kv], axis=2)  # [b, s, 2, h, d]
            kv_tensor = paddle.to_tensor(kv, stop_gradient=False)

        q_actual_seqlen_tensor = paddle.to_tensor(
            self.q_actual_seqlen, dtype="int32", stop_gradient=True
        )
        kv_actual_seqlen_tensor = paddle.to_tensor(
            self.kv_actual_seqlen, dtype="int32", stop_gradient=True
        )

        if self.attn_mode == "self_attn":
            (
                softmax_out,
                fmha_out,
            ) = _legacy_C_ops.fused_dot_product_self_attention(
                qkv_tensor,
                q_actual_seqlen_tensor,
                kv_actual_seqlen_tensor,
                'scaling_factor',
                self.scaling_factor,
                'attn_dropout_rate',
                self.dropout_prob,
            )
        else:
            (
                softmax_out,
                fmha_out,
            ) = _legacy_C_ops.fused_dot_product_cross_attention(
                q_tensor,
                kv_tensor,
                q_actual_seqlen_tensor,
                kv_actual_seqlen_tensor,
                'scaling_factor',
                self.scaling_factor,
                'attn_dropout_rate',
                self.dropout_prob,
            )

        paddle.autograd.backward(
            [fmha_out], [paddle.to_tensor(self.dout)], retain_graph=True
        )

        if self.attn_mode == "self_attn":
            q_grad = qkv_tensor.grad[:, :, 0, :, :]
            k_grad = qkv_tensor.grad[:, :, 1, :, :]
            v_grad = qkv_tensor.grad[:, :, 2, :, :]
        else:
            q_grad = q_tensor.grad
            k_grad = kv_tensor.grad[:, :, 0, :, :]
            v_grad = kv_tensor.grad[:, :, 1, :, :]

        return fmha_out, q_grad, k_grad, v_grad, softmax_out

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
        outputs_fused = self._get_cudnn_fmha_out()

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
class TestFusedSelfAttentionOpWithMask(TestFusedSelfAttentionOp):
    def _set_config(self):
        self.has_attn_mask = True
        self.attn_mode = "self_attn"  # self_attn or cross_attn
        self.dropout_prob = 0.0
        self.dtype = "float16"
        self.rtol = 5e-3
        self.atol = 5e-3


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedCrossAttentionOp(TestFusedSelfAttentionOp):
    def _set_config(self):
        self.has_attn_mask = False
        self.attn_mode = "cross_attn"  # self_attn or cross_attn
        self.dropout_prob = 0.0
        self.dtype = "float16"
        self.rtol = 5e-3
        self.atol = 5e-3


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedCrossAttentionOpWithMask(TestFusedSelfAttentionOp):
    def _set_config(self):
        self.has_attn_mask = True
        self.attn_mode = "cross_attn"  # self_attn or cross_attn
        self.dropout_prob = 0.0
        self.dtype = "float16"
        self.rtol = 5e-3
        self.atol = 5e-3


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedSelfAttentionOpWithMaskCase2(TestFusedSelfAttentionOp):
    def _set_shape(self):
        self.batch_size = 8
        self.q_seqlen = 512
        self.kv_seqlen = 512
        self.num_heads = 16
        self.head_size = 64

    def _set_config(self):
        self.has_attn_mask = True
        self.attn_mode = "self_attn"  # self_attn or cross_attn
        self.dropout_prob = 0.0
        self.dtype = "float16"
        self.rtol = 5e-3
        self.atol = 5e-3


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedCrossAttentionOpWithMaskCase2(TestFusedSelfAttentionOp):
    def _set_shape(self):
        self.batch_size = 8
        self.q_seqlen = 192
        self.kv_seqlen = 192
        self.num_heads = 16
        self.head_size = 64

    def _set_config(self):
        self.has_attn_mask = True
        self.attn_mode = "cross_attn"  # self_attn or cross_attn
        self.dropout_prob = 0.0
        self.dtype = "float16"
        self.rtol = 5e-3
        self.atol = 5e-3


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedCrossAttentionOpWithMaskCase3(TestFusedSelfAttentionOp):
    def _set_shape(self):
        self.batch_size = 8
        self.q_seqlen = 448
        self.kv_seqlen = 320
        self.num_heads = 16
        self.head_size = 64

    def _set_config(self):
        self.has_attn_mask = True
        self.attn_mode = "cross_attn"  # self_attn or cross_attn
        self.dropout_prob = 0.0
        self.dtype = "float16"
        self.rtol = 5e-3
        self.atol = 5e-3


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedCrossAttentionOpWithMaskCase4(TestFusedCrossAttentionOp):
    def _set_shape(self):
        self.batch_size = 8
        self.q_seqlen = 256
        self.kv_seqlen = 192
        self.num_heads = 16
        self.head_size = 64

    def _set_config(self):
        self.has_attn_mask = True
        self.attn_mode = "cross_attn"  # self_attn or cross_attn
        self.dropout_prob = 0.0
        self.dtype = "float16"
        self.rtol = 5e-3
        self.atol = 5e-3


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedSelfAttentionOpDropout(TestFusedSelfAttentionOp):
    def sef_config(self):
        self.has_attn_mask = True
        self.attn_mode = "self_attn"  # self_attn or cross_attn
        self.dropout_prob = 0.1
        self.dtype = "float16"
        self.rtol = 5e-3
        self.atol = 5e-3

    def _check_dropout(self):
        _, _, _, _, cudnn_softmax_out = self._get_cudnn_fmha_out()
        cudnn_softmax_out = cudnn_softmax_out.numpy()

        # cuDNN's random generator is not the same as Paddle's, so we can't
        # compare the dropout output of cuDNN and Paddle directly.
        # cudnn set the dropped position to negative value, so we can check
        # the dropout rate by counting the number of negative values.
        cudnn_dropout_position = cudnn_softmax_out < 0
        # The number of masked positions is the product of the actual sequence
        # length of Q and K. The number of heads is considered here because
        # the dropout is applied to all heads.
        # There is a little difference between the dropout rate of Paddle and
        # cuDNN. Paddle's MultiheadAttention uses dropout on the whole output
        # tensor, while cuDNN uses dropout on valid (not masked) positions.
        cudnn_dropout_masked_size = (
            np.dot(self.q_actual_seqlen, self.kv_actual_seqlen) * self.num_heads
        )
        cudnn_dropout_rate = (
            np.sum(cudnn_dropout_position) / cudnn_dropout_masked_size
        )
        np.testing.assert_allclose(
            cudnn_dropout_rate,
            self.dropout_prob,
            atol=self.atol,
            rtol=self.rtol,
            err_msg="Checking < dropout rate > failed",
        )

    def test_output(self):
        self._check_dropout()


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedCrossAttentionOpDropout(TestFusedSelfAttentionOpDropout):
    def sef_config(self):
        self.has_attn_mask = True
        self.attn_mode = "cross_attn"  # self_attn or cross_attn
        self.dropout_prob = 0.2
        self.dtype = "float16"
        self.rtol = 5e-3
        self.atol = 5e-3


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedSelfAttentionOpBF16(TestFusedSelfAttentionOp):
    def sef_config(self):
        self.has_attn_mask = False
        self.attn_mode = "self_attn"
        self.dropout_prob = 0.0
        self.dtype = "bfloat16"
        self.rtol = 5e-3
        self.atol = 5e-3


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedSelfAttentionOpBF16Case2(TestFusedSelfAttentionOp):
    def _set_shape(self):
        self.batch_size = 8
        self.q_seqlen = 512
        self.kv_seqlen = 512
        self.num_heads = 16
        self.head_size = 64

    def sef_config(self):
        self.has_attn_mask = True
        self.attn_mode = "self_attn"
        self.dropout_prob = 0.0
        self.dtype = "bfloat16"
        self.rtol = 5e-3
        self.atol = 5e-3


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedCrossAttentionOpBF16(TestFusedCrossAttentionOp):
    def _set_shape(self):
        self.batch_size = 8
        self.q_seqlen = 192
        self.kv_seqlen = 320
        self.num_heads = 16
        self.head_size = 64

    def sef_config(self):
        self.has_attn_mask = True
        self.attn_mode = "cross_attn"
        self.dropout_prob = 0.0
        self.dtype = "bfloat16"
        self.rtol = 5e-3
        self.atol = 5e-3


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedCrossAttentionOpBF16Case2(TestFusedCrossAttentionOp):
    def _set_shape(self):
        self.batch_size = 8
        self.q_seqlen = 448
        self.kv_seqlen = 320
        self.num_heads = 16
        self.head_size = 64

    def sef_config(self):
        self.has_attn_mask = True
        self.attn_mode = "cross_attn"
        self.dropout_prob = 0.0
        self.dtype = "bfloat16"
        self.rtol = 5e-3
        self.atol = 5e-3


if __name__ == "__main__":
    unittest.main()
