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

import os
import unittest

os.environ["FLAGS_cudnn_deterministic"] = "1"

import numpy as np
from op_test import (
    OpTest,
    convert_float_to_uint16,
    convert_uint16_to_float,
)

import paddle
import paddle.nn.functional as F
from paddle.incubate.nn.functional import (
    cudnn_flash_attention,
    fused_dot_product_attention,
)

np.random.seed(2023)


def skip_unit_test():
    return (
        not paddle.is_compiled_with_cuda()
        or paddle.device.cuda.get_device_capability()[0] < 8
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
        self.is_causal_masking = False
        self.dropout_prob = 0.0
        self.dtype = "float16"
        self.rtol = 5e-4
        self.atol = 5e-4
        self.check_deterministic = False

    def setUp(self):
        self._set_shape()
        self._set_config()
        # has_attn_mask and is_causal_masking can't be True at the same time
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

        if self.is_causal_masking:
            self.attn_mask = np.ones(
                (1, 1, self.q_seqlen, self.kv_seqlen),
                dtype=np.float16,
            )
            self.attn_mask = np.triu(self.attn_mask, k=1)
            self.attn_mask = self.attn_mask * -1e4
        else:
            # create a mask with 50% of the elements set to -1e4, the rest to 0
            self.attn_mask = np.random.choice(
                [0, -1e4],
                size=(1, 1, self.q_seqlen, self.kv_seqlen),
                p=[0.5, 0.5],
            )
        self.attn_mask = paddle.to_tensor(
            self.attn_mask, stop_gradient=True, dtype=self.dtype
        )

        dout_shape = self.q_shape
        self.dout = _random(dout_shape)

    def _get_reference_out(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        q_tensor = paddle.to_tensor(self.q, stop_gradient=False)
        # print(q_tensor)
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

        if self.attn_mask is not None:
            attn_mask_out = qk_out + self.attn_mask
        else:
            attn_mask_out = qk_out
        softmax_out = F.softmax(attn_mask_out)

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

        return (
            mha_out,
            q_tensor.grad,
            k_tensor.grad,
            v_tensor.grad,
        )

    def _get_fused_attn_out(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        q_tensor = paddle.to_tensor(self.q, stop_gradient=False)
        k_tensor = paddle.to_tensor(self.k, stop_gradient=False)
        v_tensor = paddle.to_tensor(self.v, stop_gradient=False)

        attn_mask = self.attn_mask
        if self.is_causal_masking:
            attn_mask = None
        fmha_out = fused_dot_product_attention(
            q_tensor,
            k_tensor,
            v_tensor,
            attn_mask,
            self.dropout_prob,
            self.is_causal_masking,
            training=True,
        )

        paddle.autograd.backward(
            [fmha_out], [paddle.to_tensor(self.dout)], retain_graph=True
        )

        return (
            fmha_out,
            q_tensor.grad,
            k_tensor.grad,
            v_tensor.grad,
        )

    def _get_cudnn_flash_attn_out(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        q_tensor = paddle.to_tensor(self.q, stop_gradient=False)
        k_tensor = paddle.to_tensor(self.k, stop_gradient=False)
        v_tensor = paddle.to_tensor(self.v, stop_gradient=False)

        bias = self.attn_mask
        if self.is_causal_masking:
            bias = None
        if self.is_causal_masking:
            attn_mask = None
        fmha_out = cudnn_flash_attention(
            q_tensor,
            k_tensor,
            v_tensor,
            bias=bias,
            cu_seqlen_q=None,
            cu_seqlen_k=None,
            scaling_factor=self.scaling_factor,
            dropout_prob=self.dropout_prob,
            training=True,
            mask_type="causal" if self.is_causal_masking else None,
            bias_type="post_scale_bias" if bias is not None else None,
        )

        paddle.autograd.backward(
            [fmha_out], [paddle.to_tensor(self.dout)], retain_graph=True
        )

        return (
            fmha_out,
            q_tensor.grad,
            k_tensor.grad,
            v_tensor.grad,
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
        outputs_fused_1 = self._get_fused_attn_out()
        outputs_fused_2 = self._get_cudnn_flash_attn_out()

        for i in range(len(output_names)):
            ref_res = outputs_ref[i]
            fused_res_1 = outputs_fused_1[i]
            fused_res_2 = outputs_fused_2[i]
            np.testing.assert_allclose(
                _convert(ref_res.numpy()),
                _convert(fused_res_1.numpy()),
                atol=self.atol,
                rtol=self.rtol,
                err_msg=f"Checking < {output_names[i]} > failed",
            )
            np.testing.assert_allclose(
                _convert(ref_res.numpy()),
                _convert(fused_res_2.numpy()),
                atol=self.atol,
                rtol=self.rtol,
                err_msg=f"Checking < {output_names[i]} > failed",
            )
            if self.check_deterministic:
                np.testing.assert_allclose(
                    _convert(fused_res_1.numpy()),
                    _convert(fused_res_2.numpy()),
                    atol=1e-12,
                    rtol=1e-12,
                    err_msg=f"Checking < {output_names[i]} > failed",
                )

    def test_output(self):
        self._compare_output()


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedAttentionOpFP16Case2(TestFusedAttentionOpFP16):
    def _set_shape(self):
        self.batch_size = 2
        self.q_seqlen = 1024
        self.kv_seqlen = 1024
        self.num_heads = 2
        self.head_size = 64


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedAttentionOpFP16Case3(TestFusedAttentionOpFP16):
    def _set_shape(self):
        self.batch_size = 1
        self.q_seqlen = 2048
        self.kv_seqlen = 2048
        self.num_heads = 2
        self.head_size = 128


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedAttentionOpFP16WithCausalMask(TestFusedAttentionOpFP16):
    def _set_shape(self):
        self.batch_size = 2
        self.q_seqlen = 1024
        self.kv_seqlen = 1024
        self.num_heads = 2
        self.head_size = 128

    def _set_config(self):
        self.is_causal_masking = True
        self.dropout_prob = 0.0
        self.dtype = "float16"
        self.rtol = 1e-3
        self.atol = 1e-3
        self.check_deterministic = False


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedAttentionOpBF16(TestFusedAttentionOpFP16):
    def _set_shape(self):
        self.batch_size = 1
        self.q_seqlen = 2048
        self.kv_seqlen = 2048
        self.num_heads = 2
        self.head_size = 128

    def _set_config(self):
        self.is_causal_masking = False
        self.dropout_prob = 0.0
        self.dtype = "bfloat16"
        self.rtol = 5e-4
        self.atol = 5e-4
        self.check_deterministic = True


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedAttentionOpBF16WithCausalMask(TestFusedAttentionOpFP16):
    def _set_shape(self):
        self.batch_size = 1
        self.q_seqlen = 2048
        self.kv_seqlen = 2048
        self.num_heads = 2
        self.head_size = 128

    def _set_config(self):
        self.is_causal_masking = True
        self.dropout_prob = 0.0
        self.dtype = "bfloat16"
        self.rtol = 5e-4
        self.atol = 5e-4
        self.check_deterministic = True


if __name__ == "__main__":
    unittest.main()
