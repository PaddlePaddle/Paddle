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

import paddle
from paddle.incubate.nn.functional import (
    cudnn_flash_attention,
    fused_dot_product_attention,
)

np.random.seed(2024)


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
class TestFusedDotProductAttentionStatic(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.b = 2
        self.s_q = 128
        self.s_kv = 128
        self.h = 2
        self.d = 64
        self.q_shape = (self.b, self.s_q, self.h, self.d)
        self.kv_shape = (self.b, self.s_kv, self.h, self.d)
        self.mask_shape = (self.b, 1, self.s_q, self.s_kv)
        self.dtype = 'float16'

    def test_static_op(self):
        paddle.disable_static()
        q_data = np.random.random(self.q_shape)
        k_data = np.random.random(self.kv_shape)
        v_data = np.random.random(self.kv_shape)
        mask_data = np.random.random(self.mask_shape)
        q = paddle.to_tensor(
            q_data, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        k = paddle.to_tensor(
            k_data, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        v = paddle.to_tensor(
            v_data, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        mask = paddle.to_tensor(
            mask_data, place=self.place, dtype=self.dtype, stop_gradient=True
        )
        out0 = fused_dot_product_attention(q, k, v, mask)
        out1 = cudnn_flash_attention(
            q,
            k,
            v,
            mask,
            None,
            None,
            1.0,
            0.0,
            True,
            None,
            "post_scale_bias",
        )
        paddle.enable_static()
        paddle.seed(312)

        # call fused_dot_product_attention in static mode
        with paddle.static.program_guard(paddle.static.Program()):
            q = paddle.static.data(
                name="q", shape=self.q_shape, dtype=self.dtype
            )
            k = paddle.static.data(
                name="k", shape=self.kv_shape, dtype=self.dtype
            )
            v = paddle.static.data(
                name="v", shape=self.kv_shape, dtype=self.dtype
            )
            mask = paddle.static.data(
                name="mask", shape=self.mask_shape, dtype=self.dtype
            )

            outs = fused_dot_product_attention(q, k, v, mask)

            exe = paddle.static.Executor(self.place)
            out_s = exe.run(
                feed={
                    "q": q_data.astype('float16'),
                    "k": k_data.astype('float16'),
                    "v": v_data.astype('float16'),
                    "mask": mask_data.astype('float16'),
                },
                fetch_list=[outs],
            )
            np.testing.assert_allclose(out_s[0], out0)

        # call cudnn_flash_attention in static mode
        with paddle.static.program_guard(paddle.static.Program()):
            q = paddle.static.data(
                name="q", shape=self.q_shape, dtype=self.dtype
            )
            k = paddle.static.data(
                name="k", shape=self.kv_shape, dtype=self.dtype
            )
            v = paddle.static.data(
                name="v", shape=self.kv_shape, dtype=self.dtype
            )
            mask = paddle.static.data(
                name="mask", shape=self.mask_shape, dtype=self.dtype
            )

            outs = cudnn_flash_attention(
                q,
                k,
                v,
                mask,
                None,
                None,
                1.0,
                0.0,
                True,
                None,
                "post_scale_bias",
            )

            exe = paddle.static.Executor(self.place)
            out_s = exe.run(
                feed={
                    "q": q_data.astype('float16'),
                    "k": k_data.astype('float16'),
                    "v": v_data.astype('float16'),
                    "mask": mask_data.astype('float16'),
                },
                fetch_list=[outs],
            )
            np.testing.assert_allclose(out_s[0], out1)


if __name__ == "__main__":
    unittest.main()
