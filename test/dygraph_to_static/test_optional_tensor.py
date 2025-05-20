# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_pir_only,
)

import paddle


def call_fused_rms_norm(x, y):
    return paddle.incubate.nn.functional.fused_rms_norm(
        x,
        y,
        None,
        1e-6,
        begin_norm_axis=1,
    )


class TestOptionalTensorOutput(Dy2StTestBase):
    @test_pir_only
    def test_fused_rms_norm(self):
        if not paddle.is_compiled_with_cuda():
            return
        fn = call_fused_rms_norm
        static_fn = paddle.jit.to_static(fn)

        x = paddle.randn([1410, 5120], dtype='float32')
        y = paddle.randn([5120], dtype='float32')
        x.stop_gradient = False

        out_1_dy, out_2_dy = fn(x, y)
        out_1_st, out_2_st = static_fn(x, y)
        np.testing.assert_equal(out_1_dy.numpy(), out_1_st.numpy())
        self.assertFalse(out_2_dy._is_initialized())
        self.assertFalse(out_2_st._is_initialized())


if __name__ == '__main__':
    unittest.main()
