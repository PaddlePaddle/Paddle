# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F


@unittest.skipIf(
    not paddle.is_compiled_with_cuda(),
    "Skipping tests: CUDA is not available.",
)
class TestSlowConv2dDilated(unittest.TestCase):
    def setUp(self):
        # Save old flag states
        self.old_flag_acc = paddle.get_flags(
            ['FLAGS_use_accuracy_compatible_kernel']
        )
        self.old_flag_disable = paddle.get_flags(['FLAGS_conv2d_disable_cudnn'])

        self.place = paddle.CUDAPlace(0)

        np.random.seed(2026)
        paddle.seed(2026)

    def tearDown(self):
        # Restore flags
        paddle.set_flags(self.old_flag_acc)
        paddle.set_flags(self.old_flag_disable)

    def _init_data(self, dtype, layout, with_bias):
        groups = 1
        N = 2
        C_in = 4
        C_out = 4
        H, W = 16, 16
        K = 3

        if layout == "NCHW":
            input_shape = [N, C_in, H, W]
        else:  # NHWC
            input_shape = [N, H, W, C_in]

        weight_shape = [C_out, C_in // groups, K, K]

        np_x = np.random.randn(*input_shape).astype('float32')
        np_w = np.random.randn(*weight_shape).astype('float32')

        np_b = None
        if with_bias:
            np_b = np.random.randn(C_out).astype('float32')

        return np_x, np_w, np_b, groups

    def _run_op(
        self, np_x, np_w, np_b, dtype, layout, groups, disable_cudnn_flag
    ):
        paddle.set_flags({'FLAGS_conv2d_disable_cudnn': disable_cudnn_flag})
        paddle.set_flags({'FLAGS_use_accuracy_compatible_kernel': 1})

        x = paddle.to_tensor(np_x, place=self.place, dtype=dtype)
        x.stop_gradient = False

        w = paddle.to_tensor(np_w, place=self.place, dtype=dtype)
        w.stop_gradient = False

        b = None
        if np_b is not None:
            b = paddle.to_tensor(np_b, place=self.place, dtype=dtype)
            b.stop_gradient = False

        out = F.conv2d(
            x,
            w,
            b,
            stride=1,
            padding=1,
            dilation=2,
            groups=groups,
            data_format=layout,
        )

        loss = out.sum()
        loss.backward()

        return out.numpy()

    def _check_implementation(self, dtype, layout="NCHW", with_bias=True):
        np_x, np_w, np_b, groups = self._init_data(dtype, layout, with_bias)
        self._run_op(
            np_x, np_w, np_b, dtype, layout, groups, disable_cudnn_flag=1
        )

    # =================================================================
    # Test Cases for Registered Types
    # =================================================================
    def test_fp16(self):
        self._check_implementation('float16', layout="NCHW", with_bias=True)
        self._check_implementation('float16', layout="NCHW", with_bias=False)
        self._check_implementation('float16', layout="NHWC", with_bias=True)
        self._check_implementation('float16', layout="NHWC", with_bias=False)

    def test_fp32(self):
        self._check_implementation('float32', layout="NCHW", with_bias=True)
        self._check_implementation('float32', layout="NCHW", with_bias=False)
        self._check_implementation('float32', layout="NHWC", with_bias=True)
        self._check_implementation('float32', layout="NHWC", with_bias=False)


@unittest.skipIf(
    not paddle.is_compiled_with_cuda(),
    "Skipping tests: CUDA is not available.",
)
class TestSlowConv3dDilated(unittest.TestCase):
    def setUp(self):
        # Save old flag states
        self.old_flag_acc = paddle.get_flags(
            ['FLAGS_use_accuracy_compatible_kernel']
        )
        self.old_flag_disable = paddle.get_flags(['FLAGS_conv3d_disable_cudnn'])

        self.place = paddle.CUDAPlace(0)

        np.random.seed(2026)
        paddle.seed(2026)

    def tearDown(self):
        # Restore flags
        paddle.set_flags(self.old_flag_acc)
        paddle.set_flags(self.old_flag_disable)

    def _init_data(self, dtype, layout, with_bias):
        groups = 1
        N = 2
        C_in = 4
        C_out = 4
        D, H, W = 8, 8, 8
        K = 3

        if layout == "NCDHW":
            input_shape = [N, C_in, D, H, W]
        else:  # NDHWC
            input_shape = [N, D, H, W, C_in]

        weight_shape = [C_out, C_in // groups, K, K, K]

        np_x = np.random.randn(*input_shape).astype('float32')
        np_w = np.random.randn(*weight_shape).astype('float32')

        np_b = None
        if with_bias:
            np_b = np.random.randn(C_out).astype('float32')

        return np_x, np_w, np_b, groups

    def _run_op(
        self, np_x, np_w, np_b, dtype, layout, groups, disable_cudnn_flag
    ):
        paddle.set_flags({'FLAGS_conv3d_disable_cudnn': disable_cudnn_flag})
        paddle.set_flags({'FLAGS_use_accuracy_compatible_kernel': 1})

        x = paddle.to_tensor(np_x, place=self.place, dtype=dtype)
        x.stop_gradient = False

        w = paddle.to_tensor(np_w, place=self.place, dtype=dtype)
        w.stop_gradient = False

        b = None
        if np_b is not None:
            b = paddle.to_tensor(np_b, place=self.place, dtype=dtype)
            b.stop_gradient = False

        out = F.conv3d(
            x,
            w,
            b,
            stride=1,
            padding=1,
            dilation=2,
            groups=groups,
            data_format=layout,
        )

        loss = out.sum()
        loss.backward()

        return out.numpy()

    def _check_implementation(self, dtype, layout="NCDHW", with_bias=True):
        np_x, np_w, np_b, groups = self._init_data(dtype, layout, with_bias)
        self._run_op(
            np_x, np_w, np_b, dtype, layout, groups, disable_cudnn_flag=1
        )

    # =================================================================
    # Test Cases for Registered Types
    # =================================================================
    def test_fp16(self):
        self._check_implementation('float16', layout="NCDHW", with_bias=True)
        self._check_implementation('float16', layout="NCDHW", with_bias=False)
        self._check_implementation('float16', layout="NDHWC", with_bias=True)
        self._check_implementation('float16', layout="NDHWC", with_bias=False)

    def test_fp32(self):
        self._check_implementation('float32', layout="NCDHW", with_bias=True)
        self._check_implementation('float32', layout="NCDHW", with_bias=False)
        self._check_implementation('float32', layout="NDHWC", with_bias=True)
        self._check_implementation('float32', layout="NDHWC", with_bias=False)


@unittest.skipIf(
    not paddle.is_compiled_with_cuda(), "CUDA is required for coverage test"
)
class TestConvCudnnCoverage(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)

        self.old_acc_flag = paddle.get_flags(
            ['FLAGS_use_accuracy_compatible_kernel']
        )
        paddle.set_flags({'FLAGS_use_accuracy_compatible_kernel': 1})

        self.old_cudnn_flag = paddle.get_flags(
            ['FLAGS_conv2d_disable_cudnn', 'FLAGS_conv3d_disable_cudnn']
        )
        paddle.set_flags({'FLAGS_conv2d_disable_cudnn': 0})
        paddle.set_flags({'FLAGS_conv3d_disable_cudnn': 0})

    def tearDown(self):
        paddle.set_flags(self.old_acc_flag)
        paddle.set_flags(self.old_cudnn_flag)

    def test_conv2d_fp64_coverage(self):
        x = paddle.randn([1, 3, 16, 16], dtype='float64')
        w = paddle.randn([3, 3, 3, 3], dtype='float64')

        out = F.conv2d(x, w, use_cudnn=True)

    def test_conv2d_nhwc_coverage(self):
        x = paddle.randn([1, 16, 16, 3], dtype='float32')
        w = paddle.randn([3, 3, 3, 3], dtype='float32')

        out = F.conv2d(x, w, data_format="NHWC", use_cudnn=True)

    def test_conv3d_ndhwc_coverage(self):
        x = paddle.randn([1, 8, 8, 8, 3], dtype='float32')
        w = paddle.randn([3, 3, 3, 3, 3], dtype='float32')

        out = F.conv3d(x, w, data_format="NDHWC", use_cudnn=True)
        self.assertEqual(out.shape, [1, 6, 6, 6, 3])

    def test_conv3d_fp16_bug_logic_coverage(self):
        x = paddle.randn([1, 3, 8, 8, 8], dtype='float16')
        w = paddle.randn([3, 3, 3, 3, 3], dtype='float16')
        out = F.conv3d(x, w, data_format="NCDHW", use_cudnn=True)

        self.assertEqual(out.dtype, paddle.float16)

    def test_cudnn_disabled_coverage(self):
        x = paddle.randn([1, 3, 16, 16], dtype='float32')
        w = paddle.randn([3, 3, 3, 3], dtype='float32')

        out = F.conv2d(x, w, use_cudnn=False)
        self.assertTrue(out is not None)


if __name__ == '__main__':
    unittest.main()
