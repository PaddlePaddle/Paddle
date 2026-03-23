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
from unittest import mock

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle.nn.functional.conv import (
    _MEMORY_FORMAT_CHANNELS_LAST,
    _MEMORY_FORMAT_CHANNELS_LAST_3D,
    _MEMORY_FORMAT_CONTIGUOUS,
    _cudnn_conv_suggest_memory_format,
)


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
class TestCudnnConvCoverage(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)

    def test_cudnn_conv_suggest_memory_format_coverage(self):
        x_fp32 = paddle.randn([1, 3, 16, 16], dtype='float32')
        w_fp32_4d = paddle.randn([3, 3, 3, 3], dtype='float32')
        w_fp32_5d = paddle.randn([3, 3, 3, 3, 3], dtype='float32')
        x_fp64 = paddle.cast(x_fp32, 'float64')
        w_fp64_4d = paddle.cast(w_fp32_4d, 'float64')

        self.assertEqual(
            _cudnn_conv_suggest_memory_format(x_fp64, w_fp32_4d),
            _MEMORY_FORMAT_CONTIGUOUS,
        )
        self.assertEqual(
            _cudnn_conv_suggest_memory_format(x_fp32, w_fp64_4d),
            _MEMORY_FORMAT_CONTIGUOUS,
        )

        with mock.patch(
            'paddle.nn.functional.conv.get_cudnn_version', return_value=8500
        ):
            self.assertEqual(
                _cudnn_conv_suggest_memory_format(
                    x_fp32, w_fp32_4d, data_format="NHWC"
                ),
                _MEMORY_FORMAT_CHANNELS_LAST,
            )

            self.assertEqual(
                _cudnn_conv_suggest_memory_format(
                    x_fp32, w_fp32_5d, data_format="NHWC"
                ),
                _MEMORY_FORMAT_CHANNELS_LAST_3D,
            )

            self.assertEqual(
                _cudnn_conv_suggest_memory_format(
                    x_fp32, w_fp32_4d, data_format="NCHW"
                ),
                _MEMORY_FORMAT_CONTIGUOUS,
            )

        with mock.patch(
            'paddle.nn.functional.conv.get_cudnn_version', return_value=7000
        ):
            self.assertEqual(
                _cudnn_conv_suggest_memory_format(
                    x_fp32, w_fp32_4d, data_format="NHWC"
                ),
                _MEMORY_FORMAT_CONTIGUOUS,
            )

    def test_is_cudnn_supported_coverage(self):
        from paddle.nn.functional.conv import _is_cudnn_supported

        x_gpu_fp16 = paddle.randn([1, 3, 8, 8, 8], dtype='float16').to(
            self.place
        )
        x_cpu = paddle.randn([1, 3, 8, 8, 8], dtype='float16').cpu()

        self.assertFalse(_is_cudnn_supported(x_gpu_fp16, None, "NCDHW", False))

        self.assertFalse(_is_cudnn_supported(x_cpu, None, "NCDHW", True))

        w_trivial = paddle.randn([3, 3, 1, 1, 1], dtype='float16')
        w_non_trivial = paddle.randn([3, 3, 3, 3, 3], dtype='float16')

        with mock.patch(
            'paddle.nn.functional.conv.get_cudnn_version', return_value=91000
        ):
            self.assertFalse(
                _is_cudnn_supported(x_gpu_fp16, w_non_trivial, "NCDHW", True)
            )

            self.assertTrue(
                _is_cudnn_supported(x_gpu_fp16, w_trivial, "NCDHW", True)
            )

            x_gpu_fp32 = paddle.randn([1, 3, 8, 8, 8], dtype='float32').to(
                self.place
            )
            self.assertTrue(
                _is_cudnn_supported(x_gpu_fp32, w_non_trivial, "NCDHW", True)
            )

        with mock.patch(
            'paddle.nn.functional.conv.get_cudnn_version', return_value=85000
        ):
            self.assertTrue(
                _is_cudnn_supported(x_gpu_fp16, w_non_trivial, "NCDHW", True)
            )


if __name__ == '__main__':
    unittest.main()
