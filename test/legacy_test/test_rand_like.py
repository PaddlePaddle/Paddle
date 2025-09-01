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

import paddle
from paddle import base, core


class TestRandLikeAPI(unittest.TestCase):
    """
    Test python API for rand_like function.
    """

    def setUp(self):
        self.x_float16 = np.zeros((10, 12)).astype("float16")
        self.x_float32 = np.zeros((10, 12)).astype("float32")
        self.x_float64 = np.zeros((10, 12)).astype("float64")
        self.dtype = ["float16", "float32", "float64"]

    def test_static_api_basic(self):
        """Test basic static API functionality"""
        paddle.enable_static()
        try:
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x_float32 = paddle.static.data(
                    name="x_float32", shape=[10, 12], dtype="float32"
                )

                # Test with default parameters
                out1 = paddle.rand_like(x_float32)

                # Test with specified name
                out2 = paddle.rand_like(x_float32, name="test_rand_like")

                place = base.CPUPlace()
                if core.is_compiled_with_cuda():
                    place = base.CUDAPlace(0)

                exe = paddle.static.Executor(place)
                outs = exe.run(
                    feed={'x_float32': self.x_float32}, fetch_list=[out1, out2]
                )

                for out in outs:
                    self.assertEqual(out.shape, (10, 12))
                    self.assertEqual(out.dtype, np.float32)
                    self.assertTrue(((out >= 0.0) & (out <= 1.0)).all())
        finally:
            paddle.disable_static()

    def test_static_api_with_dtype(self):
        """Test static API with different dtype specifications"""
        paddle.enable_static()
        try:
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x_float32 = paddle.static.data(
                    name="x_float32", shape=[10, 12], dtype="float32"
                )

                place = base.CPUPlace()
                if core.is_compiled_with_cuda():
                    place = base.CUDAPlace(0)

                exe = paddle.static.Executor(place)

                # Test with different dtypes
                for dtype in self.dtype:
                    if dtype == "float16" and not core.is_compiled_with_cuda():
                        continue

                    out = paddle.rand_like(x_float32, dtype=dtype)
                    result = exe.run(
                        feed={'x_float32': self.x_float32}, fetch_list=[out]
                    )[0]

                    self.assertEqual(result.shape, (10, 12))
                    self.assertEqual(result.dtype, np.dtype(dtype))
                    self.assertTrue(((result >= 0.0) & (result <= 1.0)).all())
        finally:
            paddle.disable_static()

    def test_static_api_with_device(self):
        """Test static API with device specification"""
        paddle.enable_static()
        try:
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x_float32 = paddle.static.data(
                    name="x_float32", shape=[10, 12], dtype="float32"
                )

                # Test with CPU device
                out1 = paddle.rand_like(x_float32, device=base.CPUPlace())

                place = base.CPUPlace()
                exe = paddle.static.Executor(place)
                result = exe.run(
                    feed={'x_float32': self.x_float32}, fetch_list=[out1]
                )[0]

                self.assertEqual(result.shape, (10, 12))
                self.assertTrue(((result >= 0.0) & (result <= 1.0)).all())

                # Test with CUDA device if available
                if core.is_compiled_with_cuda():
                    out2 = paddle.rand_like(x_float32, device=base.CUDAPlace(0))
                    place_cuda = base.CUDAPlace(0)
                    exe_cuda = paddle.static.Executor(place_cuda)
                    result_cuda = exe_cuda.run(
                        feed={'x_float32': self.x_float32}, fetch_list=[out2]
                    )[0]

                    self.assertEqual(result_cuda.shape, (10, 12))
                    self.assertTrue(
                        ((result_cuda >= 0.0) & (result_cuda <= 1.0)).all()
                    )
        finally:
            paddle.disable_static()

    def test_dygraph_api_basic(self):
        """Test basic dygraph API functionality"""
        for x_np in [self.x_float32, self.x_float64]:
            x = paddle.to_tensor(x_np)

            # Test with default parameters
            out1 = paddle.rand_like(x)
            self.assertEqual(out1.shape, x.shape)
            self.assertEqual(out1.dtype, x.dtype)
            self.assertTrue(
                ((out1.numpy() >= 0.0) & (out1.numpy() <= 1.0)).all()
            )

            # Test with name parameter
            out2 = paddle.rand_like(x, name="test_rand_like")
            self.assertEqual(out2.shape, x.shape)
            self.assertEqual(out2.dtype, x.dtype)
            self.assertTrue(
                ((out2.numpy() >= 0.0) & (out2.numpy() <= 1.0)).all()
            )

        # Test with float16 if CUDA is available
        if core.is_compiled_with_cuda():
            x = paddle.to_tensor(self.x_float16)
            out = paddle.rand_like(x)
            self.assertEqual(out.shape, x.shape)
            self.assertEqual(out.dtype, x.dtype)
            self.assertTrue(((out.numpy() >= 0.0) & (out.numpy() <= 1.0)).all())

    def test_dygraph_api_with_dtype(self):
        """Test dygraph API with different dtype specifications"""
        x = paddle.to_tensor(self.x_float32)

        for dtype in self.dtype:
            if dtype == "float16" and not core.is_compiled_with_cuda():
                continue

            out = paddle.rand_like(x, dtype=dtype)
            self.assertEqual(out.shape, x.shape)
            self.assertEqual(out.dtype, getattr(paddle, dtype))
            self.assertTrue(((out.numpy() >= 0.0) & (out.numpy() <= 1.0)).all())

    def test_dygraph_api_with_requires_grad(self):
        """Test dygraph API with requires_grad parameter"""
        x = paddle.to_tensor(self.x_float32)

        # Test requires_grad=True
        out1 = paddle.rand_like(x, requires_grad=True)
        self.assertEqual(out1.shape, x.shape)
        self.assertFalse(out1.stop_gradient)
        self.assertTrue(((out1.numpy() >= 0.0) & (out1.numpy() <= 1.0)).all())

        # Test requires_grad=False
        out2 = paddle.rand_like(x, requires_grad=False)
        self.assertEqual(out2.shape, x.shape)
        self.assertTrue(out2.stop_gradient)
        self.assertTrue(((out2.numpy() >= 0.0) & (out2.numpy() <= 1.0)).all())

    def test_dygraph_api_with_device(self):
        """Test dygraph API with device specification"""
        x = paddle.to_tensor(self.x_float32)

        # Test with CPU device
        out1 = paddle.rand_like(x, device=paddle.CPUPlace())
        self.assertEqual(out1.shape, x.shape)
        self.assertEqual(out1.dtype, x.dtype)
        self.assertTrue(out1.place.is_cpu_place())
        self.assertTrue(((out1.numpy() >= 0.0) & (out1.numpy() <= 1.0)).all())

        # Test with CUDA device if available
        if core.is_compiled_with_cuda():
            out2 = paddle.rand_like(x, device=paddle.CUDAPlace(0))
            self.assertEqual(out2.shape, x.shape)
            self.assertEqual(out2.dtype, x.dtype)
            self.assertTrue(out2.place.is_gpu_place())
            self.assertTrue(
                ((out2.numpy() >= 0.0) & (out2.numpy() <= 1.0)).all()
            )

    def test_dygraph_api_combined_params(self):
        """Test dygraph API with combined parameters"""
        x = paddle.to_tensor(self.x_float32)

        # Test dtype + requires_grad
        out1 = paddle.rand_like(x, dtype="float64", requires_grad=True)
        self.assertEqual(out1.shape, x.shape)
        self.assertEqual(out1.dtype, paddle.float64)
        self.assertFalse(out1.stop_gradient)
        self.assertTrue(((out1.numpy() >= 0.0) & (out1.numpy() <= 1.0)).all())

        # Test all parameters together
        out2 = paddle.rand_like(
            x, name="combined_test", dtype="float64", requires_grad=False
        )
        self.assertEqual(out2.shape, x.shape)
        self.assertEqual(out2.dtype, paddle.float64)
        self.assertTrue(out2.stop_gradient)
        self.assertTrue(((out2.numpy() >= 0.0) & (out2.numpy() <= 1.0)).all())

    def test_different_shapes(self):
        """Test with different input shapes"""
        shapes = [
            [
                1,
            ],
            [5, 3],
            [2, 4, 6],
            [1, 2, 3, 4],
        ]

        for shape in shapes:
            x = paddle.zeros(shape, dtype='float32')
            out = paddle.rand_like(x)
            self.assertEqual(out.shape, shape)
            self.assertTrue(((out.numpy() >= 0.0) & (out.numpy() <= 1.0)).all())

    def test_default_dtype_behavior(self):
        """Test default dtype behavior"""
        # Test that output dtype matches input dtype when dtype=None
        dtypes_to_test = ['float32', 'float64']
        if core.is_compiled_with_cuda():
            dtypes_to_test.append('float16')

        for dtype_str in dtypes_to_test:
            x = paddle.zeros((3, 4), dtype=dtype_str)
            out = paddle.rand_like(x)  # dtype=None (default)
            self.assertEqual(out.dtype, x.dtype)
            self.assertTrue(((out.numpy() >= 0.0) & (out.numpy() <= 1.0)).all())


class TestRandLikeOpForDygraph(unittest.TestCase):
    """
    Test rand_like operation in dygraph mode with different scenarios.
    """

    def run_net(self, use_cuda=False):
        place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
        with base.dygraph.guard(place):
            # Test basic functionality
            x1 = paddle.zeros([3, 4], dtype='float32')
            out1 = paddle.rand_like(x1)

            # Test with different dtype
            x2 = paddle.zeros([3, 4], dtype='float32')
            out2 = paddle.rand_like(x2, dtype='float64')

            # Test with requires_grad
            x3 = paddle.zeros([2, 5], dtype='float32')
            out3 = paddle.rand_like(x3, requires_grad=True)

            # Test with device specification
            x4 = paddle.zeros([4, 3], dtype='float32')
            out4 = paddle.rand_like(x4, device=place)

            # Test with all parameters including device
            x5 = paddle.zeros([2, 3], dtype='float32')
            out5 = paddle.rand_like(
                x5,
                name="test_all_params",
                dtype='float64',
                device=place,
                requires_grad=False,
            )

    def test_run(self):
        self.run_net(False)
        if core.is_compiled_with_cuda():
            self.run_net(True)


if __name__ == "__main__":
    unittest.main()
