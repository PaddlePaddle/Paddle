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
from op_test import get_device_place
from utils import dygraph_guard, static_guard

import paddle
from paddle import base, core


# Test python API
class TestRandnLikeAPI(unittest.TestCase):
    def setUp(self):
        self.x_float16 = np.zeros((10, 12)).astype("float16")
        self.x_float32 = np.zeros((10, 12)).astype("float32")
        self.x_float64 = np.zeros((10, 12)).astype("float64")

        self.dtype = ["float16", "float32", "float64"]
        self.place = get_device_place()

    def test_static_api_basic(self):
        """Test basic static API functionality"""
        with (
            static_guard(),
            paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ),
        ):
            x_float32 = paddle.static.data(
                name="x_float32", shape=[10, 12], dtype="float32"
            )

            # Test with default parameters
            out1 = paddle.randn_like(x_float32)

            # Test with specified name
            out2 = paddle.randn_like(x_float32, name="test_randn_like")

            exe = paddle.static.Executor(self.place)
            outs = exe.run(
                feed={'x_float32': self.x_float32}, fetch_list=[out1, out2]
            )

            for out in outs:
                self.assertEqual(out.shape, (10, 12))
                self.assertEqual(out.dtype, np.float32)
                # Test normal distribution range (approximately 99.7% within 3 std)
                self.assertTrue(((out >= -25) & (out <= 25)).all())

    def test_static_api_with_device(self):
        """Test static API with device specification"""
        with (
            static_guard(),
            paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ),
        ):
            x_float32 = paddle.static.data(
                name="x_float32", shape=[10, 12], dtype="float32"
            )

            # Test with CPU device
            out1 = paddle.randn_like(x_float32, device=base.CPUPlace())

            place = base.CPUPlace()
            exe = paddle.static.Executor(place)
            result = exe.run(
                feed={'x_float32': self.x_float32}, fetch_list=[out1]
            )[0]

            self.assertEqual(result.shape, (10, 12))
            self.assertTrue(((result >= -25) & (result <= 25)).all())

            # Test with CUDA device if available
            if core.is_compiled_with_cuda():
                out2 = paddle.randn_like(x_float32, device=base.CUDAPlace(0))
                place_cuda = base.CUDAPlace(0)
                exe_cuda = paddle.static.Executor(place_cuda)
                result_cuda = exe_cuda.run(
                    feed={'x_float32': self.x_float32}, fetch_list=[out2]
                )[0]

                self.assertEqual(result_cuda.shape, (10, 12))
                self.assertTrue(
                    ((result_cuda >= -25) & (result_cuda <= 25)).all()
                )

    def test_static_api_with_dtype(self):
        """Test static API with different dtype specifications"""
        with (
            static_guard(),
            paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ),
        ):
            x_float32 = paddle.static.data(
                name="x_float32", shape=[10, 12], dtype="float32"
            )

            exe = paddle.static.Executor(self.place)

            # Test with different dtypes
            for dtype in self.dtype:
                if dtype == "float16" and not core.is_compiled_with_cuda():
                    continue

                out = paddle.randn_like(x_float32, dtype=dtype)
                result = exe.run(
                    feed={'x_float32': self.x_float32}, fetch_list=[out]
                )[0]

                self.assertEqual(result.shape, (10, 12))
                self.assertEqual(result.dtype, np.dtype(dtype))
                self.assertTrue(((result >= -25) & (result <= 25)).all())

    def test_static_api_with_fp16(self):
        with static_guard():
            if paddle.is_compiled_with_cuda():
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    x_float16 = paddle.static.data(
                        name="x_float16", shape=[10, 12], dtype="float16"
                    )
                    exe = paddle.static.Executor(self.place)
                    outlist1 = [
                        paddle.randn_like(x_float16, dtype=dtype)
                        for dtype in self.dtype
                    ]
                    outs1 = exe.run(
                        feed={'x_float16': self.x_float16}, fetch_list=outlist1
                    )
                    for out, dtype in zip(outs1, self.dtype):
                        self.assertEqual(out.dtype, np.dtype(dtype))
                        self.assertTrue(
                            ((out >= -25) & (out <= 25)).all(), True
                        )

    def test_static_api_with_fp32(self):
        with (
            static_guard(),
            paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ),
        ):
            x_float32 = paddle.static.data(
                name="x_float32", shape=[10, 12], dtype="float32"
            )
            exe = paddle.static.Executor(self.place)
            outlist2 = [
                paddle.randn_like(x_float32, dtype=dtype)
                for dtype in self.dtype
            ]
            outs2 = exe.run(
                feed={'x_float32': self.x_float32}, fetch_list=outlist2
            )
            for out, dtype in zip(outs2, self.dtype):
                self.assertEqual(out.dtype, np.dtype(dtype))
                self.assertTrue(((out >= -25) & (out <= 25)).all(), True)

    def test_static_api_with_fp64(self):
        with (
            static_guard(),
            paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ),
        ):
            x_float64 = paddle.static.data(
                name="x_float64", shape=[10, 12], dtype="float64"
            )
            exe = paddle.static.Executor(self.place)
            outlist3 = [
                paddle.randn_like(x_float64, dtype=dtype)
                for dtype in self.dtype
            ]
            outs3 = exe.run(
                feed={'x_float64': self.x_float64}, fetch_list=outlist3
            )
            for out, dtype in zip(outs3, self.dtype):
                self.assertEqual(out.dtype, np.dtype(dtype))
                self.assertTrue(((out >= -25) & (out <= 25)).all(), True)

    def test_dygraph_api_basic(self):
        """Test basic dygraph API functionality"""
        with dygraph_guard():
            for x_np in [self.x_float32, self.x_float64]:
                x = paddle.to_tensor(x_np, place=self.place)

                # Test with default parameters
                out1 = paddle.randn_like(x)
                self.assertEqual(out1.shape, x.shape)
                self.assertEqual(out1.dtype, x.dtype)
                # Check device consistency
                self.assertEqual(str(x.place), str(out1.place))
                self.assertTrue(
                    ((out1.numpy() >= -25) & (out1.numpy() <= 25)).all()
                )

                # Test with name parameter
                out2 = paddle.randn_like(x, name="test_randn_like")
                self.assertEqual(out2.shape, x.shape)
                self.assertEqual(out2.dtype, x.dtype)
                # Check device consistency
                self.assertEqual(str(x.place), str(out2.place))
                self.assertTrue(
                    ((out2.numpy() >= -25) & (out2.numpy() <= 25)).all()
                )

            # Test with float16 if CUDA is available
            if core.is_compiled_with_cuda():
                x = paddle.to_tensor(self.x_float16, place=self.place)
                out = paddle.randn_like(x)
                self.assertEqual(out.shape, x.shape)
                self.assertEqual(out.dtype, x.dtype)
                # Check device consistency
                self.assertEqual(str(x.place), str(out.place))
                self.assertTrue(
                    ((out.numpy() >= -25) & (out.numpy() <= 25)).all()
                )

    def test_dygraph_api_with_dtype(self):
        """Test dygraph API with different dtype specifications"""
        with dygraph_guard():
            x = paddle.to_tensor(self.x_float32, place=self.place)

            for dtype in self.dtype:
                if dtype == "float16" and not core.is_compiled_with_cuda():
                    continue

                out = paddle.randn_like(x, dtype=dtype)
                self.assertEqual(out.shape, x.shape)
                self.assertEqual(out.dtype, getattr(paddle, dtype))
                # Check device consistency with input
                self.assertEqual(str(x.place), str(out.place))
                self.assertTrue(
                    ((out.numpy() >= -25) & (out.numpy() <= 25)).all()
                )

    def test_dygraph_api_with_requires_grad(self):
        """Test dygraph API with requires_grad parameter"""
        with dygraph_guard():
            x = paddle.to_tensor(self.x_float32, place=self.place)

            # Test requires_grad=True
            out1 = paddle.randn_like(x, requires_grad=True)
            self.assertEqual(out1.shape, x.shape)
            self.assertEqual(out1.dtype, x.dtype)
            self.assertFalse(out1.stop_gradient)
            # Check device consistency
            self.assertEqual(str(x.place), str(out1.place))
            self.assertTrue(
                ((out1.numpy() >= -25) & (out1.numpy() <= 25)).all()
            )

            # Test requires_grad=False
            out2 = paddle.randn_like(x, requires_grad=False)
            self.assertEqual(out2.shape, x.shape)
            self.assertEqual(out2.dtype, x.dtype)
            self.assertTrue(out2.stop_gradient)
            # Check device consistency
            self.assertEqual(str(x.place), str(out2.place))
            self.assertTrue(
                ((out2.numpy() >= -25) & (out2.numpy() <= 25)).all()
            )

    def test_dygraph_api_with_device(self):
        """Test dygraph API with device specification"""
        with dygraph_guard():
            x = paddle.to_tensor(self.x_float32)

            # Test with CPU device
            out1 = paddle.randn_like(x, device=paddle.CPUPlace())
            self.assertEqual(out1.shape, x.shape)
            self.assertEqual(out1.dtype, x.dtype)
            self.assertTrue(out1.place.is_cpu_place())
            self.assertTrue(
                ((out1.numpy() >= -25) & (out1.numpy() <= 25)).all()
            )

            # Test with CUDA device if available
            if core.is_compiled_with_cuda():
                out2 = paddle.randn_like(x, device=paddle.CUDAPlace(0))
                self.assertEqual(out2.shape, x.shape)
                self.assertEqual(out2.dtype, x.dtype)
                self.assertTrue(out2.place.is_gpu_place())
                self.assertTrue(
                    ((out2.numpy() >= -25) & (out2.numpy() <= 25)).all()
                )

    def test_dygraph_api_combined_params(self):
        """Test dygraph API with combined parameters"""
        with dygraph_guard():
            x = paddle.to_tensor(self.x_float32)

            # Test dtype + requires_grad
            out1 = paddle.randn_like(x, dtype="float64", requires_grad=True)
            self.assertEqual(out1.shape, x.shape)
            self.assertEqual(out1.dtype, paddle.float64)
            self.assertFalse(out1.stop_gradient)
            self.assertTrue(
                ((out1.numpy() >= -25) & (out1.numpy() <= 25)).all()
            )

            # Test all parameters together
            out2 = paddle.randn_like(
                x,
                name="combined_test",
                dtype="float64",
                device=paddle.CPUPlace(),
                requires_grad=False,
            )
            self.assertEqual(out2.shape, x.shape)
            self.assertEqual(out2.dtype, paddle.float64)
            self.assertTrue(out2.stop_gradient)
            self.assertTrue(out2.place.is_cpu_place())
            self.assertTrue(
                ((out2.numpy() >= -25) & (out2.numpy() <= 25)).all()
            )

    def test_device_consistency_default_behavior(self):
        """Test that output tensor is on the same device as input tensor by default"""
        with dygraph_guard():
            # Test CPU case
            x_cpu = paddle.to_tensor(self.x_float32, place=paddle.CPUPlace())
            out_cpu = paddle.randn_like(x_cpu)  # No device specified

            self.assertTrue(x_cpu.place.is_cpu_place())
            self.assertTrue(out_cpu.place.is_cpu_place())
            self.assertEqual(str(x_cpu.place), str(out_cpu.place))

            # Test CUDA case if available
            if core.is_compiled_with_cuda():
                x_cuda = paddle.to_tensor(
                    self.x_float32, place=paddle.CUDAPlace(0)
                )
                out_cuda = paddle.randn_like(x_cuda)  # No device specified

                self.assertTrue(x_cuda.place.is_gpu_place())
                self.assertTrue(out_cuda.place.is_gpu_place())
                self.assertEqual(str(x_cuda.place), str(out_cuda.place))

    def test_device_override_behavior(self):
        """Test that explicitly specified device overrides input tensor device"""
        with dygraph_guard():
            if not core.is_compiled_with_cuda():
                return

            # Create tensor on GPU
            x_gpu = paddle.to_tensor(self.x_float32, place=paddle.CUDAPlace(0))

            # Force output to CPU using device parameter
            out_cpu = paddle.randn_like(x_gpu, device=paddle.CPUPlace())

            self.assertTrue(x_gpu.place.is_gpu_place())
            self.assertTrue(out_cpu.place.is_cpu_place())
            self.assertNotEqual(str(x_gpu.place), str(out_cpu.place))

            # Create tensor on CPU
            x_cpu = paddle.to_tensor(self.x_float32, place=paddle.CPUPlace())

            # Force output to GPU using device parameter
            out_gpu = paddle.randn_like(x_cpu, device=paddle.CUDAPlace(0))

            self.assertTrue(x_cpu.place.is_cpu_place())
            self.assertTrue(out_gpu.place.is_gpu_place())
            self.assertNotEqual(str(x_cpu.place), str(out_gpu.place))

    def test_different_shapes(self):
        """Test with different input shapes"""
        with dygraph_guard():
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
                out = paddle.randn_like(x)
                self.assertEqual(out.shape, shape)
                self.assertEqual(str(x.place), str(out.place))
                self.assertTrue(
                    ((out.numpy() >= -25) & (out.numpy() <= 25)).all()
                )

    def test_default_dtype_behavior(self):
        """Test default dtype behavior"""
        with dygraph_guard():
            # Test that output dtype matches input dtype when dtype=None
            dtypes_to_test = ['float32', 'float64']
            if core.is_compiled_with_cuda():
                dtypes_to_test.append('float16')

            for dtype_str in dtypes_to_test:
                x = paddle.zeros((3, 4), dtype=dtype_str)
                out = paddle.randn_like(x)  # dtype=None (default)
                self.assertEqual(out.dtype, x.dtype)
                self.assertEqual(str(x.place), str(out.place))
                self.assertTrue(
                    ((out.numpy() >= -25) & (out.numpy() <= 25)).all()
                )

    def test_dygraph_api(self):
        """Legacy test method - kept for backward compatibility"""
        with dygraph_guard():
            for x in [
                self.x_float32,
                self.x_float64,
            ]:
                x_inputs = paddle.to_tensor(x, place=self.place)
                for dtype in self.dtype:
                    out = paddle.randn_like(x_inputs, dtype=dtype)
                    self.assertEqual(out.numpy().dtype, np.dtype(dtype))
                    self.assertTrue(
                        ((out.numpy() >= -25) & (out.numpy() <= 25)).all(), True
                    )

            x_inputs = paddle.to_tensor(self.x_float32)
            out = paddle.randn_like(x_inputs)
            self.assertEqual(out.numpy().dtype, np.dtype("float32"))
            self.assertTrue(
                ((out.numpy() >= -25) & (out.numpy() <= 25)).all(), True
            )

            if paddle.is_compiled_with_cuda():
                x_inputs = paddle.to_tensor(self.x_float16)
                for dtype in self.dtype:
                    out = paddle.randn_like(x_inputs, dtype=dtype)
                    self.assertEqual(out.numpy().dtype, np.dtype(dtype))
                    self.assertTrue(
                        ((out.numpy() >= -25) & (out.numpy() <= 25)).all(), True
                    )


class TestRandnLikeOpForDygraph(unittest.TestCase):
    """
    Test randn_like operation in dygraph mode with different scenarios.
    """

    def run_net(self, use_cuda=False):
        place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
        with base.dygraph.guard(place):
            # Test basic functionality
            x1 = paddle.zeros([3, 4], dtype='float32')
            out1 = paddle.randn_like(x1)

            # Test with different dtype
            x2 = paddle.zeros([3, 4], dtype='float32')
            out2 = paddle.randn_like(x2, dtype='float64')

            # Test with requires_grad
            x3 = paddle.zeros([2, 5], dtype='float32')
            out3 = paddle.randn_like(x3, requires_grad=True)

            # Test with device specification
            x4 = paddle.zeros([4, 3], dtype='float32')
            out4 = paddle.randn_like(x4, device=place)

            # Test with all parameters including device
            x5 = paddle.zeros([2, 3], dtype='float32')
            out5 = paddle.randn_like(
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
