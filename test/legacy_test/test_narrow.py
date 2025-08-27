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
from paddle import base


def check_narrow_alias(input_tensor, output_tensor, dim, start):
    """
    Check whether output_tensor is a view (alias) of input_tensor.
    """
    import numpy as np

    # Skip empty tensors
    if output_tensor.numel() == 0:
        return True

    # Prepare index for the first element in output_tensor
    idx_out = tuple([0] * output_tensor.ndim)
    # Prepare the corresponding index in input_tensor
    idx_in = [0] * input_tensor.ndim
    idx_in[dim] = start
    idx_in = tuple(idx_in)
    # Save original value
    origin_val = output_tensor[idx_out].numpy().copy()
    # Value to write
    test_val = np.array(999, dtype=output_tensor.numpy().dtype)
    if str(output_tensor.dtype) == "paddle.bool":
        test_val = np.array(True, dtype=output_tensor.numpy().dtype)

    # Try inplace modification
    try:
        output_tensor[idx_out] = test_val
    except Exception as e:
        print("inplace failed:", e)
        return

    # Read the corresponding value from input_tensor and output_tensor
    input_val = input_tensor[idx_in].numpy()
    output_val = output_tensor[idx_out].numpy()

    # Restore the original value
    output_tensor[idx_out] = origin_val

    # Check if they both changed to test_val (alias)
    is_alias = np.allclose(input_val, test_val) and np.allclose(
        output_val, test_val
    )
    return is_alias


@unittest.skipIf(paddle.device.get_device().startswith("xpu"), "Skip on XPU")
class TestNarrowBase(unittest.TestCase):
    @unittest.skipIf(
        paddle.device.get_device().startswith("xpu"), "Skip on XPU"
    )
    def setUp(self):
        self.input_np = np.array([1, 2, 3, 4, 5], dtype='float32')
        self.input_shape = self.input_np.shape
        self.input_dtype = 'float32'
        self.op_static = lambda x: paddle.narrow(x, dim=0, start=1, length=3)
        self.op_dygraph = lambda x: paddle.narrow(x, dim=0, start=1, length=3)
        self.expected = lambda x: x[1:4]
        self.places = [None, paddle.CPUPlace()]
        self.dim = 0
        self.start = 1
        self.length = 3

    def check_dygraph_result(self, place):
        with base.dygraph.guard(place):
            # check forward
            input = paddle.to_tensor(self.input_np, stop_gradient=False)
            result = self.op_dygraph(input)
            expect = (
                self.expected(self.input_np)
                if callable(self.expected)
                else self.expected
            )
            np.testing.assert_allclose(result.numpy(), expect, rtol=1e-05)

            # check backward
            result.sum().backward()
            mask = np.zeros_like(self.input_np)
            dim = self.dim
            start = self.start
            length = self.length
            if dim < 0:
                dim += self.input_np.ndim
            slices = [slice(None)] * self.input_np.ndim
            slices[dim] = slice(start, start + length)
            mask[tuple(slices)] = 1
            np.testing.assert_allclose(input.grad.numpy(), mask, rtol=1e-05)

            # check inplace
            is_alias = check_narrow_alias(input, result, self.dim, self.start)
            self.assertTrue(
                is_alias,
                f"narrow should be an alias! input={input.numpy()}, result={result.numpy()}",
            )

    @unittest.skipIf(
        paddle.device.get_device().startswith("xpu"), "Skip on XPU"
    )
    def test_dygraph(self):
        for place in self.places:
            self.check_dygraph_result(place=place)


class TestPaddleNarrow2D(TestNarrowBase):
    def setUp(self):
        self.input_np = np.arange(1, 10, dtype='int32').reshape(3, 3)
        self.input_shape = self.input_np.shape
        self.input_dtype = 'int32'
        self.op_static = lambda x: paddle.narrow(x, dim=1, start=0, length=2)
        self.op_dygraph = lambda x: paddle.narrow(x, dim=1, start=0, length=2)
        self.expected = lambda x: x[:, 0:2]
        self.places = [None, paddle.CPUPlace()]
        self.dim = 1
        self.start = 0
        self.length = 2


class TestPaddleNarrow3D(TestNarrowBase):
    def setUp(self):
        self.input_np = np.arange(2 * 3 * 4, dtype='int64').reshape(2, 3, 4)
        self.input_shape = self.input_np.shape
        self.input_dtype = 'int64'
        self.op_static = lambda x: paddle.narrow(x, dim=2, start=1, length=2)
        self.op_dygraph = lambda x: paddle.narrow(x, dim=2, start=1, length=2)
        self.expected = lambda x: x[:, :, 1:3]
        self.places = [None, paddle.CPUPlace()]
        self.dim = 2
        self.start = 1
        self.length = 2


class TestPaddleNarrowStart0(TestNarrowBase):
    def setUp(self):
        self.input_np = np.array([1, 2, 3], dtype='float32')
        self.input_shape = self.input_np.shape
        self.input_dtype = 'float32'
        self.op_static = lambda x: paddle.narrow(x, dim=0, start=0, length=1)
        self.op_dygraph = lambda x: paddle.narrow(x, dim=0, start=0, length=1)
        self.expected = lambda x: x[0:1]
        self.places = [None, paddle.CPUPlace()]
        self.dim = 0
        self.start = 0
        self.length = 1


class TestPaddleNarrowLength0(TestNarrowBase):
    def setUp(self):
        self.input_np = np.arange(6, dtype='float32')
        self.input_shape = self.input_np.shape
        self.input_dtype = 'float32'
        self.op_static = lambda x: paddle.narrow(x, dim=0, start=2, length=0)
        self.op_dygraph = lambda x: paddle.narrow(x, dim=0, start=2, length=0)
        self.expected = lambda x: x[2:2]
        self.places = [None, paddle.CPUPlace()]
        self.dim = 0
        self.start = 2
        self.length = 0


class TestPaddleNarrowNegativeAxis(TestNarrowBase):
    def setUp(self):
        self.input_np = np.arange(6, dtype='float32').reshape(2, 3)
        self.input_shape = self.input_np.shape
        self.input_dtype = 'float32'
        self.op_static = lambda x: paddle.narrow(x, dim=-1, start=1, length=2)
        self.op_dygraph = lambda x: paddle.narrow(x, dim=-1, start=1, length=2)
        self.expected = lambda x: x[:, 1:3]
        self.places = [None, paddle.CPUPlace()]
        self.dim = -1
        self.start = 1
        self.length = 2


class TestPaddleNarrowDtypeInt(TestNarrowBase):
    def setUp(self):
        self.input_np = np.arange(10, dtype='int32')
        self.input_shape = self.input_np.shape
        self.input_dtype = 'int32'
        self.op_static = lambda x: paddle.narrow(x, dim=0, start=3, length=2)
        self.op_dygraph = lambda x: paddle.narrow(x, dim=0, start=3, length=2)
        self.expected = lambda x: x[3:5]
        self.places = [None, paddle.CPUPlace()]
        self.dim = 0
        self.start = 3
        self.length = 2


class TestPaddleNarrowDtypeBool(TestNarrowBase):
    def setUp(self):
        self.input_np = np.array([True, False, True, False])
        self.input_shape = self.input_np.shape
        self.input_dtype = 'bool'
        self.op_static = lambda x: paddle.narrow(x, dim=0, start=1, length=2)
        self.op_dygraph = lambda x: paddle.narrow(x, dim=0, start=1, length=2)
        self.expected = lambda x: x[1:3]
        self.places = [None, paddle.CPUPlace()]
        self.dim = 0
        self.start = 1
        self.length = 2


class TestPaddleNarrowLargeTensor(TestNarrowBase):
    def setUp(self):
        self.input_np = np.random.randn(10000).astype('float32')
        self.input_shape = self.input_np.shape
        self.input_dtype = 'float32'
        self.op_static = lambda x: paddle.narrow(
            x, dim=0, start=5000, length=101
        )
        self.op_dygraph = lambda x: paddle.narrow(
            x, dim=0, start=5000, length=101
        )
        self.expected = lambda x: x[5000 : 5000 + 101]
        self.places = [None, paddle.CPUPlace()]
        self.dim = 0
        self.start = 5000
        self.length = 101


class TestPaddleNarrowOutOfBounds(unittest.TestCase):
    def test_out_of_bounds(self):
        arr = np.arange(5, dtype='int32')
        with self.assertRaises(AssertionError):
            paddle.narrow(paddle.to_tensor(arr), dim=0, start=4, length=2)
        self.places = [None, paddle.CPUPlace()]


class TestPaddleNarrowNegativeStart(unittest.TestCase):
    def test_negative_start(self):
        arr = np.arange(5, dtype='float32')
        with self.assertRaises(AssertionError):
            paddle.narrow(paddle.to_tensor(arr), dim=0, start=-1, length=2)
        self.places = [None, paddle.CPUPlace()]


class TestPaddleNarrowMultiDim(TestNarrowBase):
    def setUp(self):
        self.input_np = np.arange(24).reshape((2, 3, 4)).astype('float32')
        self.input_shape = self.input_np.shape
        self.input_dtype = 'float32'
        self.op_static = lambda x: paddle.narrow(x, dim=1, start=1, length=1)
        self.op_dygraph = lambda x: paddle.narrow(x, dim=1, start=1, length=1)
        self.expected = lambda x: x[:, 1:2, :]
        self.places = [None, paddle.CPUPlace()]
        self.dim = 1
        self.start = 1
        self.length = 1


class TestPaddleNarrowEmptyTensor(TestNarrowBase):
    def setUp(self):
        self.input_np = np.empty((0, 4), dtype='float32')
        self.input_shape = self.input_np.shape
        self.input_dtype = 'float32'
        self.op_static = lambda x: paddle.narrow(x, dim=0, start=0, length=0)
        self.op_dygraph = lambda x: paddle.narrow(x, dim=0, start=0, length=0)
        self.expected = lambda x: x[0:0, :]
        self.places = [None, paddle.CPUPlace()]
        self.dim = 0
        self.start = 0
        self.length = 0


@unittest.skipIf(paddle.device.get_device().startswith("xpu"), "Skip on XPU")
class TestNarrowExtra(unittest.TestCase):
    @unittest.skipIf(
        paddle.device.get_device().startswith("xpu"), "Skip on XPU"
    )
    def test_start_tensor(self):
        arr = np.arange(10, dtype='int64')
        x = paddle.to_tensor(arr)
        s = paddle.to_tensor(3, dtype='int64')
        out = paddle.narrow(x, dim=0, start=s, length=2)
        np.testing.assert_array_equal(out.numpy(), arr[3:5])

    @unittest.skipIf(
        paddle.device.get_device().startswith("xpu"), "Skip on XPU"
    )
    def test_start_tensor_wrong_dtype(self):
        arr = np.arange(10, dtype='float32')
        x = paddle.to_tensor(arr)
        s = paddle.to_tensor(3.1, dtype='float32')
        with self.assertRaises(AssertionError):
            paddle.narrow(x, dim=0, start=s, length=2)

    @unittest.skipIf(
        paddle.device.get_device().startswith("xpu"), "Skip on XPU"
    )
    def test_start_tensor_wrong_shape(self):
        arr = np.arange(10, dtype='float32')
        x = paddle.to_tensor(arr)
        s = paddle.to_tensor([1, 2], dtype='int64')
        with self.assertRaises(AssertionError):
            paddle.narrow(x, dim=0, start=s, length=2)

    @unittest.skipIf(
        paddle.device.get_device().startswith("xpu"), "Skip on XPU"
    )
    def test_dim_out_of_range(self):
        arr = np.arange(10)
        x = paddle.to_tensor(arr)
        with self.assertRaises(IndexError):
            paddle.narrow(x, dim=2, start=0, length=1)
        with self.assertRaises(IndexError):
            paddle.narrow(x, dim=-2, start=0, length=1)

    @unittest.skipIf(
        paddle.device.get_device().startswith("xpu"), "Skip on XPU"
    )
    def test_start_out_of_range(self):
        arr = np.arange(5)
        x = paddle.to_tensor(arr)
        with self.assertRaises(AssertionError):
            paddle.narrow(x, dim=0, start=6, length=1)
        with self.assertRaises(AssertionError):
            paddle.narrow(x, dim=0, start=-6, length=1)

    @unittest.skipIf(
        paddle.device.get_device().startswith("xpu"), "Skip on XPU"
    )
    def test_length_negative(self):
        arr = np.arange(5)
        x = paddle.to_tensor(arr)
        with self.assertRaises(AssertionError):
            paddle.narrow(x, dim=0, start=1, length=-1)

    @unittest.skipIf(
        paddle.device.get_device().startswith("xpu"), "Skip on XPU"
    )
    def test_0_dim_tensor(self):
        x = paddle.to_tensor(111)
        with self.assertRaises(AssertionError):
            paddle.narrow(x, dim=0, start=0, length=1)

    @unittest.skipIf(
        paddle.device.get_device().startswith("xpu"), "Skip on XPU"
    )
    def test_start_plus_length_overflow(self):
        arr = np.arange(5)
        x = paddle.to_tensor(arr)
        with self.assertRaises(AssertionError):
            paddle.narrow(x, dim=0, start=3, length=3)

    @unittest.skipIf(
        paddle.device.get_device().startswith("xpu"), "Skip on XPU"
    )
    def test_negative_start(self):
        arr = np.arange(8)
        x = paddle.to_tensor(arr)
        out = paddle.narrow(x, dim=0, start=-3, length=2)
        np.testing.assert_array_equal(out.numpy(), arr[5:7])

    @unittest.skipIf(
        paddle.device.get_device().startswith("xpu"), "Skip on XPU"
    )
    def test_negative_dim(self):
        arr = np.arange(12).reshape(3, 4)
        x = paddle.to_tensor(arr)
        out = paddle.narrow(x, dim=-1, start=2, length=2)
        np.testing.assert_array_equal(out.numpy(), arr[:, 2:4])


if __name__ == '__main__':
    unittest.main()
