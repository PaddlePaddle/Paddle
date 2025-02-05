#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import unittest
from contextlib import contextmanager

import numpy as np
from op_test import OpTest

import paddle
from paddle import base, static


class TestElementwiseModOp(OpTest):
    def init_kernel_type(self):
        self.use_mkldnn = False

    def setUp(self):
        self.op_type = "elementwise_floordiv"
        self.prim_op_type = "comp"
        self.python_api = paddle.floor_divide
        self.public_python_api = paddle.floor_divide
        self.dtype = np.int32
        self.axis = -1
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(self.x),
            'Y': OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.out}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def init_input_output(self):
        self.x = np.random.uniform(0, 10000, [10, 10]).astype(self.dtype)
        self.y = np.random.uniform(0, 1000, [10, 10]).astype(self.dtype)
        self.out = np.floor_divide(self.x, self.y)

    def init_dtype(self):
        pass

    def init_axis(self):
        pass


class TestElementwiseFloorDivOp_ZeroDim1(TestElementwiseModOp):
    def init_input_output(self):
        self.x = np.random.uniform(0, 10000, []).astype(self.dtype)
        self.y = np.random.uniform(0, 1000, []).astype(self.dtype)
        self.out = np.floor_divide(self.x, self.y)


class TestElementwiseFloorDivOp_ZeroDim2(TestElementwiseModOp):
    def init_input_output(self):
        self.x = np.random.uniform(0, 10000, [10, 10]).astype(self.dtype)
        self.y = np.random.uniform(0, 1000, []).astype(self.dtype)
        self.out = np.floor_divide(self.x, self.y)


class TestElementwiseFloorDivOp_ZeroDim3(TestElementwiseModOp):
    def init_input_output(self):
        self.x = np.random.uniform(0, 10000, []).astype(self.dtype)
        self.y = np.random.uniform(0, 1000, [10, 10]).astype(self.dtype)
        self.out = np.floor_divide(self.x, self.y)


class TestElementwiseModOp_scalar(TestElementwiseModOp):
    def init_input_output(self):
        scale_x = random.randint(0, 100000000)
        scale_y = random.randint(1, 100000000)
        self.x = (np.random.rand(2, 3, 4) * scale_x).astype(self.dtype)
        self.y = (np.random.rand(1) * scale_y + 1).astype(self.dtype)
        self.out = np.floor_divide(self.x, self.y)


class TestElementwiseModOpInverse(TestElementwiseModOp):
    def init_input_output(self):
        self.x = np.random.uniform(0, 10000, [10]).astype(self.dtype)
        self.y = np.random.uniform(0, 1000, [10, 10]).astype(self.dtype)
        self.out = np.floor_divide(self.x, self.y)


class TestElementwiseFloorDivOp_OneDim(TestElementwiseModOp):
    def init_input_output(self):
        self.x = np.random.uniform(0, 10000, [10]).astype(self.dtype)
        self.y = np.random.uniform(0, 1000, [10]).astype(self.dtype)
        self.out = np.floor_divide(self.x, self.y)


@contextmanager
def device_guard(device=None):
    old = paddle.get_device()
    yield paddle.set_device(device)
    paddle.set_device(old)


class TestFloorDivideOp(unittest.TestCase):

    def test_static(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not base.core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            for dtype in (
                'int32',
                'int64',
                'float16',
                'float32',
                'float64',
            ):
                np_x = np.array([2, 3, 8, 7]).astype(dtype)
                np_y = np.array([1, 5, 3, 3]).astype(dtype)
                mp, sp = static.Program(), static.Program()
                with static.program_guard(mp, sp):
                    x = static.data("x", shape=[4], dtype=dtype)
                    y = static.data("y", shape=[4], dtype=dtype)
                    z = paddle.floor_divide(x, y)
                exe = static.Executor(p)
                exe.run(sp)
                [np_z] = exe.run(
                    mp, feed={"x": np_x, "y": np_y}, fetch_list=[z]
                )
                z_expected = np.floor_divide(np_x, np_y)
                self.assertEqual((np_z == z_expected).all(), True)

            np_x = np.array([2, 3, 8, 7]).astype("uint16")
            np_y = np.array([1, 5, 3, 3]).astype("uint16")
            mp, sp = static.Program(), static.Program()
            with static.program_guard(mp, sp):
                x = static.data("x", shape=[4], dtype="uint16")
                y = static.data("y", shape=[4], dtype="uint16")
                z = paddle.floor_divide(x, y)
            exe = static.Executor(p)
            exe.run(sp)
            [np_z] = exe.run(mp, feed={"x": np_x, "y": np_y}, fetch_list=[z])
            z_expected = np.array([16384, 0, 16384, 16384], dtype='uint16')
            self.assertEqual((np_z == z_expected).all(), True)

    def test_dygraph(self):
        paddle.disable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not base.core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            for dtype in (
                'uint8',
                'int8',
                'int16',
                'int32',
                'int64',
                'float16',
                'float32',
                'float64',
            ):
                np_x = np.array([2, 3, 8, 7]).astype(dtype)
                np_y = np.array([1, 5, 3, 3]).astype(dtype)
                x = paddle.to_tensor(np_x)
                y = paddle.to_tensor(np_y)
                z = paddle.floor_divide(x, y)
                np_z = z.numpy()
                z_expected = np.floor_divide(np_x, np_y)
                self.assertEqual((np_z == z_expected).all(), True)

            np_x = np.array([2, 3, 8, 7])
            np_y = np.array([1, 5, 3, 3])
            x = paddle.to_tensor(np_x, dtype='bfloat16')
            y = paddle.to_tensor(np_y, dtype="bfloat16")
            z = paddle.floor_divide(x, y)
            np_z = z.numpy()
            z_expected = np.array([16384, 0, 16384, 16384], dtype='uint16')
            self.assertEqual((np_z == z_expected).all(), True)

            for dtype in (
                'int8',
                'int16',
                'int32',
                'int64',
                'float16',
                'float32',
                'float64',
            ):
                np_x = -np.array([2, 3, 8, 7]).astype(dtype)
                np_y = np.array([1, 5, 3, 3]).astype(dtype)
                x = paddle.to_tensor(np_x)
                y = paddle.to_tensor(np_y)
                z = paddle.floor_divide(x, y)
                np_z = z.numpy()
                z_expected = np.floor_divide(np_x, np_y)
                self.assertEqual((np_z == z_expected).all(), True)

            np_x = -np.array([2, 3, 8, 7])
            np_y = np.array([1, 5, 3, 3])
            x = paddle.to_tensor(np_x, dtype='bfloat16')
            y = paddle.to_tensor(np_y, dtype="bfloat16")
            z = paddle.floor_divide(x, y)
            np_z = z.numpy()
            z_expected = np.array([49152, 49024, 49216, 49216], dtype='uint16')
            self.assertEqual((np_z == z_expected).all(), True)

            for dtype in ('float32', 'float64', 'float16'):
                try:
                    # divide by zero
                    np_x = np.array([2])
                    np_y = np.array([0, 0, 0])
                    x = paddle.to_tensor(np_x, dtype=dtype)
                    y = paddle.to_tensor(np_y, dtype=dtype)
                    z = paddle.floor_divide(x, y)
                    np_z = z.numpy()
                    # [np.inf, np.inf, np.inf]
                    z_expected = np.floor_divide(np_x, np_y)
                    self.assertEqual((np_z == z_expected).all(), True)
                except Exception as e:
                    pass

            # divide by zero
            np_x = np.array([2])
            np_y = np.array([0, 0, 0])
            x = paddle.to_tensor(np_x, dtype='bfloat16')
            y = paddle.to_tensor(np_y, dtype="bfloat16")
            z = paddle.floor_divide(x, y)
            np_z = z.numpy()
            z_expected = np.array([32640, 32640, 32640], dtype='uint16')
            self.assertEqual((np_z == z_expected).all(), True)

        with device_guard('cpu'):
            # divide by zero
            np_x = np.array([2, 3, 4])
            np_y = np.array([0])
            x = paddle.to_tensor(np_x)
            y = paddle.to_tensor(np_y)
            try:
                z = x // y
            except Exception as e:
                pass

            # divide by zero
            for dtype in ("uint8", 'int8', 'int16', 'int32', 'int64'):
                np_x = np.array([2])
                np_y = np.array([0, 0, 0])
                x = paddle.to_tensor(np_x, dtype=dtype)
                y = paddle.to_tensor(np_y, dtype=dtype)
                try:
                    z = x // y
                except Exception as e:
                    pass

        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
