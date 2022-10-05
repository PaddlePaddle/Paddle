#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
import paddle.fluid.core as core

import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework


def reference_unique_consecutive(X, return_inverse=False, return_counts=False):
    """
    Reference unique_consecutive implementation using python.
    Args:
        x(Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        return_inverse(bool, optional): If True, also return the indices for where elements in
            the original input ended up in the returned unique consecutive tensor. Default is False.
        return_counts(bool, optional): If True, also return the counts for each unique consecutive element.
    """
    X = list(X)
    counts_vec = [1] * len(X)
    i = 0
    counts = 1
    last = 0
    inverse_vec = [0] * len(X)
    inverse_vec[last] = i
    cnt = 0
    while i < len(X) - 1:
        if X[i] == X[i + 1]:
            if return_counts:
                counts_vec[cnt] += 1
            del X[i]
        else:
            i += 1
            cnt += 1
        if return_inverse:
            last += 1
            inverse_vec[last] = i
    if return_counts:
        counts_vec = counts_vec[:len(X)]
    if return_inverse and return_counts:
        return X, np.array(inverse_vec), np.array(counts_vec)
    elif return_counts:
        return X, np.array(counts_vec)
    elif return_inverse:
        return X, np.array(inverse_vec)
    else:
        return X


class TestUniqueConsecutiveOp(OpTest):
    """case 1"""

    def config(self):
        self.x_size = 100
        self.x_range = 20
        self.return_inverse = False
        self.return_counts = False
        self.python_api = paddle.unique_consecutive

    def init_kernel_type(self):
        self.dtype = "float32" if core.is_compiled_with_rocm() else "float64"

    def setUp(self):
        self.init_kernel_type()
        self.config()
        self.op_type = "unique_consecutive"
        x = np.random.randint(self.x_range, size=self.x_size).astype(self.dtype)
        result = reference_unique_consecutive(x, self.return_inverse,
                                              self.return_counts)
        out = reference_unique_consecutive(x)
        out = np.array(out).astype(self.dtype)
        self.inputs = {
            'X': x,
        }
        self.python_out_sig = ["Out"]
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        self.outputs = {
            'Out': out,
        }

    def test_check_output(self):
        self.check_output(check_eager=True)


class TestUniqueConsecutiveOp2(TestUniqueConsecutiveOp):
    """case 2"""

    def config(self):
        self.x_size = 100
        self.x_range = 20
        self.return_inverse = True
        self.return_counts = False
        self.python_api = paddle.unique_consecutive

    def setUp(self):
        self.init_kernel_type()
        self.config()
        self.op_type = "unique_consecutive"
        x = np.random.randint(self.x_range, size=self.x_size).astype(self.dtype)
        result, inverse = reference_unique_consecutive(x, self.return_inverse,
                                                       self.return_counts)
        result = np.array(result).astype(self.dtype)
        inverse = inverse.astype(self.dtype)
        self.inputs = {
            'X': x,
        }
        self.attrs = {
            'return_inverse': self.return_inverse,
            'dtype': int(core.VarDesc.VarType.INT32)
        }
        self.python_out_sig = ["Out"]
        self.outputs = {'Out': result, 'Index': inverse}


class TestUniqueConsecutiveOp3(TestUniqueConsecutiveOp):
    """case 3"""

    def config(self):
        self.x_size = 100
        self.x_range = 20
        self.return_inverse = False
        self.return_counts = True
        self.python_api = paddle.unique_consecutive

    def setUp(self):
        self.init_kernel_type()
        self.config()
        self.op_type = "unique_consecutive"
        x = np.random.randint(self.x_range, size=self.x_size).astype(self.dtype)
        result, counts = reference_unique_consecutive(x, self.return_inverse,
                                                      self.return_counts)
        result = np.array(result).astype(self.dtype)
        counts = counts.astype(self.dtype)
        self.inputs = {
            'X': x,
        }
        self.attrs = {
            'return_counts': self.return_counts,
            'dtype': int(core.VarDesc.VarType.INT32)
        }
        self.python_out_sig = ["Out"]
        self.outputs = {'Out': result, 'Counts': counts}


class TestUniqueConsecutiveOp4(TestUniqueConsecutiveOp):
    """case 4"""

    def config(self):
        self.x_size = 100
        self.x_range = 20
        self.return_inverse = True
        self.return_counts = True
        self.python_api = paddle.unique_consecutive

    def setUp(self):
        self.init_kernel_type()
        self.config()
        self.op_type = "unique_consecutive"
        x = np.random.randint(self.x_range, size=self.x_size).astype(self.dtype)
        result, inverse, counts = reference_unique_consecutive(
            x, self.return_inverse, self.return_counts)
        result = np.array(result).astype(self.dtype)
        inverse = inverse.astype(self.dtype)
        counts = counts.astype(self.dtype)
        self.inputs = {
            'X': x,
        }
        self.attrs = {
            'return_inverse': self.return_inverse,
            'return_counts': self.return_counts,
            'dtype': int(core.VarDesc.VarType.INT32)
        }
        self.python_out_sig = ["Out"]
        self.outputs = {'Out': result, 'Index': inverse, 'Counts': counts}


class TestUniqueConsecutiveAPI(unittest.TestCase):

    def setUp(self):
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def check_static_result(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            paddle.enable_static()
            input_x = fluid.data(name="input_x", shape=[
                100,
            ], dtype="float32")
            result = paddle.unique_consecutive(input_x)
            x_np = np.random.randint(20, size=100).astype("float32")
            exe = fluid.Executor(place)
            fetches = exe.run(fluid.default_main_program(),
                              feed={"input_x": x_np},
                              fetch_list=[result])

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with fluid.dygraph.guard(place):
                input_x = np.random.randint(20, size=100).astype("float64")
                x = paddle.to_tensor(input_x)
                result = paddle.unique_consecutive(x)


class TestUniqueConsecutiveCase2API(unittest.TestCase):

    def setUp(self):
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def check_static_result(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            paddle.enable_static()
            input_x = fluid.data(name="input_x", shape=[
                100,
            ], dtype="float32")
            result, inverse, counts = paddle.unique_consecutive(
                input_x, return_inverse=True, return_counts=True)
            x_np = np.random.randint(20, size=100).astype("float32")
            exe = fluid.Executor(place)
            fetches = exe.run(fluid.default_main_program(),
                              feed={"input_x": x_np},
                              fetch_list=[result])

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with fluid.dygraph.guard(place):
                input_x = np.random.randint(20, size=100).astype("float64")
                x = paddle.to_tensor(input_x)
                result, inverse, counts = paddle.unique_consecutive(
                    x, return_inverse=True, return_counts=True)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
