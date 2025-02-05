#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


def random_unique_float(shape, dtype):
    # create a random float array with 10x length
    numel = np.prod(shape)
    arr = np.random.uniform(-10.0, 10.0, numel * 10).astype(dtype)
    arr = np.unique(arr)
    assert (
        arr.shape[0] >= numel
    ), f"failed to create enough unique values: {arr.shape[0]} vs {numel}"
    arr = arr[:numel]
    np.random.shuffle(arr)
    arr = arr.reshape(shape)
    return arr


def numpy_topk(x, k=1, axis=-1, largest=True):
    if axis < 0:
        axis = len(x.shape) + axis
    if largest:
        indices = np.argsort(-x, axis=axis)
    else:
        indices = np.argsort(x, axis=axis)
    if largest:
        value = -np.sort(-x, axis=axis)
    else:
        value = np.sort(x, axis=axis)
    indices = indices.take(indices=range(0, k), axis=axis)
    value = value.take(indices=range(0, k), axis=axis)
    return value, indices


class XPUTestTopKV2Op(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'top_k_v2'
        self.use_dynamic_create_class = False

    class TestTopkOp(XPUOpTest):
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.op_type = "top_k_v2"
            self.dtype = self.in_type
            self.init_args()
            self.input_data = random_unique_float(
                self.input_data_shape, self.dtype
            )
            self.inputs = {'X': self.input_data}
            self.attrs = {
                'k': self.k,
                'axis': self.axis,
                'largest': self.largest,
            }
            output, indices = numpy_topk(
                self.input_data, axis=self.axis, k=self.k, largest=self.largest
            )
            self.outputs = {'Out': output, 'Indices': indices}

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

        def init_args(self):
            self.k = 3
            self.axis = 1
            self.largest = True
            self.input_data_shape = (10, 20)

    class TestTopkOp1(TestTopkOp):
        def init_args(self):
            self.k = 3
            self.axis = 1
            self.largest = True
            # too many values for fp16 will lead to failure in random_unique_float function
            if self.dtype == np.float16:
                self.input_data_shape = (100, 55)
            else:
                self.input_data_shape = (100, 155)

    class TestTopkOp2(TestTopkOp):
        def init_args(self):
            self.k = 3
            self.axis = 1
            self.largest = True
            self.input_data_shape = (10, 10, 5)

    class TestTopkOp3(TestTopkOp):
        def init_args(self):
            self.k = 5
            self.axis = 1
            self.largest = True
            self.input_data_shape = (10, 10, 5)

    class TestTopkOp4(TestTopkOp):
        def init_args(self):
            self.k = 1
            self.axis = 1
            self.largest = True
            self.input_data_shape = (10, 10, 5)

    class TestTopkOp5(TestTopkOp):
        def init_args(self):
            self.k = 3
            self.axis = 2
            self.largest = True
            self.input_data_shape = (10, 10, 5)

    class TestTopkOp6(TestTopkOp):
        def init_args(self):
            self.k = 5
            self.axis = 1
            self.largest = True
            # too many values for fp16 will lead to failure in random_unique_float function
            if self.dtype == np.float16:
                self.input_data_shape = (8, 32, 32)
            else:
                self.input_data_shape = (8, 32, 64)

    class TestTopkOp7(TestTopkOp):
        def init_args(self):
            self.k = 10
            self.axis = 2
            self.largest = True
            self.input_data_shape = (8, 5, 10, 16)

    class TestTopkOp8(TestTopkOp):
        def init_args(self):
            self.k = 1
            self.axis = 1
            self.largest = True
            # too many values for fp16 will lead to failure in random_unique_float function
            if self.dtype == np.float16:
                self.input_data_shape = (8, 32, 32)
            else:
                self.input_data_shape = (8, 32, 64)

    class TestTopkOp9(TestTopkOp):
        def init_args(self):
            self.k = 3
            self.axis = 1
            self.largest = True
            self.input_data_shape = (10, 10, 5)

    class TestTopkOp10(TestTopkOp):
        def init_args(self):
            self.k = 3
            self.axis = 1
            self.largest = True
            self.input_data_shape = (10, 10, 5)

    class TestTopkOp11(TestTopkOp):
        def init_args(self):
            self.k = 5
            self.axis = 1
            self.largest = True
            self.input_data_shape = (10, 10, 5)

    class TestTopkOp12(TestTopkOp):
        def init_args(self):
            self.k = 1
            self.axis = 1
            self.largest = True
            self.input_data_shape = (10, 10, 5)

    class TestTopkSmallestOp1(TestTopkOp):
        def init_args(self):
            self.k = 3
            self.axis = 1
            self.largest = False
            # too many values for fp16 will lead to failure in random_unique_float function
            if self.dtype == np.float16:
                self.input_data_shape = (100, 55)
            else:
                self.input_data_shape = (100, 155)

    class TestTopkSmallestOp2(TestTopkOp):
        def init_args(self):
            self.k = 3
            self.axis = 1
            self.largest = False
            self.input_data_shape = (10, 10, 5)

    class TestTopkSmallestOp3(TestTopkOp):
        def init_args(self):
            self.k = 5
            self.axis = 1
            self.largest = False
            self.input_data_shape = (10, 10, 5)

    class TestTopkSmallestOp4(TestTopkOp):
        def init_args(self):
            self.k = 1
            self.axis = 1
            self.largest = False
            self.input_data_shape = (10, 10, 5)

    class TestTopkSmallestOp5(TestTopkOp):
        def init_args(self):
            self.k = 3
            self.axis = 2
            self.largest = False
            self.input_data_shape = (10, 10, 5)

    class TestTopkSmallestOp6(TestTopkOp):
        def init_args(self):
            self.k = 5
            self.axis = 1
            self.largest = False
            # too many values for fp16 will lead to failure in random_unique_float function
            if self.dtype == np.float16:
                self.input_data_shape = (8, 32, 32)
            else:
                self.input_data_shape = (8, 32, 64)

    class TestTopkSmallestOp7(TestTopkOp):
        def init_args(self):
            self.k = 10
            self.axis = 2
            self.largest = False
            self.input_data_shape = (8, 5, 10, 16)

    class TestTopkSmallestOp8(TestTopkOp):
        def init_args(self):
            self.k = 1
            self.axis = 1
            self.largest = False
            # too many values for fp16 will lead to failure in random_unique_float function
            if self.dtype == np.float16:
                self.input_data_shape = (8, 32, 32)
            else:
                self.input_data_shape = (8, 32, 64)

    class TestTopkSmallestOp9(TestTopkOp):
        def init_args(self):
            self.k = 3
            self.axis = 1
            self.largest = False
            self.input_data_shape = (10, 10, 5)

    class TestTopkSmallestOp10(TestTopkOp):
        def init_args(self):
            self.k = 3
            self.axis = 1
            self.largest = False
            self.input_data_shape = (10, 10, 5)

    class TestTopkSmallestOp11(TestTopkOp):
        def init_args(self):
            self.k = 5
            self.axis = 1
            self.largest = False
            self.input_data_shape = (10, 10, 5)

    class TestTopkSmallestOp12(TestTopkOp):
        def init_args(self):
            self.k = 1
            self.axis = 1
            self.largest = False
            self.input_data_shape = (10, 10, 5)


support_types = get_xpu_op_support_types('top_k_v2')
for stype in support_types:
    create_test_class(globals(), XPUTestTopKV2Op, stype)

if __name__ == "__main__":
    unittest.main()
