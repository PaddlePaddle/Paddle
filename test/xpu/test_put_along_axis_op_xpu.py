#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import unittest

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


class XPUTestPutAlongAxis(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'put_along_axis'

    class TestXPUPutAlongAxisOpAssign(XPUOpTest):
        def setUp(self):

            self.init_config()
            self.init_data()
            self.x = np.random.random(self.x_shape).astype(
                self.dtype if self.dtype != np.uint16 else np.float32
            )
            self.value = np.random.random(self.index.shape).astype(
                self.dtype if self.dtype != np.uint16 else np.float32
            )
            broadcast_shape_list = list(self.x_shape)

            self.broadcast_shape = tuple(broadcast_shape_list)
            self.index_broadcast = np.broadcast_to(
                self.index, self.broadcast_shape
            )
            self.value_broadcast = np.broadcast_to(
                self.value, self.broadcast_shape
            )
            self.target = copy.deepcopy(self.x)
            mean_record = {}
            for i in range(self.index_broadcast.shape[0]):
                for j in range(self.index_broadcast.shape[1]):
                    for k in range(self.index_broadcast.shape[2]):
                        loc_ = [i, j, k]
                        loc_[self.axis] = self.index_broadcast[i, j, k]
                        if self.reduce == "assign":
                            self.target[loc_[0], loc_[1], loc_[2]] = (
                                self.value_broadcast[i, j, k]
                            )
                        elif self.reduce == "add":
                            self.target[
                                loc_[0], loc_[1], loc_[2]
                            ] += self.value_broadcast[i, j, k]
                        elif self.reduce == "mul" or self.reduce == "multiply":
                            self.target[
                                loc_[0], loc_[1], loc_[2]
                            ] *= self.value_broadcast[i, j, k]
                        elif self.reduce == "mean":
                            self.target[
                                loc_[0], loc_[1], loc_[2]
                            ] += self.value_broadcast[i, j, k]
                            loc = tuple(loc_)
                            if loc in mean_record.keys():
                                mean_record[loc] += 1
                            else:
                                mean_record[loc] = 1
                        elif self.reduce == "amax":
                            self.target[loc_[0], loc_[1], loc_[2]] = max(
                                self.target[loc_[0], loc_[1], loc_[2]],
                                self.value_broadcast[i, j, k],
                            )
                        elif self.reduce == "amin":
                            self.target[loc_[0], loc_[1], loc_[2]] = min(
                                self.target[loc_[0], loc_[1], loc_[2]],
                                self.value_broadcast[i, j, k],
                            )
                        elif self.reduce == "max":
                            self.target[loc_[0], loc_[1], loc_[2]] = max(
                                self.target[loc_[0], loc_[1], loc_[2]],
                                self.value_broadcast[i, j, k],
                            )
            if self.reduce == "mean":
                for loc in mean_record:
                    self.target[loc] /= mean_record[loc] + 1

            self.inputs = {
                'Input': (
                    self.x
                    if self.dtype != np.uint16
                    else convert_float_to_uint16(self.x)
                ),
                'Index': self.index_broadcast,
                'Value': (
                    self.value_broadcast
                    if self.dtype != np.uint16
                    else convert_float_to_uint16(self.value_broadcast)
                ),
            }
            self.attrs = {
                'Axis': self.axis,
                'Reduce': self.reduce,
                'Include_self': True,
            }
            self.outputs = {
                'Result': (
                    self.target
                    if self.dtype != np.uint16
                    else convert_float_to_uint16(self.target)
                )
            }

        def init_config(self):
            self.op_type = "put_along_axis"
            self.place = paddle.XPUPlace(0)
            self.dtype = self.in_type

        def init_data(self):
            self.x_shape = (10, 10, 10)
            self.reduce = "assign"
            self.index_type = np.int32
            self.index = np.array([[[0]]]).astype(self.index_type)
            self.axis = 1

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                self.check_output_with_place(self.place)

        def test_check_grad(self):
            if paddle.is_compiled_with_xpu():
                self.check_grad_with_place(
                    self.place, ['Input', 'Value'], 'Result'
                )

    class TestAddCase1(TestXPUPutAlongAxisOpAssign):
        def init_data(self):
            self.in_type = self.dtype
            self.reduce = "add"
            self.x_shape = (10, 10, 10)
            self.index_type = np.int64
            self.index = np.array([[[0]]]).astype(self.index_type)
            self.axis = 1

    class TestAddCase2(TestXPUPutAlongAxisOpAssign):
        def init_data(self):
            self.in_type = self.dtype
            self.reduce = "add"
            self.x_shape = (12, 14, 16)
            self.index_type = np.int64
            self.index = np.random.randint(0, 12, size=(12, 14, 1))
            self.axis = 2

    class TestMulCase1(TestXPUPutAlongAxisOpAssign):
        def init_data(self):
            self.in_type = self.dtype
            self.reduce = "mul"
            self.x_shape = (16, 6, 12)
            self.index_type = np.int32
            self.index = np.random.randint(0, 6, size=(16, 1, 12))
            self.axis = 1

    class TestMulCase2(TestXPUPutAlongAxisOpAssign):
        def init_data(self):
            self.in_type = self.dtype
            self.reduce = "mul"
            self.x_shape = (8, 6, 12)
            self.index_type = np.int32
            self.index = np.random.randint(0, 6, size=(8, 1, 12))
            self.axis = 2

    class TestMeanCase1(TestXPUPutAlongAxisOpAssign):
        def init_data(self):
            self.in_type = self.dtype
            self.reduce = "mean"
            self.x_shape = (2, 2, 2)
            self.index_type = np.int64
            self.index = np.array([[[0, 0], [1, 0]]]).astype('int64')
            self.axis = 1

    class TestMeanCase2(TestXPUPutAlongAxisOpAssign):
        def init_data(self):
            self.in_type = self.dtype
            self.reduce = "mean"
            self.x_shape = (6, 8, 10)
            self.index_type = np.int32
            self.index = np.random.randint(0, 6, size=(1, 8, 10)).astype(
                self.index_type
            )
            self.axis = 0

    class TestMaxCase1(TestXPUPutAlongAxisOpAssign):
        def init_data(self):
            self.in_type = self.dtype
            self.reduce = "amax"
            self.x_shape = (12, 13, 10)
            self.index_type = np.int32
            self.index = np.array([[[7]]]).astype(self.index_type)
            self.index = np.random.randint(0, 12, size=(12, 13, 1))
            self.axis = 1

    class TestMaxCase2(TestXPUPutAlongAxisOpAssign):
        def init_data(self):
            self.in_type = self.dtype
            self.reduce = "amax"
            self.x_shape = (16, 13, 10)
            self.index_type = np.int32
            self.index = np.array([[[7]]]).astype(self.index_type)
            self.index = np.random.randint(0, 10, size=(1, 13, 1))
            self.axis = 2

    class TestMinCase1(TestXPUPutAlongAxisOpAssign):
        def init_data(self):
            self.in_type = self.dtype
            self.reduce = "amin"
            self.x_shape = (5, 5, 5)
            self.index_type = np.int64
            self.index = np.zeros((5, 5, 1)).astype(self.index_type)
            self.axis = 1

    class TestMinCase2(TestXPUPutAlongAxisOpAssign):
        def init_data(self):
            self.in_type = self.dtype
            self.reduce = "amin"
            self.x_shape = (12, 13, 14)
            self.index_type = np.int32
            self.index = np.random.randint(0, 12, size=(12, 13, 1))
            self.axis = 2


support_types = get_xpu_op_support_types('put_along_axis')
for stype in support_types:
    create_test_class(globals(), XPUTestPutAlongAxis, stype)

if __name__ == "__main__":
    unittest.main()
