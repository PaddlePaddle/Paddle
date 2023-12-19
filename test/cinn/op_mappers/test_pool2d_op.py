# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

from cinn.common import is_compiled_with_cudnn
from op_mapper_test import OpMapperTest

import paddle


@unittest.skipIf(
    not is_compiled_with_cudnn(), "x86 test will be skipped due to timeout."
)
class TestPool2dOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {"x": self.random([2, 3, 7, 7], "float64")}
        self.data_format = "NCHW"
        self.pooling_type = "avg"
        self.kernel_size = [3, 3]
        self.stride = [1, 1]
        self.padding = [0, 0]
        self.padding_algorithm = "EXPLICIT"
        self.global_pooling = False
        self.ceil_mode = False
        self.exclusive = False
        self.adaptive = False
        self.use_cudnn = True

    def set_op_type(self):
        return "pool2d"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype,
        )
        return {'X': [x]}

    def set_op_attrs(self):
        return {
            "pooling_type": self.pooling_type,
            "ksize": self.kernel_size,
            "global_pooling": self.global_pooling,
            "strides": self.stride,
            "paddings": self.padding,
            "exclusive": self.exclusive,
            "adaptive": self.adaptive,
            "ceil_mode": self.ceil_mode,
            "data_format": self.data_format,
            "padding_algorithm": self.padding_algorithm,
            "use_cudnn": self.use_cudnn,
        }

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestPool2dOpCase1(TestPool2dOp):
    def init_input_data(self):
        super().init_input_data()
        self.pooling_type = "max"


class TestPool2dOpCase2(TestPool2dOp):
    def init_input_data(self):
        super().init_input_data()
        self.kernel_size = [5, 5]
        self.stride = [2, 2]
        self.padding = [1, 1]


class TestPool2dOpCase3(TestPool2dOp):
    def init_input_data(self):
        super().init_input_data()
        self.global_pooling = True


class TestPool2dOpCase4(TestPool2dOp):
    def init_input_data(self):
        super().init_input_data()
        self.ceil_mode = True


class TestPool2dOpCase5(TestPool2dOp):
    def init_input_data(self):
        super().init_input_data()
        self.exclusive = True


class TestPool2dOpCase6(TestPool2dOp):
    def init_input_data(self):
        super().init_input_data()
        self.kernel_size = [2, 2]
        self.stride = [1, 1]
        self.padding = [0, 0]
        self.adaptive = True


class TestPool2dOpCase7(TestPool2dOp):
    def init_input_data(self):
        super().init_input_data()
        self.padding_algorithm = "VALID"


class TestPool2dOpCase8(TestPool2dOp):
    def init_input_data(self):
        super().init_input_data()
        self.padding_algorithm = "SAME"


class TestPool2dOpCase9(TestPool2dOp):
    def init_input_data(self):
        super().init_input_data()
        self.feed_data = {"x": self.random([2, 7, 7, 3], "float64")}
        self.data_format = "NHWC"


if __name__ == "__main__":
    unittest.main()
