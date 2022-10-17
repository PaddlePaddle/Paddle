#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
import sys

sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid

paddle.enable_static()
np.random.seed(10)


class TestExpandAsOpRank1(OpTest):

    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "expand_as_v2"
        x = np.random.rand(100).astype("float32")
        target_tensor = np.random.rand(2, 100).astype("float32")
        self.inputs = {'X': x}
        self.attrs = {'target_shape': target_tensor.shape}
        bcast_dims = [2, 1]
        output = np.tile(self.inputs['X'], bcast_dims)
        self.outputs = {'Out': output}

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass


class TestExpandAsOpRank2(OpTest):

    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "expand_as_v2"
        x = np.random.rand(10, 12).astype("float32")
        target_tensor = np.random.rand(10, 12).astype("float32")
        self.inputs = {'X': x}
        self.attrs = {'target_shape': target_tensor.shape}
        bcast_dims = [1, 1]
        output = np.tile(self.inputs['X'], bcast_dims)
        self.outputs = {'Out': output}

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass


class TestExpandAsOpRank3(OpTest):

    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "expand_as_v2"
        x = np.random.rand(2, 3, 20).astype("float32")
        target_tensor = np.random.rand(2, 3, 20).astype("float32")
        self.inputs = {'X': x}
        self.attrs = {'target_shape': target_tensor.shape}
        bcast_dims = [1, 1, 1]
        output = np.tile(self.inputs['X'], bcast_dims)
        self.outputs = {'Out': output}

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass


class TestExpandAsOpRank4(OpTest):

    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "expand_as_v2"
        x = np.random.rand(1, 1, 7, 16).astype("float32")
        target_tensor = np.random.rand(4, 6, 7, 16).astype("float32")
        self.inputs = {'X': x}
        self.attrs = {'target_shape': target_tensor.shape}
        bcast_dims = [4, 6, 1, 1]
        output = np.tile(self.inputs['X'], bcast_dims)
        self.outputs = {'Out': output}

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass


# Test python API
class TestExpandAsV2API(unittest.TestCase):

    def test_api(self):
        input1 = np.random.random([12, 14]).astype("float32")
        input2 = np.random.random([2, 12, 14]).astype("float32")
        x = fluid.layers.data(name='x',
                              shape=[12, 14],
                              append_batch_size=False,
                              dtype="float32")

        y = fluid.layers.data(name='target_tensor',
                              shape=[2, 12, 14],
                              append_batch_size=False,
                              dtype="float32")

        out_1 = paddle.expand_as(x, y=y)

        exe = fluid.Executor(place=fluid.NPUPlace(0))
        res_1 = exe.run(fluid.default_main_program(),
                        feed={
                            "x": input1,
                            "target_tensor": input2
                        },
                        fetch_list=[out_1])
        assert np.array_equal(res_1[0], np.tile(input1, (2, 1, 1)))


if __name__ == '__main__':
    unittest.main()
