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
import sys

sys.path.append('..')
import numpy as np
from op_test import OpTest
import paddle.fluid as fluid
import paddle

paddle.enable_static()


def test_class1(op_type, typename):

    class TestExpandAsBasic(OpTest):

        def setUp(self):
            self.set_mlu()
            self.op_type = "expand_as_v2"
            self.python_api = paddle.expand_as
            x = np.random.rand(100).astype(typename)
            target_tensor = np.random.rand(2, 100).astype(typename)
            self.inputs = {'X': x}
            self.attrs = {'target_shape': target_tensor.shape}
            bcast_dims = [2, 1]
            output = np.tile(self.inputs['X'], bcast_dims)
            self.outputs = {'Out': output}

        def set_mlu(self):
            self.__class__.use_mlu = True
            self.place = paddle.device.MLUPlace(0)
            self.__class__.no_need_check_grad = True

        def test_check_output(self):
            self.check_output_with_place(self.place)

    cls_name = str(op_type) + "_" + str(typename) + "_1"
    TestExpandAsBasic.__name__ = cls_name
    globals()[cls_name] = TestExpandAsBasic


def test_class2(op_type, typename):

    class TestExpandAsOpRank2(OpTest):

        def setUp(self):
            self.set_mlu()
            self.op_type = "expand_as_v2"
            self.python_api = paddle.expand_as
            x = np.random.rand(10, 12).astype(typename)
            target_tensor = np.random.rand(10, 12).astype(typename)
            self.inputs = {'X': x}
            self.attrs = {'target_shape': target_tensor.shape}
            bcast_dims = [1, 1]
            output = np.tile(self.inputs['X'], bcast_dims)
            self.outputs = {'Out': output}

        def set_mlu(self):
            self.__class__.use_mlu = True
            self.place = paddle.device.MLUPlace(0)
            self.__class__.no_need_check_grad = True

        def test_check_output(self):
            self.check_output_with_place(self.place)

    cls_name = str(op_type) + "_" + str(typename) + "_2"
    TestExpandAsOpRank2.__name__ = cls_name
    globals()[cls_name] = TestExpandAsOpRank2


def test_class3(op_type, typename):

    class TestExpandAsOpRank3(OpTest):

        def setUp(self):
            self.set_mlu()
            self.op_type = "expand_as_v2"
            self.python_api = paddle.expand_as
            x = np.random.rand(2, 3, 20).astype(typename)
            target_tensor = np.random.rand(2, 3, 20).astype(typename)
            self.inputs = {'X': x}
            self.attrs = {'target_shape': target_tensor.shape}
            bcast_dims = [1, 1, 1]
            output = np.tile(self.inputs['X'], bcast_dims)
            self.outputs = {'Out': output}

        def set_mlu(self):
            self.__class__.use_mlu = True
            self.place = paddle.device.MLUPlace(0)
            self.__class__.no_need_check_grad = True

        def test_check_output(self):
            self.check_output_with_place(self.place)

    cls_name = str(op_type) + "_" + str(typename) + "_3"
    TestExpandAsOpRank3.__name__ = cls_name
    globals()[cls_name] = TestExpandAsOpRank3


def test_class4(op_type, typename):

    class TestExpandAsOpRank4(OpTest):

        def setUp(self):
            self.set_mlu()
            self.op_type = "expand_as_v2"
            self.python_api = paddle.expand_as
            x = np.random.rand(1, 1, 7, 16).astype(typename)
            target_tensor = np.random.rand(4, 6, 7, 16).astype(typename)
            self.inputs = {'X': x}
            self.attrs = {'target_shape': target_tensor.shape}
            bcast_dims = [4, 6, 1, 1]
            output = np.tile(self.inputs['X'], bcast_dims)
            self.outputs = {'Out': output}

        def set_mlu(self):
            self.__class__.use_mlu = True
            self.place = paddle.device.MLUPlace(0)
            self.__class__.no_need_check_grad = True

        def test_check_output(self):
            self.check_output_with_place(self.place)

    cls_name = str(op_type) + "_" + str(typename) + "_4"
    TestExpandAsOpRank4.__name__ = cls_name
    globals()[cls_name] = TestExpandAsOpRank4


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

        exe = fluid.Executor(place=fluid.MLUPlace(0))
        res_1 = exe.run(fluid.default_main_program(),
                        feed={
                            "x": input1,
                            "target_tensor": input2
                        },
                        fetch_list=[out_1])
        assert np.array_equal(res_1[0], np.tile(input1, (2, 1, 1)))


for _typename in {
        'float16', 'float32', 'int64', 'int32', 'int8', 'uint8', 'bool'
}:
    test_class1('expand_as_v2', _typename)
    test_class2('expand_as_v2', _typename)
    test_class3('expand_as_v2', _typename)
    test_class4('expand_as_v2', _typename)

if __name__ == "__main__":
    unittest.main()
