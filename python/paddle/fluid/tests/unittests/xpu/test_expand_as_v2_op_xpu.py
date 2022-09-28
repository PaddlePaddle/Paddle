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
from op_test_xpu import XPUOpTest
import paddle
import paddle.fluid as fluid
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()
np.random.seed(10)


class XPUTestExpandAsV2Op(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'expand_as_v2'
        self.use_dynamic_create_class = False

    class TestExpandAsV2XPUOp(XPUOpTest):

        def setUp(self):
            self.init_dtype()
            self.set_xpu()
            self.op_type = "expand_as_v2"
            self.place = paddle.XPUPlace(0)
            self.set_inputs()
            self.set_output()

        def init_dtype(self):
            self.dtype = self.in_type

        def set_inputs(self):
            x = np.random.rand(100).astype(self.dtype)
            self.inputs = {'X': x}
            target_tensor = np.random.rand(2, 100).astype(self.dtype)
            self.attrs = {'target_shape': target_tensor.shape}

        def set_output(self):
            bcast_dims = [2, 1]
            output = np.tile(self.inputs['X'], bcast_dims)
            self.outputs = {'Out': output}

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True
            self.__class__.op_type = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)

    class TestExpandAsOpRank2(TestExpandAsV2XPUOp):

        def set_inputs(self):
            x = np.random.rand(10, 12).astype(self.dtype)
            self.inputs = {'X': x}
            target_tensor = np.random.rand(10, 12).astype(self.dtype)
            self.attrs = {'target_shape': target_tensor.shape}

        def set_output(self):
            bcast_dims = [1, 1]
            output = np.tile(self.inputs['X'], bcast_dims)
            self.outputs = {'Out': output}

    class TestExpandAsOpRank3(TestExpandAsV2XPUOp):

        def set_inputs(self):
            x = np.random.rand(2, 3, 20).astype(self.dtype)
            self.inputs = {'X': x}
            target_tensor = np.random.rand(2, 3, 20).astype(self.dtype)
            self.attrs = {'target_shape': target_tensor.shape}

        def set_output(self):
            bcast_dims = [1, 1, 1]
            output = np.tile(self.inputs['X'], bcast_dims)
            self.outputs = {'Out': output}

    class TestExpandAsOpRank4(TestExpandAsV2XPUOp):

        def set_inputs(self):
            x = np.random.rand(1, 1, 7, 16).astype(self.dtype)
            self.inputs = {'X': x}
            target_tensor = np.random.rand(1, 1, 7, 16).astype(self.dtype)
            self.attrs = {'target_shape': target_tensor.shape}

        def set_output(self):
            bcast_dims = [4, 6, 1, 1]
            output = np.tile(self.inputs['X'], bcast_dims)
            self.outputs = {'Out': output}

    class TestExpandAsOpRank5(TestExpandAsV2XPUOp):

        def set_inputs(self):
            x = np.random.rand(1, 1, 7, 16, 1).astype(self.dtype)
            self.inputs = {'X': x}
            target_tensor = np.random.rand(1, 1, 7, 16, 1).astype(self.dtype)
            self.attrs = {'target_shape': target_tensor.shape}

        def set_output(self):
            bcast_dims = [4, 6, 1, 1, 1]
            output = np.tile(self.inputs['X'], bcast_dims)
            self.outputs = {'Out': output}

    class TestExpandAsOpRank6(TestExpandAsV2XPUOp):

        def set_inputs(self):
            x = np.random.rand(1, 1, 7, 16, 1, 1).astype(self.dtype)
            self.inputs = {'X': x}
            target_tensor = np.random.rand(1, 1, 7, 16, 1, 1).astype(self.dtype)
            self.attrs = {'target_shape': target_tensor.shape}

        def set_output(self):
            bcast_dims = [4, 6, 1, 1, 1, 1]
            output = np.tile(self.inputs['X'], bcast_dims)
            self.outputs = {'Out': output}


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

        exe = fluid.Executor(place=fluid.XPUPlace(0))
        res_1 = exe.run(fluid.default_main_program(),
                        feed={
                            "x": input1,
                            "target_tensor": input2
                        },
                        fetch_list=[out_1])
        assert np.array_equal(res_1[0], np.tile(input1, (2, 1, 1)))


support_types = get_xpu_op_support_types('expand_as_v2')
for stype in support_types:
    create_test_class(globals(), XPUTestExpandAsV2Op, stype)

if __name__ == '__main__':
    unittest.main()
