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

import sys

sys.path.append("..")
import unittest
import numpy as np
import os
import paddle
import paddle.fluid as fluid
from op_test import OpTest
import paddle.fluid.core as core
from paddle.fluid.dygraph.base import switch_to_static_graph

paddle.enable_static()


class TestScatterOp(OpTest):

    def setUp(self):
        self.op_type = "scatter"
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.python_api = paddle.scatter
        ref_np = np.ones((3, 50)).astype("float32")
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 50)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output_with_place(self.place, check_eager=False)

    def test_check_grad(self):
        self.check_grad(["X", "Updates"], "Out", check_eager=False)


class TestScatterOp0(OpTest):

    def setUp(self):
        self.op_type = "scatter"
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.python_api = paddle.scatter
        ref_np = np.ones((3, 3)).astype("float32")
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.attrs = {'overwrite': True}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output_with_place(self.place, check_eager=False)

    def test_check_grad(self):
        self.check_grad(["X", "Updates"], "Out", check_eager=False)


class TestScatterOp1(OpTest):

    def setUp(self):
        self.op_type = "scatter"
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.python_api = paddle.scatter
        ref_np = np.ones((3, 3)).astype("float32")
        zeros_np = np.zeros([2, 3]).astype('float32')
        index_np = np.array([1, 1]).astype("int32")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = zeros_np
        for i in range(0, len(index_np)):
            output_np[index_np[i]] += updates_np[i]
        self.attrs = {'overwrite': False}
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output_with_place(self.place, check_eager=False)

    def test_check_grad(self):
        self.check_grad(["X", "Updates"], "Out", check_eager=False)


class TestScatterOp2(OpTest):

    def setUp(self):
        self.op_type = "scatter"
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.python_api = paddle.scatter
        ref_np = np.ones((3, 3)).astype("float32")
        index_np = np.array([1, 2]).astype("int64")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output_with_place(self.place, check_eager=False)

    def test_check_grad(self):
        self.check_grad(['X', 'Updates'], 'Out', check_eager=False)


class TestScatterAPI(unittest.TestCase):

    def setUp(self):
        self.places = [paddle.device.MLUPlace(0)]
        self.__class__.use_mlu = True
        self.executed_api()

    def executed_api(self):
        self.scatter = paddle.scatter

    def check_static_result(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input = fluid.data(name="input", shape=[3, 2], dtype="float32")
            index = fluid.data(name="index", shape=[4], dtype="int64")
            updates = fluid.data(name="updates", shape=[4, 2], dtype="float32")
            result = self.scatter(input, index, updates, False)

            input_data = np.array([[1, 1], [2, 2], [3, 3]]).astype(np.float32)
            index_data = np.array([2, 1, 0, 1]).astype(np.int64)
            updates_data = np.array([[1, 1], [2, 2], [3, 3],
                                     [4, 4]]).astype(np.float32)

            exe = fluid.Executor(place)
            fetches = exe.run(fluid.default_main_program(),
                              feed={
                                  "input": input_data,
                                  "index": index_data,
                                  "updates": updates_data
                              },
                              fetch_list=[result])
            self.assertEqual((fetches[0] == \
                              np.array([[3., 3.],[6., 6.],[1., 1.]])).all(), True)

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with fluid.dygraph.guard(place):
                x_data = np.array([[1, 1], [2, 2], [3, 3]]).astype(np.float32)
                index_data = np.array([2, 1, 0, 1]).astype(np.int64)
                updates_data = np.array([[1, 1], [2, 2], [3, 3],
                                         [4, 4]]).astype(np.float32)

                x = fluid.dygraph.to_variable(x_data)
                index = fluid.dygraph.to_variable(index_data)
                updates = fluid.dygraph.to_variable(updates_data)

                output1 = self.scatter(x, index, updates, overwrite=False)
                self.assertEqual((output1.numpy() == \
                                  np.array([[3., 3.],[6., 6.],[1., 1.]])).all(), True)

    def test_large_data(self):
        if os.name == "nt":
            return

        x = np.random.rand(183826, 256).astype("float32")
        index = np.ones(8388608, dtype="int64")
        updates = np.ones(shape=[8388608, 256], dtype="float32")

        def test_dygraph():
            with fluid.dygraph.guard():
                mlu_out = paddle.scatter(paddle.to_tensor(x),
                                         paddle.to_tensor(index),
                                         paddle.to_tensor(updates))
                return mlu_out.numpy()

        @switch_to_static_graph
        def test_static_graph():
            with paddle.static.program_guard(paddle.static.Program(),
                                             paddle.static.Program()):
                x_t = paddle.static.data(name="x", dtype=x.dtype, shape=x.shape)
                index_t = paddle.static.data(name="index",
                                             dtype=index.dtype,
                                             shape=index.shape)
                updates_t = paddle.static.data(name="updates",
                                               dtype=updates.dtype,
                                               shape=updates.shape)
                out_t = paddle.scatter(x_t, index_t, updates_t)
                feed = {
                    x_t.name: x,
                    index_t.name: index,
                    updates_t.name: updates
                }
                fetch = [out_t]

                mlu_exe = paddle.static.Executor(paddle.device.MLUPlace(0))
                mlu_value = mlu_exe.run(feed=feed, fetch_list=fetch)[0]
                return mlu_value

        np.testing.assert_allclose(test_dygraph(), test_static_graph())


class TestScatterOpFp16(OpTest):

    def setUp(self):
        self.op_type = "scatter"
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.python_api = paddle.scatter
        ref_np = np.ones((3, 3)).astype("float16")
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 3)).astype("float16")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.attrs = {'overwrite': True}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output_with_place(self.place, check_eager=False)

    def test_check_grad(self):
        self.check_grad(["X", "Updates"], "Out", check_eager=False)


class TestScatterInplaceAPI(TestScatterAPI):

    def executed_api(self):
        self.scatter = paddle.scatter_


if __name__ == "__main__":
    unittest.main()
