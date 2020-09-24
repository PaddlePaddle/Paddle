#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
from op_test import OpTest
import paddle.fluid.core as core


class TestScatterOp(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        ref_np = np.ones((3, 50)).astype("float32")
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 50)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Updates'], 'Out', in_place=True)


class TestScatterOp0(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        ref_np = np.ones((3, 3)).astype("float32")
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.attrs = {'overwrite': True}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Updates'], 'Out', in_place=True)


class TestScatterOp1(OpTest):
    def setUp(self):
        self.op_type = "scatter"
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
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Updates'], 'Out', in_place=True)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestScatterOp2(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        ref_np = np.ones((3, 3)).astype("float32")
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-3)

    def test_check_grad(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(place, ['Updates'], 'Out', in_place=True)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestScatterOp3(OpTest):
    def setUp(self):
        self.op_type = "scatter"
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
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-3)

    def test_check_grad(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(place, ['Updates'], 'Out', in_place=True)


class TestScatterOp4(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        ref_np = np.ones((3, 3)).astype("float32")
        index_np = np.array([1, 2]).astype("int64")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Updates'], 'Out', in_place=True)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestScatterOp5(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        ref_np = np.ones((3, 3)).astype("float32")
        index_np = np.array([1, 2]).astype("int64")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-3)

    def test_check_grad(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(place, ['Updates'], 'Out', in_place=True)


class TestScatterAPI(unittest.TestCase):
    def setUp(self):
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def check_static_result(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input = fluid.data(name="input", shape=[3, 2], dtype="float64")
            index = fluid.data(name="index", shape=[4], dtype="int64")
            updates = fluid.data(name="updates", shape=[4, 2], dtype="float64")
            result = paddle.scatter(input, index, updates, False)

            input_data = np.array([[1, 1], [2, 2], [3, 3]]).astype(np.float64)
            index_data = np.array([2, 1, 0, 1]).astype(np.int64)
            updates_data = np.array(
                [[1, 1], [2, 2], [3, 3], [4, 4]]).astype(np.float64)

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
                x_data = np.array([[1, 1], [2, 2], [3, 3]]).astype(np.float64)
                index_data = np.array([2, 1, 0, 1]).astype(np.int64)
                updates_data = np.array(
                    [[1, 1], [2, 2], [3, 3], [4, 4]]).astype(np.float64)

                x = fluid.dygraph.to_variable(x_data)
                index = fluid.dygraph.to_variable(index_data)
                updates = fluid.dygraph.to_variable(updates_data)

                output1 = paddle.scatter(x, index, updates, overwrite=False)
                self.assertEqual((output1.numpy() == \
                                  np.array([[3., 3.],[6., 6.],[1., 1.]])).all(), True)


if __name__ == "__main__":
    unittest.main()
