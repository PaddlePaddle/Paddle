#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
import paddle.fluid as fluid
import paddle
from paddle.fluid.dygraph.base import switch_to_static_graph


def numpy_scatter_nd(ref, index, updates, fun):
    ref_shape = ref.shape
    index_shape = index.shape

    end_size = index_shape[-1]
    remain_numl = 1
    for i in range(len(index_shape) - 1):
        remain_numl *= index_shape[i]

    slice_size = 1
    for i in range(end_size, len(ref_shape)):
        slice_size *= ref_shape[i]

    flat_index = index.reshape([remain_numl] + list(index_shape[-1:]))
    flat_updates = updates.reshape((remain_numl, slice_size))
    flat_output = ref.reshape(list(ref_shape[:end_size]) + [slice_size])

    for i_up, i_out in enumerate(flat_index):
        i_out = tuple(i_out)
        flat_output[i_out] = fun(flat_output[i_out], flat_updates[i_up])
    return flat_output.reshape(ref.shape)


def numpy_scatter_nd_add(ref, index, updates):
    return numpy_scatter_nd(ref, index, updates, lambda x, y: x + y)


def judge_update_shape(ref, index):
    ref_shape = ref.shape
    index_shape = index.shape
    update_shape = []
    for i in range(len(index_shape) - 1):
        update_shape.append(index_shape[i])
    for i in range(index_shape[-1], len(ref_shape), 1):
        update_shape.append(ref_shape[i])
    return update_shape


class TestScatterNdAddSimpleOp(OpTest):
    """
    A simple example
    """

    def setUp(self):
        self.op_type = "scatter_nd_add"
        self.python_api = paddle.scatter_nd_add
        ref_np = np.random.random([100]).astype("float64")
        index_np = np.random.randint(0, 100, [100, 1]).astype("int32")
        updates_np = np.random.random([100]).astype("float64")
        expect_np = numpy_scatter_nd_add(ref_np.copy(), index_np, updates_np)

        self.inputs = {'X': ref_np, 'Index': index_np, 'Updates': updates_np}
        self.outputs = {'Out': expect_np}

    def test_check_output(self):
        self.check_output(check_eager=False)

    def test_check_grad(self):
        self.check_grad(['X', 'Updates'], 'Out', check_eager=False)


class TestScatterNdAddWithEmptyIndex(OpTest):
    """
    Index has empty element
    """

    def setUp(self):
        self.op_type = "scatter_nd_add"
        self.python_api = paddle.scatter_nd_add
        ref_np = np.random.random((10, 10)).astype("float64")
        index_np = np.array([[], []]).astype("int32")
        updates_np = np.random.random((2, 10, 10)).astype("float64")

        expect_np = numpy_scatter_nd_add(ref_np.copy(), index_np, updates_np)

        self.inputs = {'X': ref_np, 'Index': index_np, 'Updates': updates_np}
        self.outputs = {'Out': expect_np}

    def test_check_output(self):
        self.check_output(check_eager=False)

    def test_check_grad(self):
        self.check_grad(['X', 'Updates'], 'Out', check_eager=False)


class TestScatterNdAddWithHighRankSame(OpTest):
    """
    Both Index and X have high rank, and Rank(Index) = Rank(X)
    """

    def setUp(self):
        self.op_type = "scatter_nd_add"
        self.python_api = paddle.scatter_nd_add
        shape = (3, 2, 2, 1, 10)
        ref_np = np.random.rand(*shape).astype("float64")
        index_np = np.vstack(
            [np.random.randint(
                0, s, size=100) for s in shape]).T.astype("int32")
        update_shape = judge_update_shape(ref_np, index_np)
        updates_np = np.random.rand(*update_shape).astype("float64")
        expect_np = numpy_scatter_nd_add(ref_np.copy(), index_np, updates_np)

        self.inputs = {'X': ref_np, 'Index': index_np, 'Updates': updates_np}
        self.outputs = {'Out': expect_np}

    def test_check_output(self):
        self.check_output(check_eager=False)

    def test_check_grad(self):
        self.check_grad(['X', 'Updates'], 'Out', check_eager=False)


class TestScatterNdAddWithHighRankDiff(OpTest):
    """
    Both Index and X have high rank, and Rank(Index) < Rank(X)
    """

    def setUp(self):
        self.op_type = "scatter_nd_add"
        self.python_api = paddle.scatter_nd_add
        shape = (8, 2, 2, 1, 10)
        ref_np = np.random.rand(*shape).astype("double")
        index = np.vstack([np.random.randint(0, s, size=500) for s in shape]).T
        index_np = index.reshape([10, 5, 10, 5]).astype("int64")
        update_shape = judge_update_shape(ref_np, index_np)
        updates_np = np.random.rand(*update_shape).astype("double")
        expect_np = numpy_scatter_nd_add(ref_np.copy(), index_np, updates_np)

        self.inputs = {'X': ref_np, 'Index': index_np, 'Updates': updates_np}
        self.outputs = {'Out': expect_np}

    def test_check_output(self):
        self.check_output(check_eager=False)

    def test_check_grad(self):
        self.check_grad(['X', 'Updates'], 'Out', check_eager=False)


#Test Python API
class TestScatterNdOpAPI(unittest.TestCase):
    """
    test scatter_nd_add api and scatter_nd api
    """

    def testcase1(self):
        ref1 = fluid.layers.data(
            name='ref1',
            shape=[10, 9, 8, 1, 3],
            dtype='float32',
            append_batch_size=False)
        index1 = fluid.layers.data(
            name='index1',
            shape=[5, 5, 8, 5],
            dtype='int32',
            append_batch_size=False)
        updates1 = fluid.layers.data(
            name='update1',
            shape=[5, 5, 8],
            dtype='float32',
            append_batch_size=False)
        output1 = fluid.layers.scatter_nd_add(ref1, index1, updates1)

    def testcase2(self):
        ref2 = fluid.layers.data(
            name='ref2',
            shape=[10, 9, 8, 1, 3],
            dtype='double',
            append_batch_size=False)
        index2 = fluid.layers.data(
            name='index2',
            shape=[5, 8, 5],
            dtype='int32',
            append_batch_size=False)
        updates2 = fluid.layers.data(
            name='update2',
            shape=[5, 8],
            dtype='double',
            append_batch_size=False)
        output2 = fluid.layers.scatter_nd_add(
            ref2, index2, updates2, name="scatter_nd_add")

    def testcase3(self):
        shape3 = [10, 9, 8, 1, 3]
        index3 = fluid.layers.data(
            name='index3',
            shape=[5, 5, 8, 5],
            dtype='int32',
            append_batch_size=False)
        updates3 = fluid.layers.data(
            name='update3',
            shape=[5, 5, 8],
            dtype='float32',
            append_batch_size=False)
        output3 = fluid.layers.scatter_nd(index3, updates3, shape3)

    def testcase4(self):
        shape4 = [10, 9, 8, 1, 3]
        index4 = fluid.layers.data(
            name='index4',
            shape=[5, 5, 8, 5],
            dtype='int32',
            append_batch_size=False)
        updates4 = fluid.layers.data(
            name='update4',
            shape=[5, 5, 8],
            dtype='double',
            append_batch_size=False)
        output4 = fluid.layers.scatter_nd(
            index4, updates4, shape4, name='scatter_nd')

    def testcase5(self):
        if not fluid.core.is_compiled_with_cuda():
            return

        shape = [2, 3, 4]
        x = np.arange(int(np.prod(shape))).reshape(shape)
        index = np.array([[0, 0, 2], [0, 1, 2]])
        val = np.array([-1, -3])

        with fluid.dygraph.guard():
            device = paddle.get_device()
            paddle.set_device('gpu')
            gpu_value = paddle.scatter_nd_add(
                paddle.to_tensor(x),
                paddle.to_tensor(index), paddle.to_tensor(val))
            paddle.set_device('cpu')
            cpu_value = paddle.scatter_nd_add(
                paddle.to_tensor(x),
                paddle.to_tensor(index), paddle.to_tensor(val))
            self.assertTrue(
                np.array_equal(gpu_value.numpy(), cpu_value.numpy()))
            paddle.set_device(device)

        @switch_to_static_graph
        def test_static_graph():
            with paddle.static.program_guard(paddle.static.Program(),
                                             paddle.static.Program()):
                x_t = paddle.static.data(name="x", dtype=x.dtype, shape=x.shape)
                index_t = paddle.static.data(
                    name="index", dtype=index.dtype, shape=index.shape)
                val_t = paddle.static.data(
                    name="val", dtype=val.dtype, shape=val.shape)
                out_t = paddle.scatter_nd_add(x_t, index_t, val_t)
                feed = {x_t.name: x, index_t.name: index, val_t.name: val}
                fetch = [out_t]

                gpu_exe = paddle.static.Executor(paddle.CUDAPlace(0))
                gpu_value = gpu_exe.run(feed=feed, fetch_list=fetch)[0]
                cpu_exe = paddle.static.Executor(paddle.CPUPlace())
                cpu_value = cpu_exe.run(feed=feed, fetch_list=fetch)[0]
                self.assertTrue(np.array_equal(gpu_value, cpu_value))

        test_static_graph()


#Test Raise Error
class TestScatterNdOpRaise(unittest.TestCase):
    def test_check_raise(self):
        def check_raise_is_test():
            try:
                ref5 = fluid.layers.data(
                    name='ref5', shape=[3, 4, 5], dtype='float32')
                index5 = fluid.layers.data(
                    name='index5', shape=[2, 10], dtype='int32')
                updates5 = fluid.layers.data(
                    name='updates5', shape=[2, 10], dtype='float32')
                output5 = fluid.layers.scatter_nd_add(ref5, index5, updates5)
            except Exception as e:
                t = \
                "The last dimension of Input(Index)'s shape should be no greater "
                if t in str(e):
                    raise IndexError

        self.assertRaises(IndexError, check_raise_is_test)

    def test_check_raise2(self):
        with self.assertRaises(ValueError):
            ref6 = fluid.layers.data(
                name='ref6',
                shape=[10, 9, 8, 1, 3],
                dtype='double',
                append_batch_size=False)
            index6 = fluid.layers.data(
                name='index6',
                shape=[5, 8, 5],
                dtype='int32',
                append_batch_size=False)
            updates6 = fluid.layers.data(
                name='update6',
                shape=[5, 8],
                dtype='float32',
                append_batch_size=False)
            output6 = fluid.layers.scatter_nd_add(ref6, index6, updates6)

    def test_check_raise3(self):
        def check_raise_is_test():
            try:
                shape = [3, 4, 5]
                index7 = fluid.layers.data(
                    name='index7', shape=[2, 1], dtype='int32')
                updates7 = fluid.layers.data(
                    name='updates7', shape=[2, 4, 5, 20], dtype='float32')
                output7 = fluid.layers.scatter_nd(index7, updates7, shape)
            except Exception as e:
                t = \
                "Updates has wrong shape"
                if t in str(e):
                    raise ValueError

        self.assertRaises(ValueError, check_raise_is_test)


class TestDygraph(unittest.TestCase):
    def test_dygraph(self):
        with fluid.dygraph.guard(fluid.CPUPlace()):
            index_data = np.array([[1, 1], [0, 1], [1, 3]]).astype(np.int64)
            index = fluid.dygraph.to_variable(index_data)
            updates = paddle.rand(shape=[3, 9, 10], dtype='float32')
            shape = [3, 5, 9, 10]
            output = paddle.scatter_nd(index, updates, shape)

    def test_dygraph_1(self):
        with fluid.dygraph.guard(fluid.CPUPlace()):
            x = paddle.rand(shape=[3, 5, 9, 10], dtype='float32')
            updates = paddle.rand(shape=[3, 9, 10], dtype='float32')
            index_data = np.array([[1, 1], [0, 1], [1, 3]]).astype(np.int64)
            index = fluid.dygraph.to_variable(index_data)
            output = paddle.scatter_nd_add(x, index, updates)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
