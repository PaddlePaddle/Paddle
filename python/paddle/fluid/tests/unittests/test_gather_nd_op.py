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

import unittest

import numpy as np
from op_test import OpTest

import paddle
import paddle.fluid as fluid


class TestGatherNdOpWithEmptyIndex(OpTest):
    # Index has empty element, which means copy entire tensor

    def setUp(self):
        self.op_type = "gather_nd"
        self.prim_op_type = "prim"
        self.python_api = paddle.gather_nd
        self.enable_cinn = False
        xnp = np.random.random((5, 20)).astype("float64")
        self.inputs = {'X': xnp, 'Index': np.array([[], []]).astype("int32")}
        self.outputs = {
            'Out': np.vstack((xnp[np.newaxis, :], xnp[np.newaxis, :]))
        }

    def test_check_output(self):
        self.check_output(check_eager=False)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_eager=False, check_prim=True)


class TestGatherNdOpWithIndex1(OpTest):
    def setUp(self):
        self.op_type = "gather_nd"
        self.prim_op_type = "prim"
        self.python_api = paddle.gather_nd
        self.enable_cinn = False
        xnp = np.random.random((5, 20)).astype("float64")
        self.inputs = {'X': xnp, 'Index': np.array([1]).astype("int32")}
        self.outputs = {'Out': self.inputs["X"][self.inputs["Index"]]}

    def test_check_output(self):
        self.check_output(check_eager=False)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_eager=False, check_prim=True)


class TestGatherNdOpWithLowIndex(OpTest):
    # Index has low rank, X has high rank

    def setUp(self):
        self.op_type = "gather_nd"
        self.prim_op_type = "prim"
        self.python_api = paddle.gather_nd
        self.enable_cinn = False
        xnp = np.random.uniform(0, 100, (10, 10)).astype("float64")
        index = np.array([[1], [2]]).astype("int64")

        self.inputs = {'X': xnp, 'Index': index}

        self.outputs = {
            'Out': xnp[tuple(index.T)]
        }  # [[14, 25, 1], [76, 22, 3]]

    def test_check_output(self):
        self.check_output(check_eager=False)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_eager=False, check_prim=True)


class TestGatherNdOpIndex1(OpTest):
    # Index has low rank, X has high rank

    def setUp(self):
        self.op_type = "gather_nd"
        self.prim_op_type = "prim"
        self.python_api = paddle.gather_nd
        self.enable_cinn = False
        xnp = np.random.uniform(0, 100, (10, 10)).astype("float64")
        index = np.array([1, 2]).astype("int32")

        self.inputs = {'X': xnp, 'Index': index}

        self.outputs = {'Out': xnp[tuple(index.T)]}

    def test_check_output(self):
        self.check_output(check_eager=False)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_eager=False, check_prim=True)


class TestGatherNdOpWithSameIndexAsX(OpTest):
    # Index has same rank as X's rank

    def setUp(self):
        self.op_type = "gather_nd"
        self.prim_op_type = "prim"
        self.python_api = paddle.gather_nd
        self.enable_cinn = False
        xnp = np.random.uniform(0, 100, (10, 10)).astype("float64")
        index = np.array([[1, 1], [2, 1]]).astype("int64")

        self.inputs = {'X': xnp, 'Index': index}
        self.outputs = {'Out': xnp[tuple(index.T)]}  # [25, 22]

    def test_check_output(self):
        self.check_output(check_eager=False)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_eager=False, check_prim=True)


class TestGatherNdOpWithHighRankSame(OpTest):
    # Both Index and X have high rank, and Rank(Index) = Rank(X)

    def setUp(self):
        self.op_type = "gather_nd"
        self.prim_op_type = "prim"
        self.python_api = paddle.gather_nd
        self.enable_cinn = False
        shape = (5, 2, 3, 1, 10)
        xnp = np.random.rand(*shape).astype("float64")
        index = np.vstack([np.random.randint(0, s, size=2) for s in shape]).T

        self.inputs = {'X': xnp, 'Index': index.astype("int32")}
        self.outputs = {'Out': xnp[tuple(index.T)]}

    def test_check_output(self):
        self.check_output(check_eager=False)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_eager=False, check_prim=True)


class TestGatherNdOpWithHighRankDiff(OpTest):
    # Both Index and X have high rank, and Rank(Index) < Rank(X)

    def setUp(self):
        self.op_type = "gather_nd"
        self.prim_op_type = "prim"
        self.python_api = paddle.gather_nd
        self.enable_cinn = False
        shape = (2, 3, 4, 1, 10)
        xnp = np.random.rand(*shape).astype("float64")
        index = np.vstack([np.random.randint(0, s, size=200) for s in shape]).T
        index_re = index.reshape([20, 5, 2, 5])

        self.inputs = {'X': xnp, 'Index': index_re.astype("int32")}
        self.outputs = {'Out': xnp[tuple(index.T)].reshape([20, 5, 2])}

    def test_check_output(self):
        self.check_output(check_eager=False)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_eager=False, check_prim=True)


# Test Python API
class TestGatherNdOpAPI(unittest.TestCase):
    def test_case1(self):
        x1 = paddle.static.data(
            name='x1', shape=[-1, 30, 40, 50, 60], dtype='float32'
        )
        index1 = paddle.static.data(
            name='index1', shape=[-1, 2, 4], dtype='int32'
        )
        output1 = paddle.gather_nd(x1, index1)

    def test_case2(self):
        x2 = paddle.static.data(
            name='x2', shape=[-1, 30, 40, 50], dtype='float32'
        )
        index2 = paddle.static.data(
            name='index2', shape=[-1, 2, 2], dtype='int64'
        )
        output2 = paddle.gather_nd(x2, index2)

    def test_case3(self):
        x3 = paddle.static.data(name='x3', shape=[-1, 3, 4, 5], dtype='float32')
        index3 = paddle.static.data(
            name='index3', shape=[-1, 2, 1], dtype='int32'
        )
        output3 = paddle.gather_nd(x3, index3, name="gather_nd_layer")


# Test Raise Index Error
class TestGatherNdOpRaise(unittest.TestCase):
    def test_check_raise(self):
        def check_raise_is_test():
            try:
                x = paddle.static.data(
                    name='x', shape=[-1, 3, 4, 5], dtype='float32'
                )
                index = paddle.static.data(
                    name='index', shape=[-1, 2, 10], dtype='int32'
                )
                output = paddle.gather_nd(x, index)
            except Exception as e:
                t = "Input(Index).shape[-1] should be no greater than Input(X).rank"
                if t in str(e):
                    raise IndexError

        self.assertRaises(IndexError, check_raise_is_test)


class TestGatherNdError(unittest.TestCase):
    def test_error(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):

            shape = [8, 9, 6]
            x = paddle.fluid.data(shape=shape, dtype='float32', name='x')
            index = paddle.fluid.data(shape=shape, dtype='bool', name='index')
            index_float = paddle.fluid.data(
                shape=shape, dtype='float32', name='index_float'
            )
            np_x = np.random.random(shape).astype('float32')
            np_index = np.array(np.random.randint(2, size=shape, dtype=bool))

            def test_x_type():
                paddle.gather_nd(np_x, index)

            self.assertRaises(TypeError, test_x_type)

            def test_index_type():
                paddle.gather_nd(x, np_index)

            self.assertRaises(TypeError, test_index_type)

            def test_index_dtype():
                paddle.gather_nd(x, index_float)

            self.assertRaises(TypeError, test_index_dtype)


class TestGatherNdAPI2(unittest.TestCase):
    def test_static(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            data1 = paddle.static.data('data1', shape=[-1, 2], dtype='float64')
            data1.desc.set_need_check_feed(False)
            index = paddle.static.data('index', shape=[-1, 1], dtype='int32')
            index.desc.set_need_check_feed(False)
            out = paddle.gather_nd(data1, index)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            input = np.array([[1, 2], [3, 4], [5, 6]])
            index_1 = np.array([[1]]).astype('int32')
            (result,) = exe.run(
                feed={"data1": input, "index": index_1}, fetch_list=[out]
            )
            expected_output = np.array([[3, 4]])
        np.testing.assert_allclose(result, expected_output, rtol=1e-05)

    def test_static_fp16_with_gpu(self):
        if paddle.fluid.core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input = np.array(
                    [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]],
                    dtype='float16',
                )
                index = np.array([[0, 1]], dtype='int32')
                res_np = np.array([[3, 4]], dtype='float16')

                x = paddle.static.data(
                    name="x", shape=[2, 3, 2], dtype="float16"
                )
                x.desc.set_need_check_feed(False)
                idx = paddle.static.data(
                    name="index", shape=[1, 2], dtype="int32"
                )
                idx.desc.set_need_check_feed(False)

                y = paddle.gather_nd(x, idx)

                exe = paddle.static.Executor(place)
                res = exe.run(
                    paddle.static.default_main_program(),
                    feed={"x": input, "index": index},
                    fetch_list=[y],
                )

                np.testing.assert_allclose(res[0], res_np, rtol=1e-05)

    def test_imperative(self):
        paddle.disable_static()
        input_1 = np.array([[1, 2], [3, 4], [5, 6]])
        index_1 = np.array([[1]])
        input = fluid.dygraph.to_variable(input_1)
        index = fluid.dygraph.to_variable(index_1)
        output = paddle.gather(input, index)
        output_np = output.numpy()
        expected_output = np.array([[3, 4]])
        np.testing.assert_allclose(output_np, expected_output, rtol=1e-05)
        paddle.enable_static()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
