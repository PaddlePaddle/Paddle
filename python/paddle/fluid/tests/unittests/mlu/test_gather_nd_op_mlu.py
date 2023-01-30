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

<<<<<<< HEAD
=======
from __future__ import print_function
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import unittest
import sys

sys.path.append('..')
import numpy as np
from op_test import OpTest
import paddle.fluid as fluid
import paddle

paddle.enable_static()


def gather_nd_grad(x, index):
    # for TestGatherNdOpWithLowIndex
<<<<<<< HEAD
    dout_shape = index.shape[:-1] + x.shape[index.shape[-1] :]
    numel = 1
    for i in dout_shape:
        numel = numel * i
    dout = np.full(dout_shape, 1.0 / numel)
=======
    dout_shape = index.shape[:-1] + x.shape[index.shape[-1]:]
    numel = 1
    for i in dout_shape:
        numel = numel * i
    dout = np.full(dout_shape, 1. / numel)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    dx = np.full_like(x, 0)

    index = tuple(index.reshape(-1, index.shape[-1]).T)
    np.add.at(dx, index, dout)

    return dx


def test_class1(op_type, typename):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    class TestGatherNdOpWithEmptyIndex(OpTest):
        # Index has empty element, which means copy entire tensor

        def setUp(self):
            self.set_mlu()
            self.op_type = "gather_nd"
            self.python_api = paddle.gather_nd
            xnp = np.random.random((5, 20)).astype(typename)
            self.inputs = {
                'X': xnp,
<<<<<<< HEAD
                'Index': np.array([[], []]).astype("int32"),
=======
                'Index': np.array([[], []]).astype("int32")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }
            self.outputs = {
                'Out': np.vstack((xnp[np.newaxis, :], xnp[np.newaxis, :]))
            }

        def set_mlu(self):
            self.__class__.use_mlu = True
            self.place = paddle.device.MLUPlace(0)

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if typename == "float16":
                self.__class__.no_need_check_grad = True
            else:
                self.check_grad_with_place(self.place, ['X'], 'Out')

    cls_name = "{0}_{1}_1".format(op_type, typename)
    TestGatherNdOpWithEmptyIndex.__name__ = cls_name
    globals()[cls_name] = TestGatherNdOpWithEmptyIndex


def test_class2(op_type, typename):
<<<<<<< HEAD
    class TestGatherNdOpWithIndex1(OpTest):
=======

    class TestGatherNdOpWithIndex1(OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def setUp(self):
            self.set_mlu()
            self.op_type = "gather_nd"
            self.python_api = paddle.gather_nd
            xnp = np.random.random((5, 20)).astype(typename)
            self.inputs = {'X': xnp, 'Index': np.array([1]).astype("int32")}
            self.outputs = {'Out': self.inputs["X"][self.inputs["Index"]]}

        def set_mlu(self):
            self.__class__.use_mlu = True
            self.place = paddle.device.MLUPlace(0)

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if typename == "float16":
                self.__class__.no_need_check_grad = True
            else:
                self.check_grad_with_place(self.place, ['X'], 'Out')

    cls_name = "{0}_{1}_2".format(op_type, typename)
    TestGatherNdOpWithIndex1.__name__ = cls_name
    globals()[cls_name] = TestGatherNdOpWithIndex1


def test_class3(op_type, typename):
<<<<<<< HEAD
    class TestGatherNdOpWithLowIndex(OpTest):
        # Index has low rank, X has high rank
=======

    class TestGatherNdOpWithLowIndex(OpTest):
        #Index has low rank, X has high rank
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        def setUp(self):
            self.set_mlu()
            self.op_type = "gather_nd"
            self.python_api = paddle.gather_nd
            xnp = np.random.uniform(0, 100, (10, 10)).astype(typename)
            index = np.array([[1], [2]]).astype("int64")

            self.inputs = {'X': xnp, 'Index': index}
            self.outputs = {'Out': xnp[tuple(index.T)]}
            self.x_grad = gather_nd_grad(xnp, index)

        def set_mlu(self):
            self.__class__.use_mlu = True
            self.place = paddle.device.MLUPlace(0)

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if typename == "float16":
                self.__class__.no_need_check_grad = True
            else:
<<<<<<< HEAD
                self.check_grad_with_place(
                    self.place, ['X'], 'Out', user_defined_grads=[self.x_grad]
                )
=======
                self.check_grad_with_place(self.place, ['X'],
                                           'Out',
                                           user_defined_grads=[self.x_grad])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    cls_name = "{0}_{1}_3".format(op_type, typename)
    TestGatherNdOpWithLowIndex.__name__ = cls_name
    globals()[cls_name] = TestGatherNdOpWithLowIndex


def test_class4(op_type, typename):
<<<<<<< HEAD
    class TestGatherNdOpIndex1(OpTest):
        # Index has low rank, X has high rank
=======

    class TestGatherNdOpIndex1(OpTest):
        #Index has low rank, X has high rank
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        def setUp(self):
            self.set_mlu()
            self.op_type = "gather_nd"
            self.python_api = paddle.gather_nd
            xnp = np.random.uniform(0, 100, (10, 10)).astype(typename)
            index = np.array([1, 2]).astype("int32")

            self.inputs = {'X': xnp, 'Index': index}

            self.outputs = {'Out': xnp[tuple(index.T)]}

        def set_mlu(self):
            self.__class__.use_mlu = True
            self.place = paddle.device.MLUPlace(0)

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if typename == "float16":
                self.__class__.no_need_check_grad = True
            else:
                self.check_grad_with_place(self.place, ['X'], 'Out')

    cls_name = "{0}_{1}_4".format(op_type, typename)
    TestGatherNdOpIndex1.__name__ = cls_name
    globals()[cls_name] = TestGatherNdOpIndex1


def test_class5(op_type, typename):
<<<<<<< HEAD
    class TestGatherNdOpWithSameIndexAsX(OpTest):
        # Index has same rank as X's rank
=======

    class TestGatherNdOpWithSameIndexAsX(OpTest):
        #Index has same rank as X's rank
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        def setUp(self):
            self.set_mlu()
            self.op_type = "gather_nd"
            self.python_api = paddle.gather_nd
            xnp = np.random.uniform(0, 100, (10, 10)).astype(typename)
            index = np.array([[1, 1], [2, 1]]).astype("int64")

            self.inputs = {'X': xnp, 'Index': index}
<<<<<<< HEAD
            self.outputs = {'Out': xnp[tuple(index.T)]}  # [25, 22]
=======
            self.outputs = {'Out': xnp[tuple(index.T)]}  #[25, 22]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        def set_mlu(self):
            self.__class__.use_mlu = True
            self.place = paddle.device.MLUPlace(0)

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if typename == "float16":
                self.__class__.no_need_check_grad = True
            else:
                self.check_grad_with_place(self.place, ['X'], 'Out')

    cls_name = "{0}_{1}_5".format(op_type, typename)
    TestGatherNdOpWithSameIndexAsX.__name__ = cls_name
    globals()[cls_name] = TestGatherNdOpWithSameIndexAsX


def test_class6(op_type, typename):
<<<<<<< HEAD
    class TestGatherNdOpWithHighRankSame(OpTest):
        # Both Index and X have high rank, and Rank(Index) = Rank(X)
=======

    class TestGatherNdOpWithHighRankSame(OpTest):
        #Both Index and X have high rank, and Rank(Index) = Rank(X)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        def setUp(self):
            self.set_mlu()
            self.op_type = "gather_nd"
            self.python_api = paddle.gather_nd
            shape = (5, 2, 3, 1, 10)
            xnp = np.random.rand(*shape).astype(typename)
<<<<<<< HEAD
            index = np.vstack(
                [np.random.randint(0, s, size=2) for s in shape]
            ).T
=======
            index = np.vstack([np.random.randint(0, s, size=2)
                               for s in shape]).T
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.inputs = {'X': xnp, 'Index': index.astype("int32")}
            self.outputs = {'Out': xnp[tuple(index.T)]}

        def set_mlu(self):
            self.__class__.use_mlu = True
            self.place = paddle.device.MLUPlace(0)

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if typename == "float16":
                self.__class__.no_need_check_grad = True
            else:
                self.check_grad_with_place(self.place, ['X'], 'Out')

    cls_name = "{0}_{1}_6".format(op_type, typename)
    TestGatherNdOpWithHighRankSame.__name__ = cls_name
    globals()[cls_name] = TestGatherNdOpWithHighRankSame


def test_class7(op_type, typename):
<<<<<<< HEAD
    class TestGatherNdOpWithHighRankDiff(OpTest):
        # Both Index and X have high rank, and Rank(Index) < Rank(X)
=======

    class TestGatherNdOpWithHighRankDiff(OpTest):
        #Both Index and X have high rank, and Rank(Index) < Rank(X)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        def setUp(self):
            self.set_mlu()
            self.op_type = "gather_nd"
            self.python_api = paddle.gather_nd
            shape = (2, 3, 4, 1, 10)
            xnp = np.random.rand(*shape).astype(typename)
            index = np.vstack(
<<<<<<< HEAD
                [np.random.randint(0, s, size=200) for s in shape]
            ).T
=======
                [np.random.randint(0, s, size=200) for s in shape]).T
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            index_re = index.reshape([20, 5, 2, 5])

            self.inputs = {'X': xnp, 'Index': index_re.astype("int32")}
            self.outputs = {'Out': xnp[tuple(index.T)].reshape([20, 5, 2])}

        def set_mlu(self):
            self.__class__.use_mlu = True
            self.place = paddle.device.MLUPlace(0)

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if typename == "float16":
                self.__class__.no_need_check_grad = True
            else:
                self.check_grad_with_place(self.place, ['X'], 'Out')

    cls_name = "{0}_{1}_7".format(op_type, typename)
    TestGatherNdOpWithHighRankDiff.__name__ = cls_name
    globals()[cls_name] = TestGatherNdOpWithHighRankDiff


<<<<<<< HEAD
# Test Python API
class TestGatherNdAPI2(unittest.TestCase):
=======
#Test Python API
class TestGatherNdAPI2(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_imperative(self):
        paddle.disable_static()
        input_1 = np.array([[1, 2], [3, 4], [5, 6]]).astype("float32")
        index_1 = np.array([[1]]).astype("int32")
        input = fluid.dygraph.to_variable(input_1)
        index = fluid.dygraph.to_variable(index_1)
<<<<<<< HEAD
        output = paddle.gather(input, index)
=======
        output = paddle.fluid.layers.gather(input, index)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        output_np = output.numpy()
        expected_output = np.array([3, 4])
        np.testing.assert_allclose(output_np[0], expected_output, rtol=1e-6)
        paddle.enable_static()


for _typename in {'float16', 'float32'}:
    test_class1('gather_nd', _typename)
    test_class2('gather_nd', _typename)
    test_class3('gather_nd', _typename)
    test_class4('gather_nd', _typename)
    test_class5('gather_nd', _typename)
    test_class6('gather_nd', _typename)
    test_class7('gather_nd', _typename)

if __name__ == "__main__":
    unittest.main()
