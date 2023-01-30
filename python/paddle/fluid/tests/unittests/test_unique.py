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

<<<<<<< HEAD
import unittest

import numpy as np
from op_test import OpTest

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core


class TestUniqueOp(OpTest):
=======
from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.fluid.framework import _test_eager_guard


class TestUniqueOp(OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "unique"
        self.init_config()

    def test_check_output(self):
<<<<<<< HEAD
        paddle.enable_static()
        self.check_output()
        paddle.disable_static()
=======
        self.check_output()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def init_config(self):
        self.inputs = {
            'X': np.array([2, 3, 3, 1, 5, 3], dtype='int64'),
        }
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        self.outputs = {
            'Out': np.array([2, 3, 1, 5], dtype='int64'),
<<<<<<< HEAD
            'Index': np.array([0, 1, 1, 2, 3, 1], dtype='int32'),
=======
            'Index': np.array([0, 1, 1, 2, 3, 1], dtype='int32')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }


class TestOne(TestUniqueOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_config(self):
        self.inputs = {
            'X': np.array([2], dtype='int64'),
        }
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        self.outputs = {
            'Out': np.array([2], dtype='int64'),
<<<<<<< HEAD
            'Index': np.array([0], dtype='int32'),
=======
            'Index': np.array([0], dtype='int32')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }


class TestRandom(TestUniqueOp):
<<<<<<< HEAD
    def init_config(self):
        self.inputs = {'X': np.random.randint(0, 100, (150,), dtype='int64')}
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT64)}
        np_unique, np_index, reverse_index = np.unique(
            self.inputs['X'], True, True
        )
=======

    def init_config(self):
        self.inputs = {'X': np.random.randint(0, 100, (150, ), dtype='int64')}
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT64)}
        np_unique, np_index, reverse_index = np.unique(self.inputs['X'], True,
                                                       True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        np_tuple = [(np_unique[i], np_index[i]) for i in range(len(np_unique))]
        np_tuple.sort(key=lambda x: x[1])
        target_out = np.array([i[0] for i in np_tuple], dtype='int64')
        target_index = np.array(
<<<<<<< HEAD
            [list(target_out).index(i) for i in self.inputs['X']], dtype='int64'
        )
=======
            [list(target_out).index(i) for i in self.inputs['X']],
            dtype='int64')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.outputs = {'Out': target_out, 'Index': target_index}


class TestUniqueRaiseError(unittest.TestCase):
<<<<<<< HEAD
    def test_errors(self):
        paddle.enable_static()

        def test_type():
            paddle.unique([10])
=======

    def test_errors(self):

        def test_type():
            fluid.layers.unique([10])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.assertRaises(TypeError, test_type)

        def test_dtype():
            data = fluid.data(shape=[10], dtype="float16", name="input")
<<<<<<< HEAD
            paddle.unique(data)

        self.assertRaises(TypeError, test_dtype)
        paddle.disable_static()


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestOneGPU(TestUniqueOp):
=======
            fluid.layers.unique(data)

        self.assertRaises(TypeError, test_dtype)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestOneGPU(TestUniqueOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_config(self):
        self.inputs = {
            'X': np.array([2], dtype='int64'),
        }
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        self.outputs = {
            'Out': np.array([2], dtype='int64'),
<<<<<<< HEAD
            'Index': np.array([0], dtype='int32'),
=======
            'Index': np.array([0], dtype='int32')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

    def test_check_output(self):
        if core.is_compiled_with_cuda():
<<<<<<< HEAD
            paddle.enable_static()
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-5)
            paddle.disable_static()


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestRandomGPU(TestUniqueOp):
    def init_config(self):
        self.inputs = {'X': np.random.randint(0, 100, (150,), dtype='int64')}
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT64)}
        np_unique, np_index, reverse_index = np.unique(
            self.inputs['X'], True, True
        )
=======
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-5)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestRandomGPU(TestUniqueOp):

    def init_config(self):
        self.inputs = {'X': np.random.randint(0, 100, (150, ), dtype='int64')}
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT64)}
        np_unique, np_index, reverse_index = np.unique(self.inputs['X'], True,
                                                       True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        np_tuple = [(np_unique[i], np_index[i]) for i in range(len(np_unique))]
        np_tuple.sort(key=lambda x: x[1])
        target_out = np.array([i[0] for i in np_tuple], dtype='int64')
        target_index = np.array(
<<<<<<< HEAD
            [list(target_out).index(i) for i in self.inputs['X']], dtype='int64'
        )
=======
            [list(target_out).index(i) for i in self.inputs['X']],
            dtype='int64')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.outputs = {'Out': target_out, 'Index': target_index}

    def test_check_output(self):
        if core.is_compiled_with_cuda():
<<<<<<< HEAD
            paddle.enable_static()
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-5)
            paddle.disable_static()


class TestSortedUniqueOp(TestUniqueOp):
    def init_config(self):
        self.inputs = {'X': np.array([2, 3, 3, 1, 5, 3], dtype='int64')}
        unique, indices, inverse, count = np.unique(
            self.inputs['X'],
            return_index=True,
            return_inverse=True,
            return_counts=True,
            axis=None,
        )
=======
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-5)


class TestSortedUniqueOp(TestUniqueOp):

    def init_config(self):
        self.inputs = {'X': np.array([2, 3, 3, 1, 5, 3], dtype='int64')}
        unique, indices, inverse, count = np.unique(self.inputs['X'],
                                                    return_index=True,
                                                    return_inverse=True,
                                                    return_counts=True,
                                                    axis=None)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.attrs = {
            'dtype': int(core.VarDesc.VarType.INT32),
            "return_index": True,
            "return_inverse": True,
            "return_counts": True,
            "axis": None,
<<<<<<< HEAD
            "is_sorted": True,
=======
            "is_sorted": True
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        self.outputs = {
            'Out': unique,
            'Indices': indices,
            "Index": inverse,
            "Counts": count,
        }


class TestUniqueOpAxisNone(TestUniqueOp):
<<<<<<< HEAD
    def init_config(self):
        self.inputs = {'X': np.random.random((4, 7, 10)).astype('float64')}
        unique, indices, inverse, counts = np.unique(
            self.inputs['X'],
            return_index=True,
            return_inverse=True,
            return_counts=True,
            axis=None,
        )
=======

    def init_config(self):
        self.inputs = {'X': np.random.random((4, 7, 10)).astype('float64')}
        unique, indices, inverse, counts = np.unique(self.inputs['X'],
                                                     return_index=True,
                                                     return_inverse=True,
                                                     return_counts=True,
                                                     axis=None)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.attrs = {
            'dtype': int(core.VarDesc.VarType.INT32),
            "return_index": True,
            "return_inverse": True,
            "return_counts": True,
            "axis": None,
<<<<<<< HEAD
            "is_sorted": True,
=======
            "is_sorted": True
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        self.outputs = {
            'Out': unique,
            'Indices': indices,
            "Index": inverse,
            "Counts": counts,
        }


class TestUniqueOpAxis1(TestUniqueOp):
<<<<<<< HEAD
    def init_config(self):
        self.inputs = {'X': np.random.random((3, 8, 8)).astype('float64')}
        unique, indices, inverse, counts = np.unique(
            self.inputs['X'],
            return_index=True,
            return_inverse=True,
            return_counts=True,
            axis=1,
        )
=======

    def init_config(self):
        self.inputs = {'X': np.random.random((3, 8, 8)).astype('float64')}
        unique, indices, inverse, counts = np.unique(self.inputs['X'],
                                                     return_index=True,
                                                     return_inverse=True,
                                                     return_counts=True,
                                                     axis=1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.attrs = {
            'dtype': int(core.VarDesc.VarType.INT32),
            "return_index": True,
            "return_inverse": True,
            "return_counts": True,
            "axis": [1],
<<<<<<< HEAD
            "is_sorted": True,
=======
            "is_sorted": True
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        self.outputs = {
            'Out': unique,
            'Indices': indices,
            "Index": inverse,
            "Counts": counts,
        }


class TestUniqueAPI(unittest.TestCase):
<<<<<<< HEAD
    def test_dygraph_api_out(self):
=======

    def test_dygraph_api_out(self):
        paddle.disable_static()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        x_data = x_data = np.random.randint(0, 10, (120))
        x = paddle.to_tensor(x_data)
        out = paddle.unique(x)
        expected_out = np.unique(x_data)
        self.assertTrue((out.numpy() == expected_out).all(), True)
<<<<<<< HEAD

    def test_dygraph_api_attr(self):
        x_data = np.random.random((3, 5, 5)).astype("float32")
        x = paddle.to_tensor(x_data)
        out, index, inverse, counts = paddle.unique(
            x,
            return_index=True,
            return_inverse=True,
            return_counts=True,
            axis=0,
        )
        np_out, np_index, np_inverse, np_counts = np.unique(
            x_data,
            return_index=True,
            return_inverse=True,
            return_counts=True,
            axis=0,
        )
=======
        paddle.enable_static()

    def test_dygraph_api_attr(self):
        paddle.disable_static()
        x_data = np.random.random((3, 5, 5)).astype("float32")
        x = paddle.to_tensor(x_data)
        out, index, inverse, counts = paddle.unique(x,
                                                    return_index=True,
                                                    return_inverse=True,
                                                    return_counts=True,
                                                    axis=0)
        np_out, np_index, np_inverse, np_counts = np.unique(x_data,
                                                            return_index=True,
                                                            return_inverse=True,
                                                            return_counts=True,
                                                            axis=0)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.assertTrue((out.numpy() == np_out).all(), True)
        self.assertTrue((index.numpy() == np_index).all(), True)
        self.assertTrue((inverse.numpy() == np_inverse).all(), True)
        self.assertTrue((counts.numpy() == np_counts).all(), True)
<<<<<<< HEAD

    def test_dygraph_attr_dtype(self):
        x_data = x_data = np.random.randint(0, 10, (120))
        x = paddle.to_tensor(x_data)
        out, indices, inverse, counts = paddle.unique(
            x,
            return_index=True,
            return_inverse=True,
            return_counts=True,
            dtype="int32",
        )
        expected_out, np_indices, np_inverse, np_counts = np.unique(
            x_data, return_index=True, return_inverse=True, return_counts=True
        )
=======
        paddle.enable_static()

    def test_dygraph_attr_dtype(self):
        paddle.disable_static()
        x_data = x_data = np.random.randint(0, 10, (120))
        x = paddle.to_tensor(x_data)
        out, indices, inverse, counts = paddle.unique(x,
                                                      return_index=True,
                                                      return_inverse=True,
                                                      return_counts=True,
                                                      dtype="int32")
        expected_out, np_indices, np_inverse, np_counts = np.unique(
            x_data, return_index=True, return_inverse=True, return_counts=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.assertTrue((out.numpy() == expected_out).all(), True)
        self.assertTrue((indices.numpy() == np_indices).all(), True)
        self.assertTrue((inverse.numpy() == np_inverse).all(), True)
        self.assertTrue((counts.numpy() == np_counts).all(), True)
<<<<<<< HEAD

    def test_static_graph(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.fluid.data(name='x', shape=[3, 2], dtype='float64')
            unique, inverse, counts = paddle.unique(
                x, return_inverse=True, return_counts=True, axis=0
            )
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            x_np = np.array([[1, 2], [3, 4], [1, 2]]).astype('float64')
            result = exe.run(
                feed={"x": x_np}, fetch_list=[unique, inverse, counts]
            )
        np_unique, np_inverse, np_counts = np.unique(
            x_np, return_inverse=True, return_counts=True, axis=0
        )
        np.testing.assert_allclose(result[0], np_unique, rtol=1e-05)
        np.testing.assert_allclose(result[1], np_inverse, rtol=1e-05)
        np.testing.assert_allclose(result[2], np_counts, rtol=1e-05)
        paddle.disable_static()


class TestUniqueError(unittest.TestCase):
    def test_input_dtype(self):
        def test_x_dtype():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
=======
        paddle.enable_static()

    def test_dygraph_api(self):
        with _test_eager_guard():
            self.test_dygraph_api_out()
            self.test_dygraph_api_attr()
            self.test_dygraph_attr_dtype()

    def test_static_graph(self):
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            x = paddle.fluid.data(name='x', shape=[3, 2], dtype='float64')
            unique, inverse, counts = paddle.unique(x,
                                                    return_inverse=True,
                                                    return_counts=True,
                                                    axis=0)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            x_np = np.array([[1, 2], [3, 4], [1, 2]]).astype('float64')
            result = exe.run(feed={"x": x_np},
                             fetch_list=[unique, inverse, counts])
        np_unique, np_inverse, np_counts = np.unique(x_np,
                                                     return_inverse=True,
                                                     return_counts=True,
                                                     axis=0)
        np.testing.assert_allclose(result[0], np_unique, rtol=1e-05)
        np.testing.assert_allclose(result[1], np_inverse, rtol=1e-05)
        np.testing.assert_allclose(result[2], np_counts, rtol=1e-05)


class TestUniqueError(unittest.TestCase):

    def test_input_dtype(self):

        def test_x_dtype():
            with paddle.static.program_guard(paddle.static.Program(),
                                             paddle.static.Program()):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                x = paddle.fluid.data(name='x', shape=[10, 10], dtype='float16')
                result = paddle.unique(x)

            self.assertRaises(TypeError, test_x_dtype)

    def test_attr(self):
<<<<<<< HEAD
        paddle.enable_static()
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        x = paddle.fluid.data(name='x', shape=[10, 10], dtype='float64')

        def test_return_index():
            result = paddle.unique(x, return_index=0)

        self.assertRaises(TypeError, test_return_index)

        def test_return_inverse():
            result = paddle.unique(x, return_inverse='s')

        self.assertRaises(TypeError, test_return_inverse)

        def test_return_counts():
            result = paddle.unique(x, return_counts=3)

        self.assertRaises(TypeError, test_return_counts)

        def test_axis():
            result = paddle.unique(x, axis='12')

        def test_dtype():
            result = paddle.unique(x, dtype='float64')

        self.assertRaises(TypeError, test_axis)
<<<<<<< HEAD
        paddle.disable_static()
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == "__main__":
    unittest.main()
