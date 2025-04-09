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

import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16
from utils import dygraph_guard

import paddle
from paddle import base
from paddle.base.dygraph.base import switch_to_static_graph
from paddle.framework import core


def gather_numpy(x, index, axis):
    x_transpose = np.swapaxes(x, 0, axis)
    tmp_gather = x_transpose[index, ...]
    gather = np.swapaxes(tmp_gather, 0, axis)
    return gather


class TestGatherOp(OpTest):
    def setUp(self):
        self.op_type = "gather"
        self.python_api = paddle.gather
        self.public_python_api = paddle.gather
        self.config()
        self.prim_op_type = "prim"
        self.init_inputs_and_outputs()
        self.if_enable_cinn()

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True
        )

    def config(self):
        """
        For multi-dimension input
        """
        self.x_shape = (10, 20)
        self.config_dtype()
        self.index = [1, 3, 5]
        self.index_type = "int32"

    def config_dtype(self):
        self.x_type = "float64"

    def init_inputs_and_outputs(self):
        xnp = np.random.random(self.x_shape).astype(self.x_type)
        if self.x_type == 'complex64' or self.x_type == "cpmolex128":
            xnp = (
                np.random.randint(-10, 10, size=(10, 10))
                + 1j * np.random.randint(-10, 10, size=(10, 10))
            ).astype(self.x_type)
        self.inputs = {
            'X': xnp,
            'Index': np.array(self.index).astype(self.index_type),
        }
        self.outputs = {'Out': self.inputs["X"][self.inputs["Index"]]}

    def if_enable_cinn(self):
        pass


class TestGatherOp_ZeroDim(TestGatherOp):
    def config(self):
        """
        For multi-dimension input
        """
        self.x_shape = 100
        self.config_dtype()
        self.index = 2
        self.index_type = "int32"

    def if_enable_cinn(self):
        self.enable_cinn = False


class TestGatherOpFP16(TestGatherOp):
    def config_dtype(self):
        self.x_type = "float16"


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or core.cudnn_version() < 8100
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "only support compiled with CUDA and cudnn version need larger than 8.1.0 and device's compute capability is at least 8.0",
)
class TestGatherOpBFP16(TestGatherOp):
    def config_dtype(self):
        self.x_type = "float32"
        self.dtype = np.uint16

    def init_inputs_and_outputs(self):
        xnp = np.random.random(self.x_shape).astype(self.x_type)
        self.inputs = {
            'X': convert_float_to_uint16(xnp),
            'Index': np.array(self.index).astype(self.index_type),
        }
        self.outputs = {
            'Out': convert_float_to_uint16(xnp[self.inputs["Index"]])
        }

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_output(self):
        self.check_output_with_place(
            place=paddle.CUDAPlace(0), check_pir=True, check_symbol_infer=False
        )

    def test_check_grad(self):
        self.check_grad_with_place(
            paddle.CUDAPlace(0),
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


class TestGatherOpComplex64(TestGatherOp):
    def config_dtype(self):
        self.x_type = "complex64"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestGatherOpComplex128(TestGatherOp):
    def config_dtype(self):
        self.x_type = "complex128"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestCase1(TestGatherOp):
    def config(self):
        """
        For one dimension input
        """
        self.x_shape = 100
        self.config_dtype()
        self.index = [1, 3, 5]
        self.index_type = "int32"

    def config_dtype(self):
        self.x_type = "float64"


class TestCase1FP16(TestCase1):
    def config_dtype(self):
        self.x_type = "float16"


class TestCase1BFP16(TestGatherOpBFP16):
    def config(self):
        self.x_shape = 100
        self.config_dtype()
        self.index = [1, 3, 5]
        self.index_type = "int32"


class TestCase1Complex64(TestCase1):
    def config_dtype(self):
        self.x_type = "complex64"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestCase1Complex128(TestCase1):
    def config_dtype(self):
        self.x_type = "complex128"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestCase2(TestGatherOp):
    def config(self):
        """
        For int64_t index type
        """
        self.x_shape = 100
        self.config_dtype()
        self.index = [1, 3, 5]
        self.index_type = "int64"

    def config_dtype(self):
        self.x_type = "float64"


class TestCase2FP16(TestCase2):
    def config_dtype(self):
        self.x_type = "float16"


class TestCase2BFP16(TestGatherOpBFP16):
    def config(self):
        self.x_shape = 100
        self.config_dtype()
        self.index = [1, 3, 5]
        self.index_type = "int64"


class TestCase2Complex64(TestCase2):
    def config_dtype(self):
        self.x_type = "complex64"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestCase2Complex128(TestCase2):
    def config_dtype(self):
        self.x_type = "complex128"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestCase3(TestGatherOp):
    def config(self):
        """
        For other input type
        """
        self.x_shape = (10, 20)
        self.config_dtype()
        self.index = [1, 3, 5]
        self.index_type = "int64"

    def config_dtype(self):
        self.x_type = "float64"


class TestCase3Fp16(TestCase3):
    def config_dtype(self):
        self.x_type = "float16"


class TestCase3BFP16(TestGatherOpBFP16):
    def config(self):
        self.x_shape = (10, 20)
        self.config_dtype()
        self.index = [1, 3, 5]
        self.index_type = "int64"


class TestCase3Complex64(TestCase3):
    def config_dtype(self):
        self.x_type = "complex64"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestCase3Complex128(TestCase3):
    def config_dtype(self):
        self.x_type = "complex128"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestCase4(TestGatherOp):
    def config(self):
        self.x_shape = (10, 20)
        self.attrs = {'overwrite': False}
        self.config_dtype()
        self.index = [1, 1]
        self.index_type = "int32"

    def config_dtype(self):
        self.x_type = "float64"


class TestCase4FP16(TestCase4):
    def config_dtype(self):
        self.x_type = "float16"


class TestCase4BFP16(TestGatherOpBFP16):
    def config(self):
        self.x_shape = (10, 20)
        self.attrs = {'overwrite': False}
        self.config_dtype()
        self.index = [1, 1]
        self.index_type = "int32"


class TestCase4Complex64(TestCase4):
    def config_dtype(self):
        self.x_type = "complex64"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestCase4Complex128(TestCase4):
    def config_dtype(self):
        self.x_type = "complex128"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestCase5(TestGatherOp):
    def config(self):
        self.x_shape = (10, 20)
        self.attrs = {'overwrite': False}
        self.config_dtype()
        self.index = [1, 1, 3]
        self.index_type = "int32"

    def config_dtype(self):
        self.x_type = "float64"


class TestCase5BFP16(TestGatherOpBFP16):
    def config(self):
        self.x_shape = (10, 20)
        self.attrs = {'overwrite': False}
        self.config_dtype()
        self.index = [1, 1]
        self.index_type = "int32"


class TestCase5FP16(TestCase5):
    def config_dtype(self):
        self.x_type = "float16"


class TestCase5Complex64(TestCase5):
    def config_dtype(self):
        self.x_type = "complex64"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestCase5Complex128(TestCase5):
    def config_dtype(self):
        self.x_type = "complex128"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestCase6(TestGatherOp):
    def config(self):
        self.x_shape = (10, 20)
        self.attrs = {'overwrite': True}
        self.config_dtype()
        self.index = [1, 3]
        self.index_type = "int32"

    def config_dtype(self):
        self.x_type = "float64"


class TestCase6FP16(TestCase6):
    def config_dtype(self):
        self.x_type = "float16"


class TestCase6BFP16(TestGatherOpBFP16):
    def config(self):
        self.x_shape = (10, 20)
        self.attrs = {'overwrite': True}
        self.config_dtype()
        self.index = [1, 3]
        self.index_type = "int32"


class TestGatherBF16Op(OpTest):
    def setUp(self):
        self.op_type = "gather"
        self.python_api = paddle.gather
        self.dtype = np.uint16
        self.config()
        xnp = np.random.random(self.x_shape).astype(np.float32)
        axis_np = np.array(self.axis).astype(self.axis_type)
        index_np = np.array(self.index).astype(self.index_type)
        self.inputs = {
            'X': convert_float_to_uint16(xnp),
            'Index': index_np,
            'Axis': axis_np,
        }
        out = gather_numpy(self.inputs['X'], index_np, axis_np[0])
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', numeric_grad_delta=0.5, check_pir=True)

    def config(self):
        """
        For multi-dimension input
        """
        self.x_shape = (3, 88, 3)
        self.index = [1, 3, 5]
        self.index_type = "int32"
        self.axis = [1]
        self.axis_type = "int32"


class TestGatherNegativeAxis(OpTest):
    def setUp(self):
        self.op_type = "gather"
        self.python_api = paddle.gather
        self.dtype = np.uint16
        self.config()
        xnp = np.random.random(self.x_shape).astype(np.float32)
        axis_np = np.array(self.axis).astype(self.axis_type)
        index_np = np.array(self.index).astype(self.index_type)
        self.inputs = {
            'X': convert_float_to_uint16(xnp),
            'Index': index_np,
            'Axis': axis_np,
        }
        out = gather_numpy(self.inputs['X'], index_np, axis_np[0])
        self.outputs = {'Out': out}

    def test_check_output(self):
        places = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            self.check_output_with_place(place)

    def test_check_grad(self):
        places = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            self.check_grad_with_place(
                place, ['X'], 'Out', numeric_grad_delta=0.5
            )

    def config(self):
        """
        For multi-dimension input
        """
        self.x_shape = (100, 3)
        self.index = [0, 1, -2]
        self.index_type = "int32"
        self.axis = [-1]
        self.axis_type = "int32"


class TestOutOfRangeError(unittest.TestCase):
    def test_dygraph_forward_and_backward(self):
        with dygraph_guard():
            x = paddle.randn([100, 3]).cpu()
            x.stop_gradient = False
            y = paddle.gather(
                x,
                paddle.to_tensor([0, -2]).cpu(),
                axis=-1,
            )
            grad_x = paddle.grad(y, x)

    def test_dygraph_error(self):
        with dygraph_guard():
            # out of lower bound
            with self.assertRaises(IndexError):
                _ = paddle.gather(
                    paddle.randn([100, 3]).cpu(),
                    paddle.to_tensor([0, -4]).cpu(),
                    axis=1,
                )
            # out of upper bound
            with self.assertRaises(IndexError):
                _ = paddle.gather(
                    paddle.randn([100, 3]).cpu(),
                    paddle.to_tensor([0, 3]).cpu(),
                    axis=1,
                )


class TestCase6Complex64(TestCase6):
    def config_dtype(self):
        self.x_type = "complex64"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestCase6Complex128(TestCase6):
    def config_dtype(self):
        self.x_type = "complex128"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestGatherOp1(OpTest):
    def setUp(self):
        self.op_type = "gather"
        self.python_api = paddle.gather
        self.config()
        xnp = np.random.random(self.x_shape).astype(self.x_type)
        axis_np = np.array(self.axis).astype(self.index_type)
        index_np = np.array(self.index).astype(self.index_type)
        out = gather_numpy(xnp, index_np, axis_np[0])
        self.inputs = {'X': xnp, 'Index': index_np, 'Axis': axis_np}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True)

    def config(self):
        """
        For multi-dimension input
        """
        self.x_shape = (3, 88, 3)
        self.config_dtype()
        self.index = [1, 3, 5]
        self.index_type = "int32"
        self.axis = [1]
        self.axis_type = "int32"

    def config_dtype(self):
        self.x_type = "float64"


class TestGatherOp1FP16(TestGatherOp1):
    def config_dtype(self):
        self.x_type = "float16"


class TestGatherOp1Complex64(TestGatherOp1):
    def config_dtype(self):
        self.x_type = "complex64"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestGatherOp1Complex128(TestGatherOp1):
    def config_dtype(self):
        self.x_type = "complex128"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestGatherOp2(TestGatherOp1):
    def config(self):
        """
        For multi-dimension input
        """
        self.x_shape = (10, 88, 10)
        self.config_dtype()
        self.index = [1, 3, 5]
        self.index_type = "int64"
        self.axis = [0]
        self.axis_type = "int32"

    def config_dtype(self):
        self.x_type = "float64"


class TestGatherOp2FP16(TestGatherOp2):
    def config_dtype(self):
        self.x_type = "float16"


class TestGatherOp2Complex64(TestGatherOp2):
    def config_dtype(self):
        self.x_type = "complex64"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestGatherOp2Complex128(TestGatherOp2):
    def config_dtype(self):
        self.x_type = "complex128"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestGatherOp3(TestGatherOp1):
    def config(self):
        """
        For multi-dimension input
        """
        self.x_shape = (10, 88, 10)
        self.config_dtype()
        self.index = [1, 3, 5]
        self.index_type = "int64"
        self.axis = [2]
        self.axis_type = "int32"

    def config_dtype(self):
        self.x_type = "float64"


class TestGatherOp3FP16(TestGatherOp3):
    def config_dtype(self):
        self.x_type = "float16"


class TestGatherOp3Complex64(TestGatherOp3):
    def config_dtype(self):
        self.x_type = "complex64"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestGatherOp3Complex128(TestGatherOp3):
    def config_dtype(self):
        self.x_type = "complex128"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestGatherOp4(TestGatherOp1):
    def config(self):
        """
        For multi-dimension input
        """
        self.x_shape = (3, 100, 10)
        self.config_dtype()
        self.index = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.index_type = "int64"
        self.axis = [0]
        self.axis_type = "int32"
        self.attrs = {'overwrite': False}

    def config_dtype(self):
        self.x_type = "float64"


class TestGatherOp4FP16(TestGatherOp4):
    def config_dtype(self):
        self.x_type = "float16"


class TestGatherOp4Complex64(TestGatherOp4):
    def config_dtype(self):
        self.x_type = "complex64"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestGatherOp4Complex128(TestGatherOp4):
    def config_dtype(self):
        self.x_type = "complex128"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestGatherOp5(TestGatherOp):
    def config(self):
        """
        Test for negative axis
        """
        self.x_shape = (3, 100, 10)
        self.config_dtype()
        self.index = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.index_type = "int64"
        self.axis = [-1]
        self.axis_type = "int32"
        self.attrs = {'overwrite': False}

    def config_dtype(self):
        self.x_type = "float64"

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            check_pir=True,
            check_prim=True,
            check_prim_pir=True,
        )


class API_TestGather(unittest.TestCase):

    def test_out1(self):
        with base.program_guard(base.Program(), base.Program()):
            data1 = paddle.static.data('data1', shape=[-1, 2], dtype='float64')
            index = paddle.static.data('index', shape=[-1, 1], dtype='int64')
            out = paddle.gather(data1, index)
            place = base.CPUPlace()
            exe = base.Executor(place)
            input = np.array([[1, 2], [3, 4], [5, 6]]).astype('float64')
            index_1 = np.array([1, 2]).astype('int64')
            (result,) = exe.run(
                feed={"data1": input, "index": index_1}, fetch_list=[out]
            )
            expected_output = np.array([[3, 4], [5, 6]])
        np.testing.assert_allclose(result, expected_output, rtol=1e-05)

    def test_out2(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data('x', shape=[-1, 2], dtype='float64')
            index = paddle.static.data('index', shape=[-1, 1], dtype='int32')
            axis = paddle.static.data('axis', shape=[1], dtype='int32')
            out = paddle.gather(x, index, axis)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            x_np = np.array([[1, 2], [3, 4], [5, 6]]).astype('float64')
            index_np = np.array([1, 1]).astype('int32')
            axis_np = np.array([1]).astype('int32')
            (result,) = exe.run(
                feed={"x": x_np, "index": index_np, 'axis': axis_np},
                fetch_list=[out],
            )
            expected_output = gather_numpy(x_np, index_np, axis_np[0])
        np.testing.assert_allclose(result, expected_output, rtol=1e-05)


class API_TestDygraphGather(unittest.TestCase):
    def test_out1(self):
        paddle.disable_static()
        input_1 = np.array([[1, 2], [3, 4], [5, 6]])
        index_1 = np.array([1, 2])
        input = paddle.to_tensor(input_1)
        index = paddle.to_tensor(index_1)
        output = paddle.gather(input, index)
        output_np = output.numpy()
        expected_output = np.array([[3, 4], [5, 6]])
        np.testing.assert_allclose(output_np, expected_output, rtol=1e-05)
        paddle.enable_static()

    def test_out12(self):
        paddle.disable_static()
        input_1 = np.array([[1, 2], [3, 4], [5, 6]])
        index_1 = np.array([1, 2])
        x = paddle.to_tensor(input_1)
        index = paddle.to_tensor(index_1)
        output = paddle.gather(x, index, axis=0)
        output_np = output.numpy()
        expected_output = gather_numpy(input_1, index_1, axis=0)
        np.testing.assert_allclose(output_np, expected_output, rtol=1e-05)
        paddle.enable_static()

    def test_zero_index(self):
        paddle.disable_static()
        x = paddle.to_tensor([[1, 2], [3, 4]])
        index = paddle.to_tensor(np.array([]).astype('int64'))
        for axis in range(len(x.shape)):
            out = paddle.gather(x, index, axis)
            expected_shape = list(x.shape)
            expected_shape[axis] = 0
            self.assertEqual(list(out.shape), expected_shape)
        paddle.enable_static()

    def test_large_data(self):
        if not paddle.is_compiled_with_cuda():
            return

        x = np.random.rand(226862, 256).astype("float32")
        index = np.random.randint(-226862, 22682, size=(8859027))

        def test_dygraph():
            with base.dygraph.guard():
                gpu_out = paddle.gather(
                    paddle.to_tensor(x), paddle.to_tensor(index)
                )
                return gpu_out.numpy()

        @switch_to_static_graph
        def test_static_graph():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x_t = paddle.static.data(name="x", dtype=x.dtype, shape=x.shape)
                index_t = paddle.static.data(
                    name="index", dtype=index.dtype, shape=index.shape
                )
                out_t = paddle.gather(x_t, index_t)
                feed = {x_t.name: x, index_t.name: index}
                fetch = [out_t]

                gpu_exe = paddle.static.Executor(paddle.CUDAPlace(0))
                gpu_value = gpu_exe.run(feed=feed, fetch_list=fetch)[0]
                return gpu_value

        np.testing.assert_array_equal(test_dygraph(), test_static_graph())


class TestGathertError(unittest.TestCase):

    def test_error1(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            shape = [8, 9, 6]
            x = paddle.static.data(shape=shape, dtype='int8', name='x')
            axis = paddle.static.data(shape=[1], dtype='float32', name='axis')
            index = paddle.static.data(shape=shape, dtype='int32', name='index')
            index_float = paddle.static.data(
                shape=shape, dtype='float32', name='index_float'
            )

            def test_x_type():
                paddle.gather(x, index)

            self.assertRaises((TypeError, ValueError), test_x_type)

            def test_index_type():
                paddle.gather(x, index_float)

            self.assertRaises((TypeError, ValueError), test_index_type)

            def test_axis_dtype():
                paddle.gather(x, index, axis=1.11)

            self.assertRaises((TypeError, ValueError), test_axis_dtype)

            def test_axis_dtype1():
                paddle.gather(x, index, axis=axis)

            self.assertRaises((TypeError, ValueError), test_axis_dtype1)

    def test_error2(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            shape = [8, 9, 6]
            x = paddle.static.data(shape=shape, dtype='int8', name='x')
            index = paddle.static.data(shape=shape, dtype='int32', name='mask')
            index_float = paddle.static.data(
                shape=shape, dtype='float32', name='index_float'
            )

            def test_x_type():
                paddle.gather(x, index)

            self.assertRaises((TypeError, ValueError), test_x_type)

            def test_index_type():
                paddle.gather(x, index_float)

            self.assertRaises((TypeError, ValueError), test_index_type)

    def test_error3(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            shape = [8, 9, 6]
            x = paddle.static.data(shape=shape, dtype='int32', name='x')
            index = paddle.static.data(shape=shape, dtype='int32', name='index')

            def test_axis_minsize():
                paddle.gather(x, index, axis=-1)

            self.assertRaises(ValueError, test_axis_minsize)

            def test_axis_maxsize():
                paddle.gather(x, index, axis=512)

            self.assertRaises(ValueError, test_axis_maxsize)


class TestCheckOutType(unittest.TestCase):

    def test_out_type(self):
        data = paddle.static.data(shape=[16, 10], dtype='int64', name='x')
        index = paddle.static.data(shape=[4], dtype='int64', name='index')
        out = paddle.gather(data, index)
        self.assertTrue(
            out.dtype == paddle.int64 or out.dtype == core.DataType.INT64
        )

    def test_pir_out_type(self):
        with paddle.pir_utils.IrGuard():
            data = paddle.static.data(shape=[16, 10], dtype='int64', name='x')
            index = paddle.static.data(shape=[4], dtype='int64', name='index')
            out = paddle.gather(data, index)
            self.assertTrue(out.dtype == core.DataType.INT64)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
