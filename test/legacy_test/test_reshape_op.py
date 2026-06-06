# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import (
    OpTest,
    OpTestTool,
    convert_float_to_uint16,
    get_device_place,
    is_custom_device,
    skip_check_grad_ci,
)
from utils import dygraph_guard, static_guard

import paddle
from paddle import base
from paddle.base import core
from paddle.base.framework import _current_expected_place
from paddle.static import Program, program_guard


# situation 1: have shape( list, no tensor), no actual shape(Tensor)
class TestReshapeOp(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "reshape2"
        self.prim_op_type = "prim"
        self.python_api = paddle.tensor.reshape
        self.public_python_api = paddle.tensor.reshape
        self.python_out_sig = ['Out']
        self.inputs = {"X": np.random.random(self.ori_shape).astype("float32")}
        self.attrs = {"shape": self.new_shape}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.inferred_shape),
            'XShape': np.random.random(self.ori_shape).astype("float32"),
        }

    def init_data(self):
        self.ori_shape = (2, 60)
        self.new_shape = (12, 10)
        self.inferred_shape = (12, 10)

    def test_check_output(self):
        self.check_output(no_check_set=['XShape'], check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ["X"],
            "Out",
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


class TestReshapeOp_ZeroDim1(TestReshapeOp):
    def setUp(self):
        self.init_data()
        self.op_type = "reshape2"
        self.prim_op_type = "prim"
        self.enable_cinn = False
        self.python_api = paddle.tensor.reshape
        self.public_python_api = paddle.tensor.reshape
        self.python_out_sig = ['Out']
        self.inputs = {"X": np.random.random(self.ori_shape).astype("float32")}
        self.attrs = {"shape": self.new_shape}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.inferred_shape),
            'XShape': np.random.random(self.ori_shape).astype("float32"),
        }

    def init_data(self):
        self.ori_shape = ()
        self.new_shape = (1,)
        self.inferred_shape = (1,)


class TestReshapeOp_ZeroDim2(TestReshapeOp_ZeroDim1):
    def init_data(self):
        self.ori_shape = ()
        self.new_shape = (-1,)
        self.inferred_shape = (1,)


class TestReshapeOp_ZeroDim3(OpTest):
    def init_data(self):
        self.ori_shape = (1,)
        self.new_shape = ()
        self.inferred_shape = ()


@OpTestTool.skip_if(
    not (isinstance(_current_expected_place(), core.CPUPlace)),
    "GPU is not supported",
)
class TestReshapeOp_ZeroDim4(OpTest):
    def init_kernel_type(self):
        self.use_onednn = True

    def init_data(self):
        self.ori_shape = (1,)
        self.new_shape = ()
        self.inferred_shape = ()


class TestReshapeOp_ZeroSize(OpTest):
    def init_data(self):
        self.ori_shape = (0, 2)
        self.new_shape = (2, 0)
        self.inferred_shape = (2, 0)

    def setUp(self):
        self.init_data()
        self.op_type = "reshape2"
        self.python_api = paddle.tensor.reshape
        self.public_python_api = paddle.tensor.reshape
        self.python_out_sig = ['Out']
        self.inputs = {"X": np.random.random(self.ori_shape).astype("float32")}
        self.attrs = {"shape": self.new_shape}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.inferred_shape),
            'XShape': np.random.random(self.ori_shape).astype("float32"),
        }

    def test_check_output(self):
        self.check_output(no_check_set=['XShape'])

    def test_check_grad(self):
        self.check_grad(
            ["X"],
            "Out",
        )


@unittest.skipIf(
    not (paddle.is_compiled_with_cuda() or is_custom_device())
    or paddle.is_compiled_with_rocm(),
    "BFP16 test runs only on CUDA",
)
class TestReshapeBF16Op(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "reshape2"
        self.prim_op_type = "prim"
        self.enable_cinn = False
        self.python_api = paddle.tensor.reshape
        self.public_python_api = paddle.tensor.reshape
        self.python_out_sig = ['Out']
        self.dtype = np.uint16
        x = np.random.random(self.ori_shape).astype("float32")
        out = x.reshape(self.inferred_shape)
        self.inputs = {"X": convert_float_to_uint16(x)}
        self.attrs = {"shape": self.new_shape}
        self.outputs = {
            "Out": convert_float_to_uint16(out),
            'XShape': convert_float_to_uint16(
                np.random.random(self.ori_shape).astype("float32")
            ),
        }

    def init_data(self):
        self.ori_shape = (2, 60)
        self.new_shape = (12, 10)
        self.inferred_shape = (12, 10)

    def test_check_output(self):
        self.check_output(no_check_set=['XShape'], check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ["X"],
            "Out",
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )


class TestReshapeFP16Op(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "reshape2"
        self.prim_op_type = "prim"
        self.python_api = paddle.tensor.reshape
        self.public_python_api = paddle.tensor.reshape
        self.python_out_sig = ['Out']
        self.dtype = np.float16
        self.inputs = {"X": np.random.random(self.ori_shape).astype(self.dtype)}
        self.attrs = {"shape": self.new_shape}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.inferred_shape),
            'XShape': np.random.random(self.ori_shape).astype(self.dtype),
        }

    def init_data(self):
        self.ori_shape = (2, 60)
        self.new_shape = (12, 10)
        self.inferred_shape = (12, 10)

    def test_check_output(self):
        self.check_output(no_check_set=['XShape'], check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ["X"],
            "Out",
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )


class TestReshapeOpDimInfer1(TestReshapeOp):
    def init_data(self):
        self.ori_shape = (5, 25)
        self.new_shape = (5, -1, 5)
        self.inferred_shape = (5, -1, 5)


class TestReshapeOpDimInfer2(TestReshapeOp):
    def init_data(self):
        self.ori_shape = (10, 2, 6)
        self.new_shape = (10, 0, 3, -1)
        self.inferred_shape = (10, 2, 3, -1)


# situation 2: have shape(list, no tensor), have actual shape(Tensor)
class TestReshapeOpWithInputShape(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "reshape2"
        self.prim_op_type = "prim"
        self.python_api = paddle.tensor.reshape
        self.public_python_api = paddle.tensor.reshape
        self.python_out_sig = ['Out']

        self.inputs = {
            "X": np.random.random(self.ori_shape).astype("float32"),
            "Shape": np.array(self.actual_shape, dtype="int32"),
        }
        self.attrs = {"shape": self.new_shape}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.actual_shape),
            'XShape': np.random.random(self.ori_shape).astype("float32"),
        }

    def init_data(self):
        self.ori_shape = (6, 20)
        self.new_shape = (0, -1, 20)
        self.actual_shape = (2, 3, 20)

    def test_check_output(self):
        self.check_output(
            no_check_set=['XShape'], check_pir=True, check_symbol_infer=False
        )

    def test_check_grad(self):
        self.check_grad(
            ["X"],
            "Out",
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )


# Situation 3: have shape(list, have tensor), no actual shape(Tensor)
class TestReshapeOp_attr_ShapeTensor(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "reshape2"
        self.python_api = paddle.tensor.reshape
        self.public_python_api = paddle.tensor.reshape
        self.prim_op_type = "prim"
        self.python_out_sig = ['Out']

        shape_tensor = []
        for index, ele in enumerate(self.new_shape):
            shape_tensor.append(
                ("x" + str(index), np.ones(1).astype('int32') * ele)
            )

        self.inputs = {
            "X": np.random.random(self.ori_shape).astype("float32"),
            'ShapeTensor': shape_tensor,
        }
        self.attrs = {'shape': self.shape}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.inferred_shape),
            'XShape': np.random.random(self.ori_shape).astype("float32"),
        }

    def init_data(self):
        self.ori_shape = (4, 25)
        self.new_shape = (10, 10)
        self.inferred_shape = (10, 10)
        self.shape = (-1, -1)

    def test_check_output(self):
        self.check_output(
            no_check_set=['XShape'], check_pir=True, check_symbol_infer=False
        )

    def test_check_grad(self):
        self.check_grad(
            ["X"],
            "Out",
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )


class TestReshapeOpDimInfer1_attr_ShapeTensor(TestReshapeOp_attr_ShapeTensor):
    def init_data(self):
        self.ori_shape = (5, 20)
        self.new_shape = (5, -1, 20)
        self.inferred_shape = (5, -1, 20)
        self.shape = (5, -1, -1)


class TestReshapeOpDimInfer2_attr_ShapeTensor(TestReshapeOp_attr_ShapeTensor):
    def init_data(self):
        self.ori_shape = (10, 2, 6)
        self.new_shape = (10, 0, 3, -1)
        self.inferred_shape = (10, 2, 3, -1)
        self.shape = (10, 0, 3, -1)


# Situation 4: have shape(Tensor), no actual shape(Tensor)
class TestReshapeOp_attr_OnlyShape(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "reshape2"
        self.python_api = paddle.tensor.reshape
        self.public_python_api = paddle.tensor.reshape
        self.prim_op_type = "prim"
        self.python_out_sig = ['Out']

        self.inputs = {
            "X": np.random.random(self.ori_shape).astype("float32"),
            "Shape": np.array(self.new_shape, dtype="int32"),
        }
        self.attrs = {}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.inferred_shape),
            'XShape': np.random.random(self.ori_shape).astype("float32"),
        }

    def init_data(self):
        self.ori_shape = (4, 25)
        self.new_shape = (10, 10)
        self.inferred_shape = (10, 10)

    def test_check_output(self):
        self.check_output(
            no_check_set=['XShape'], check_pir=True, check_symbol_infer=False
        )

    def test_check_grad(self):
        self.check_grad(
            ["X"],
            "Out",
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )


class TestReshapeOpDimInfer1_attr_OnlyShape(TestReshapeOp_attr_OnlyShape):
    def init_data(self):
        self.ori_shape = (5, 20)
        self.new_shape = (5, -1, 10)
        self.inferred_shape = (5, -1, 10)
        self.shape = (5, -1, -1)


class TestReshapeOpDimInfer2_attr_OnlyShape(TestReshapeOp_attr_OnlyShape):
    def init_data(self):
        self.ori_shape = (10, 2, 6)
        self.new_shape = (10, 0, 3, -1)
        self.inferred_shape = (10, 2, 3, -1)
        self.shape = (10, 0, 3, -1)


# test int8 data type on CPU
class TestReshapeInt8Op(OpTest):
    def setUp(self):
        self.init_dtype()
        self.init_data()
        self.use_onednn = True
        self._cpu_only = True
        self.op_type = "reshape2"
        self.python_api = paddle.tensor.reshape
        self.python_out_sig = ['Out']
        input = np.random.randint(0, 127, self.ori_shape).astype(self.dtype)
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(input)}
        self.attrs = {
            'shape': self.new_shape,
            'use_onednn': self.use_onednn,
        }
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.inferred_shape),
            'XShape': np.random.random(self.ori_shape).astype(np.float32),
        }

    def init_dtype(self):
        self.dtype = np.int8

    def init_data(self):
        self.ori_shape = (10, 2, 6)
        self.new_shape = (10, 0, 3, -1)
        self.inferred_shape = (10, 2, 3, -1)

    def test_check_output(self):
        self.check_output_with_place(
            base.core.CPUPlace(),
            atol=1e-5,
            no_check_set=['XShape'],
            check_pir=True,
        )

    def test_check_grad(self):
        pass


# test unt8 data type on CPU
class TestReshapeUint8Op(TestReshapeInt8Op):
    def init_dtype(self):
        self.dtype = np.uint8


@skip_check_grad_ci(
    "we don't need to check grad for the bool type of reshape op"
)
class TestReshapeOpBool(TestReshapeOp):
    def setUp(self):
        self.init_data()
        self.op_type = "reshape2"
        self.python_api = paddle.tensor.reshape
        self.python_out_sig = ['Out']
        self.inputs = {
            "X": np.random.choice([True, False], size=self.ori_shape)
        }
        self.attrs = {"shape": self.new_shape}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.inferred_shape),
            'XShape': np.random.random(self.ori_shape).astype("float32"),
        }

    def test_check_grad(self):
        pass


# Test python API
class TestReshapeAPI(unittest.TestCase):
    def _set_paddle_api(self):
        self.fill_constant = paddle.tensor.fill_constant
        self.data = paddle.static.data
        self.to_tensor = paddle.to_tensor
        self._executed_api()

    def _executed_api(self):
        self.reshape = paddle.reshape

    def _test_api(self):
        paddle.enable_static()
        input = np.random.random([2, 25]).astype("float32")
        shape = [2, 5, 5]
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, paddle.static.Program()):
            positive_five = self.fill_constant([1], "int32", 5)
            x = self.data(name="x", shape=[2, 25], dtype="float32")

            actual_shape = self.data(name="shape", shape=[3], dtype="int32")

            # situation 1: have shape( list, no tensor)
            out_1 = self.reshape(x, shape)

            # situation 2: have shape(list, no tensor)
            out_2 = paddle.reshape(x, actual_shape)

            # Situation 3: have shape(list, have tensor)
            out_3 = self.reshape(x, shape=[positive_five, 10])

            # Situation 4: have shape(Tensor)
            out_4 = self.reshape(x, shape=actual_shape)

        exe = paddle.static.Executor(place=paddle.CPUPlace())
        res_1, res_2, res_3, res_4 = exe.run(
            main_prog,
            feed={"x": input, "shape": np.array([2, 5, 5]).astype("int32")},
            fetch_list=[out_1, out_2, out_3, out_4],
        )

        np.testing.assert_array_equal(res_1, input.reshape(shape))
        np.testing.assert_array_equal(res_2, input.reshape(shape))
        np.testing.assert_array_equal(res_3, input.reshape([5, 10]))
        np.testing.assert_array_equal(res_4, input.reshape(shape))

    def _test_static_dtype(self):
        places = [paddle.CPUPlace()] + (
            [get_device_place()]
            if (base.core.is_compiled_with_cuda() or is_custom_device())
            else []
        )

        dtypes = [
            'float16',
            'float32',
            'float64',
            'int16',
            'int32',
            'int64',
            'int8',
            'uint8',
            'complex64',
            'complex128',
            'bfloat16',
            'bool',
        ]
        for place in places:
            for dtype in dtypes:
                # core is not compiled with CUDA and not support the bfloat16
                if dtype == 'bfloat16' and not (
                    base.core.is_compiled_with_cuda() or is_custom_device()
                ):
                    continue

                dtype_paddle = dtype
                # numpy not support bfloat16, use uint16 instead
                dtype_numpy = dtype if dtype != 'bfloat16' else 'uint16'

                paddle.enable_static()
                input = np.random.random([2, 25]).astype(dtype_numpy)
                shape = [2, 5, 5]
                main_prog = paddle.static.Program()
                with paddle.static.program_guard(
                    main_prog, paddle.static.Program()
                ):
                    x = self.data(name="x", shape=[2, 25], dtype=dtype_paddle)
                    out_1 = self.reshape(x, shape)

                exe = paddle.static.Executor(place=place)
                res_1 = exe.run(
                    main_prog,
                    feed={"x": input},
                    fetch_list=[out_1],
                )[0]

                np.testing.assert_array_equal(res_1, input.reshape(shape))

    def test_paddle_api(self):
        self._set_paddle_api()
        self._test_api()
        self._test_static_dtype()

    def test_imperative(self):
        self._set_paddle_api()
        input = np.random.random([2, 25]).astype("float32")
        shape = [2, 5, 5]
        with base.dygraph.guard():
            x = self.to_tensor(input)
            positive_five = self.fill_constant([1], "int32", 5)

            out_1 = self.reshape(x, shape)

            out_2 = self.reshape(x, shape=[positive_five, 10])

            shape_tensor = self.to_tensor(np.array([2, 5, 5]).astype("int32"))
            out_3 = self.reshape(x, shape=shape_tensor)

        np.testing.assert_array_equal(out_1.numpy(), input.reshape(shape))
        np.testing.assert_array_equal(out_2.numpy(), input.reshape([5, 10]))
        np.testing.assert_array_equal(out_3.numpy(), input.reshape(shape))


class TestStaticReshape_(TestReshapeAPI):
    def _executed_api(self):
        self.reshape = paddle.reshape_

    def test_imperative(self):
        self._set_paddle_api()
        input = np.random.random([2, 25]).astype("float32")
        shape = [2, 5, 5]
        with base.dygraph.guard():
            x = self.to_tensor(input)
            positive_five = self.fill_constant([1], "int32", 5)

            out_1 = self.reshape(x, shape)

            out_2 = self.reshape(x, shape=[positive_five, 10])

            shape_tensor = self.to_tensor(np.array([2, 5, 5]).astype("int32"))
            out_3 = self.reshape(x, shape=shape_tensor)

        np.testing.assert_array_equal(out_1.numpy(), input.reshape(shape))
        np.testing.assert_array_equal(out_2.numpy(), input.reshape(shape))
        np.testing.assert_array_equal(out_3.numpy(), input.reshape(shape))


# Test Input Error
class TestReshapeOpError(unittest.TestCase):
    def _set_paddle_api(self):
        self.data = paddle.static.data
        self.reshape = paddle.reshape

    def _test_errors(self):
        with program_guard(Program(), Program()):
            # The x type of reshape_op must be Variable.
            def test_x_type():
                x1 = base.create_lod_tensor(
                    np.array([[-1]]), [[1]], paddle.CPUPlace()
                )
                self.reshape(x1, shape=[1])

            self.assertRaises(TypeError, test_x_type)

            def test_x_dtype_float16():
                x_float16 = self.data(
                    name="x_float16", shape=[2, 25], dtype="float16"
                )
                self.reshape(x_float16, shape=[2, 5, 5])

            test_x_dtype_float16()

            x3 = self.data(name="x3", shape=[2, 25], dtype="float32")

            # The argument shape's type of reshape_op must be list, tuple or Variable.
            def test_shape_type():
                self.reshape(x3, shape=1)

            self.assertRaises(TypeError, test_shape_type)

            # The argument shape have more than one -1.
            def test_shape_1():
                self.reshape(x3, shape=[-1, -1, 5])

            self.assertRaises(AssertionError, test_shape_1)

            # The argument shape have element 0 whose index exceed the input dimension.
            def test_shape_2():
                self.reshape(x3, [2, 5, 5, 0])

            self.assertRaises(AssertionError, test_shape_2)

            # The argument shape have more than one negative value.
            def test_shape_3():
                self.reshape(x3, [-1, -2, 5])

            self.assertRaises(AssertionError, test_shape_3)

    def test_paddle_api_error(self):
        self._set_paddle_api()
        self._test_errors()


class TestDygraphReshapeAPI(unittest.TestCase):
    def setUp(self):
        self.executed_api()

    def executed_api(self):
        self.reshape = paddle.reshape

    def test_out(self):
        paddle.disable_static()
        input_1 = np.random.random([5, 1, 10]).astype("int32")
        input = paddle.to_tensor(input_1)
        output = self.reshape(x=input, shape=[5, 10])
        out_np = output.numpy()
        expected_out = np.reshape(input_1, newshape=[5, 10])
        np.testing.assert_allclose(expected_out, out_np, rtol=1e-05)

    def test_out_uint8(self):
        paddle.disable_static()
        input_1 = np.random.random([5, 1, 10]).astype("uint8")
        input = paddle.to_tensor(input_1)
        output = self.reshape(x=input, shape=[5, 10])
        out_np = output.numpy()
        expected_out = np.reshape(input_1, newshape=[5, 10])
        np.testing.assert_allclose(expected_out, out_np, rtol=1e-05)

    def test_out_float32(self):
        paddle.disable_static()
        input_1 = np.random.random([5, 1, 10]).astype("float32")
        input = paddle.to_tensor(input_1)
        output = self.reshape(x=input, shape=[5, 10])
        out_np = output.numpy()
        expected_out = np.reshape(input_1, newshape=[5, 10])
        np.testing.assert_allclose(expected_out, out_np, rtol=1e-05)


class TestDygraphReshapeInplaceAPI(TestDygraphReshapeAPI):
    def executed_api(self):
        self.reshape = paddle.reshape_


class TestReshapeZeroTensor(unittest.TestCase):
    def test_reshape_zero_tensor_success(self):
        zero_tensor = paddle.zeros([0, 2, 3])
        # since we use "0" as the dimension copy semantically in reshape,
        # we need to copy the 0 dim in the src tensor in order to make a successful zero tensor reshape
        zero_tensor = zero_tensor.reshape([0, 6])
        self.assertTrue(list(zero_tensor.shape) == [0, 6])

    def test_reshape_zero_tensor_error(self):
        zero_tensor = paddle.zeros([0, 2, 3])
        with self.assertRaises(ValueError):
            zero_tensor.reshape([2, 3])


class TestReshapeAPI_ZeroDim(unittest.TestCase):
    def test_dygraph(self):
        with paddle.base.dygraph.guard():
            x = paddle.rand([])
            x.stop_gradient = False

            out = paddle.reshape(x, [1])
            out.retain_grads()
            out.backward()
            self.assertEqual(x.grad.shape, [])
            self.assertEqual(out.shape, [1])
            self.assertEqual(out.grad.shape, [1])

            out = paddle.reshape(x, [-1, 1])
            out.retain_grads()
            out.backward()
            self.assertEqual(x.grad.shape, [])
            self.assertEqual(out.shape, [1, 1])
            self.assertEqual(out.grad.shape, [1, 1])

            x = paddle.rand([1])
            x.stop_gradient = False
            out = paddle.reshape(x, [])
            out.retain_grads()
            out.backward()
            self.assertEqual(x.grad.shape, [1])
            self.assertEqual(out.shape, [])
            self.assertEqual(out.grad.shape, [])

    def test_static(self):
        main_prog = base.Program()
        with base.program_guard(main_prog, base.Program()):
            x = paddle.rand([])
            x.stop_gradient = False
            out = paddle.reshape(x, [-1])
            if paddle.framework.in_pir_mode():
                grads = paddle.autograd.ir_backward.grad(out, x)
                x_grad = grads[0]
                out_grad = x_grad.get_defining_op().operand_source(1)
            else:
                base.backward.append_backward(out)
                prog = paddle.static.default_main_program()
                block = prog.global_block()

                x_grad = block.var(base.framework.grad_var_name(x.name))
                out_grad = block.var(base.framework.grad_var_name(out.name))

            # Test compile shape
            self.assertEqual(tuple(x.shape), ())
            self.assertEqual(tuple(out.shape), (1,))
            self.assertEqual(tuple(x_grad.shape), ())
            self.assertEqual(tuple(out_grad.shape), (1,))

            exe = base.Executor()
            result = exe.run(main_prog, fetch_list=[x, out, x_grad, out_grad])

            # Test runtime shape
            self.assertEqual(result[0].shape, ())
            self.assertEqual(result[1].shape, (1,))
            self.assertEqual(result[2].shape, ())
            self.assertEqual(result[3].shape, (1,))


class TestReshapePirValueListShape(unittest.TestCase):
    def test_value_list_shape(self):
        with paddle.pir_utils.IrGuard():
            x = paddle.static.data('x', [3])
            shape = [1, paddle.full([], 3)]
            out = paddle.reshape(x, shape)
            self.assertEqual(out.shape, [1, -1])


class TestReshapePirTensorWithZeroShape(unittest.TestCase):
    def test_tensor_with_zero_shape(self):
        with paddle.pir_utils.IrGuard():
            x = paddle.static.data('x', [10, -1])
            shape = [0, paddle.shape(x)[1]]
            out = paddle.reshape(x, shape)
            self.assertEqual(out.shape, [10, -1])


# Test python Alias API
class TestReshapeAliasAPI(unittest.TestCase):
    def _set_paddle_api(self):
        self.fill_constant = paddle.tensor.fill_constant
        self.data = paddle.static.data
        self.to_tensor = paddle.to_tensor
        self._executed_api()

    def _executed_api(self):
        self.reshape = paddle.reshape

    def _test_api(self):
        paddle.enable_static()
        input = np.random.random([2, 25]).astype("float32")
        shape = [2, 5, 5]
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, paddle.static.Program()):
            positive_five = self.fill_constant([1], "int32", 5)
            x = self.data(name="x", shape=[2, 25], dtype="float32")
            actual_shape = self.data(name="shape", shape=[3], dtype="int32")

            out_1 = self.reshape(x, shape)
            out_2 = paddle.reshape(x, shape=actual_shape)
            out_3 = self.reshape(input=x, shape=[positive_five, 10])
            out_4 = self.reshape(input=x, shape=actual_shape)

        exe = paddle.static.Executor(place=paddle.CPUPlace())
        res_1, res_2, res_3, res_4 = exe.run(
            main_prog,
            feed={"x": input, "shape": np.array([2, 5, 5]).astype("int32")},
            fetch_list=[out_1, out_2, out_3, out_4],
        )

        np.testing.assert_array_equal(res_1, input.reshape(shape))
        np.testing.assert_array_equal(res_2, input.reshape(shape))
        np.testing.assert_array_equal(res_3, input.reshape([5, 10]))
        np.testing.assert_array_equal(res_4, input.reshape(shape))

    def _test_static_dtype(self):
        places = [paddle.CPUPlace()] + (
            [get_device_place()]
            if (base.core.is_compiled_with_cuda() or is_custom_device())
            else []
        )

        dtypes = [
            'float16',
            'float32',
            'float64',
            'int16',
            'int32',
            'int64',
            'int8',
            'uint8',
            'complex64',
            'complex128',
            'bfloat16',
            'bool',
        ]
        for place in places:
            for dtype in dtypes:
                # core is not compiled with CUDA and not support the bfloat16
                if dtype == 'bfloat16' and not (
                    base.core.is_compiled_with_cuda() or is_custom_device()
                ):
                    continue

                dtype_paddle = dtype
                # numpy not support bfloat16, use uint16 instead
                dtype_numpy = dtype if dtype != 'bfloat16' else 'uint16'

                paddle.enable_static()
                input = np.random.random([2, 25]).astype(dtype_numpy)
                shape = [2, 5, 5]
                main_prog = paddle.static.Program()
                with paddle.static.program_guard(
                    main_prog, paddle.static.Program()
                ):
                    x = self.data(name="x", shape=[2, 25], dtype=dtype_paddle)
                    out_1 = self.reshape(input=x, shape=shape)

                exe = paddle.static.Executor(place=place)
                res_1 = exe.run(
                    main_prog,
                    feed={"x": input},
                    fetch_list=[out_1],
                )[0]

                np.testing.assert_array_equal(res_1, input.reshape(shape))

    def test_paddle_api(self):
        self._set_paddle_api()
        self._test_api()
        self._test_static_dtype()

    def test_imperative(self):
        self._set_paddle_api()
        input = np.random.random([2, 25]).astype("float32")
        shape = [2, 5, 5]
        with base.dygraph.guard():
            x = self.to_tensor(input)
            positive_five = self.fill_constant([1], "int32", 5)

            out_1 = self.reshape(x, shape=shape)

            out_2 = self.reshape(input=x, shape=[positive_five, 10])

            shape_tensor = self.to_tensor(np.array([2, 5, 5]).astype("int32"))
            out_3 = self.reshape(input=x, shape=shape_tensor)

        np.testing.assert_array_equal(out_1.numpy(), input.reshape(shape))
        np.testing.assert_array_equal(out_2.numpy(), input.reshape([5, 10]))
        np.testing.assert_array_equal(out_3.numpy(), input.reshape(shape))

    def test_tensor_reshape(self):
        """The `shape` parameter accepts either variable arguments or a list/tuple.
        For example, x.reshape(2, 5, 5) is equivalent to x.reshape([2, 5, 5]).
        """

        def run_test_cases(place):
            """Helper function to run test cases on specified device."""
            input = np.random.random([2, 25]).astype("float32")
            input_tensor = paddle.to_tensor(input, place=place)

            out_1 = input_tensor.reshape([2, 5, 5])
            out_2 = input_tensor.reshape(2, 5, 5)

            np.testing.assert_array_equal(
                out_1.numpy(), input.reshape([2, 5, 5])
            )
            np.testing.assert_array_equal(
                out_2.numpy(), input.reshape([2, 5, 5])
            )

        with base.dygraph.guard():
            run_test_cases(paddle.CPUPlace())
            if paddle.base.core.is_compiled_with_cuda() or is_custom_device():
                run_test_cases(get_device_place())


class TestReshapeWithTensorShape(unittest.TestCase):
    """
    reshape supports shape like:
    paddle.reshape(x, shape=[1, 2, 3])
    paddle.reshape(x, shape=[1, Tensor(2), 3])
    paddle.reshape(x, shape=Tensor([1, 2, 3]))
    paddle.reshape(x, 1, 2, 3)  # Compatible usage
    paddle.reshape(x, 1, Tensor(2), 3)  # Compatible usage
    """

    @static_guard()
    def check_reshape_static(
        self, fn, x_shape, expected_out_shape, dynamic_dims=[]
    ):
        main_program = Program()
        with program_guard(main_program):
            x = paddle.static.data('x', shape=x_shape, dtype='float32')
            out = fn(x)
            if dynamic_dims:
                expected_out_shape_with_dynamic = list(expected_out_shape)
                for dim in dynamic_dims:
                    expected_out_shape_with_dynamic[dim] = -1
                self.assertEqual(out.shape, expected_out_shape_with_dynamic)
            else:
                self.assertEqual(out.shape, expected_out_shape)

        exe = paddle.static.Executor()
        (out_np,) = exe.run(
            main_program,
            feed={'x': np.random.random(x_shape)},
            fetch_list=[out],
        )
        self.assertEqual(list(out_np.shape), expected_out_shape)

    @dygraph_guard()
    def check_reshape_dygraph(self, fn, x_shape, expected_out_shape):
        x = paddle.to_tensor(np.random.random(x_shape).astype('float32'))
        out = fn(x)
        self.assertEqual(list(out.shape), expected_out_shape)

    def check_reshape(self, fn, x_shape, expected_out_shape):
        self.check_reshape_static(fn, x_shape, expected_out_shape)
        self.check_reshape_dygraph(fn, x_shape, expected_out_shape)

    def test_reshape_with_list_int(self):
        def reshape_fn(x):
            return paddle.reshape(x, shape=[2, 3, 4])

        self.check_reshape(reshape_fn, [2, 12], [2, 3, 4])

    def test_reshape_with_list_scalar_tensor(self):
        def reshape_fn(x):
            dim0 = paddle.full([], 2, dtype='int64')
            dim1 = paddle.full([], 3, dtype='int64')
            dim2 = paddle.full([], 4, dtype='int64')
            return paddle.reshape(x, shape=[dim0, dim1, dim2])

        self.check_reshape(reshape_fn, [2, 12], [2, 3, 4])

    def test_reshape_with_list_scalar_tensor_dynamic_dim(self):
        def reshape_fn(x):
            dim0 = paddle.full([], 1, dtype='int64') + 1  # dynamic dim
            dim1 = paddle.full([], 3, dtype='int64')
            dim2 = paddle.full([], 4, dtype='int64')
            return paddle.reshape(x, shape=[dim0, dim1, dim2])

        self.check_reshape_static(
            reshape_fn,
            x_shape=[2, 12],
            expected_out_shape=[2, 3, 4],
            dynamic_dims=[0],
        )

    def test_reshape_with_list_mix_int_tensor(self):
        def reshape_fn(x):
            dim1 = paddle.full([], 3, dtype='int64')
            return paddle.reshape(x, shape=[2, dim1, 4])

        self.check_reshape(reshape_fn, [2, 12], [2, 3, 4])

    def test_reshape_with_tensor_dynamic_dim(self):
        def reshape_fn(x):
            shape_tensor = paddle.to_tensor([1, 2, 3]) + 1  # all dynamic dims
            return paddle.reshape(x, shape=shape_tensor)

        self.check_reshape_static(
            reshape_fn,
            x_shape=[2, 12],
            expected_out_shape=[2, 3, 4],
            dynamic_dims=[0, 1, 2],
        )

    def test_reshape_with_tensor(self):
        def reshape_fn(x):
            shape_tensor = paddle.stack(
                [
                    paddle.full([], 2, dtype='int64'),
                    paddle.full([], 3, dtype='int64'),
                    paddle.full([], 4, dtype='int64'),
                ]
            )
            return paddle.reshape(x, shape=shape_tensor)

        self.check_reshape(reshape_fn, [2, 12], [2, 3, 4])

    def test_reshape_with_list_int_compatible(self):
        def reshape_fn(x):
            return paddle.reshape(x, 2, 3, 4)

        self.check_reshape(reshape_fn, [2, 12], [2, 3, 4])

    def test_reshape_with_list_scalar_tensor_compatible(self):
        def reshape_fn(x):
            dim0 = paddle.full([], 2, dtype='int64')
            dim1 = paddle.full([], 3, dtype='int64')
            dim2 = paddle.full([], 4, dtype='int64')
            return paddle.reshape(x, dim0, dim1, dim2)

        self.check_reshape(reshape_fn, [2, 12], [2, 3, 4])

    def test_reshape_with_list_mix_int_tensor_compatible(self):
        def reshape_fn(x):
            dim1 = paddle.full([], 3, dtype='int64')
            return paddle.reshape(x, 2, dim1, 4)

        self.check_reshape(reshape_fn, [2, 12], [2, 3, 4])


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
