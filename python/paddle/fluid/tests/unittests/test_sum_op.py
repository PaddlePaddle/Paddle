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

import os
import unittest
import tempfile
import numpy as np
from op_test import OpTest
import paddle
from paddle import enable_static
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.fluid.tests.unittests.op_test import (OpTest,
                                                  convert_float_to_uint16,
                                                  convert_uint16_to_float)
from paddle import _C_ops, _legacy_C_ops
from paddle.fluid.framework import _test_eager_guard
import paddle.inference as paddle_infer
import gradient_checker
from decorator_helper import prog_scope
import paddle.fluid.layers as layers


class TestSumOp(OpTest):

    def setUp(self):
        self.op_type = "sum"
        self.init_kernel_type()
        self.use_mkldnn = False
        self.init_kernel_type()
        x0 = np.random.random((3, 40)).astype(self.dtype)
        x1 = np.random.random((3, 40)).astype(self.dtype)
        x2 = np.random.random((3, 40)).astype(self.dtype)
        self.inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)]}
        y = x0 + x1 + x2
        self.outputs = {'Out': y}
        self.attrs = {'use_mkldnn': self.use_mkldnn}

    def init_kernel_type(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out')


class TestSelectedRowsSumOp(unittest.TestCase):

    def setUp(self):
        self.height = 10
        self.row_numel = 12
        self.rows = [0, 1, 2, 3, 4, 5, 6]
        self.dtype = np.float64
        self.init_kernel_type()

    def check_with_place(self, place, inplace):
        self.check_input_and_optput(core.Scope(), place, inplace, True, True,
                                    True)
        self.check_input_and_optput(core.Scope(), place, inplace, False, True,
                                    True)
        self.check_input_and_optput(core.Scope(), place, inplace, False, False,
                                    True)
        self.check_input_and_optput(core.Scope(), place, inplace, False, False,
                                    False)

    def init_kernel_type(self):
        pass

    def _get_array(self, rows, row_numel):
        array = np.ones((len(rows), row_numel)).astype(self.dtype)
        for i in range(len(rows)):
            array[i] *= rows[i]
        return array

    def check_input_and_optput(self,
                               scope,
                               place,
                               inplace,
                               w1_has_data=False,
                               w2_has_data=False,
                               w3_has_data=False):

        self.create_selected_rows(scope, place, "W1", w1_has_data)
        self.create_selected_rows(scope, place, "W2", w2_has_data)
        self.create_selected_rows(scope, place, "W3", w3_has_data)

        # create Out Variable
        if inplace:
            out_var_name = "W1"
        else:
            out_var_name = "Out"
        out = scope.var(out_var_name).get_selected_rows()

        # create and run sum operator
        sum_op = Operator("sum", X=["W1", "W2", "W3"], Out=out_var_name)
        sum_op.run(scope, place)

        has_data_w_num = 0
        for has_data in [w1_has_data, w2_has_data, w3_has_data]:
            if has_data:
                has_data_w_num += 1

        if has_data_w_num > 0:
            self.assertEqual(len(out.rows()), 7)
            np.testing.assert_array_equal(
                np.array(out.get_tensor()),
                self._get_array(self.rows, self.row_numel) * has_data_w_num)
        else:
            self.assertEqual(len(out.rows()), 0)

    def create_selected_rows(self, scope, place, var_name, has_data):
        # create and initialize W Variable
        if has_data:
            rows = self.rows
        else:
            rows = []

        var = scope.var(var_name)
        w_selected_rows = var.get_selected_rows()
        w_selected_rows.set_height(self.height)
        w_selected_rows.set_rows(rows)
        w_array = self._get_array(self.rows, self.row_numel)
        w_tensor = w_selected_rows.get_tensor()
        w_tensor.set(w_array, place)

        return var

    def test_w_is_selected_rows(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            for inplace in [True, False]:
                self.check_with_place(place, inplace)


class TestSelectedRowsSumOpInt(TestSelectedRowsSumOp):

    def init_kernel_type(self):
        self.dtype = np.int32


@unittest.skipIf(not core.supports_bfloat16(),
                 'place does not support BF16 evaluation')
class TestSelectedRowsSumBF16Op(TestSelectedRowsSumOp):

    def setUp(self):
        self.height = 10
        self.row_numel = 12
        self.rows = [0, 1, 2, 3, 4, 5, 6]
        self.dtype = np.uint16
        self.init_kernel_type()
        np.random.seed(12345)
        self.data = np.random.random(
            (len(self.rows), self.row_numel)).astype(np.float32)

    def _get_array(self, rows, row_numel):
        if len(rows) > 0:
            return convert_float_to_uint16(self.data)
        else:
            return np.ndarray((0, row_numel), dtype=self.dtype)

    def check_input_and_optput(self,
                               scope,
                               place,
                               inplace,
                               w1_has_data=False,
                               w2_has_data=False,
                               w3_has_data=False):

        self.create_selected_rows(scope, place, "W1", w1_has_data)
        self.create_selected_rows(scope, place, "W2", w2_has_data)
        self.create_selected_rows(scope, place, "W3", w3_has_data)

        # create Out Variable
        if inplace:
            out_var_name = "W1"
        else:
            out_var_name = "Out"
        out = scope.var(out_var_name).get_selected_rows()

        # create and run sum operator
        sum_op = Operator("sum", X=["W1", "W2", "W3"], Out=out_var_name)
        sum_op.run(scope, place)

        has_data_w_num = 0
        for has_data in [w1_has_data, w2_has_data, w3_has_data]:
            if has_data:
                has_data_w_num += 1

        if has_data_w_num > 0:
            self.assertEqual(len(out.rows()), 7)
            out_bf16 = np.array(out.get_tensor())
            out_fp32 = convert_uint16_to_float(out_bf16)
            ref_fp32 = convert_uint16_to_float(
                self._get_array(self.rows, self.row_numel)) * has_data_w_num
            np.testing.assert_allclose(out_fp32, ref_fp32, atol=0, rtol=0.95e-2)
        else:
            self.assertEqual(len(out.rows()), 0)

    def test_w_is_selected_rows(self):
        for inplace in [True, False]:
            self.check_with_place(core.CPUPlace(), inplace)


class TestSelectedRowsSumBF16OpBigRow(TestSelectedRowsSumBF16Op):

    def init_kernel_type(self):
        self.row_numel = 102


class TestLoDTensorAndSelectedRowsOp(TestSelectedRowsSumOp):

    def setUp(self):
        self.height = 10
        self.row_numel = 12
        self.rows = [0, 1, 2, 2, 4, 5, 6]
        self.dtype = np.float64

    def check_with_place(self, place, inplace):
        scope = core.Scope()
        if inplace:
            self.create_lod_tensor(scope, place, "x1")
            self.create_selected_rows(scope, place, "x2", True)
            out = scope.var("x1").get_tensor()
            out_name = "x1"
        else:
            self.create_selected_rows(scope, place, "x1", True)
            self.create_lod_tensor(scope, place, "x2")
            out = scope.var("out").get_tensor()
            out_name = "out"

        # create and run sum operator
        sum_op = Operator("sum", X=["x1", "x2"], Out=out_name)
        sum_op.run(scope, place)

        result = np.ones((1, self.height)).astype(np.int32).tolist()[0]
        for ele in self.rows:
            result[ele] += 1

        out_t = np.array(out)
        self.assertEqual(out_t.shape[0], self.height)
        np.testing.assert_array_equal(
            out_t,
            self._get_array([i for i in range(self.height)], self.row_numel) *
            np.tile(np.array(result).reshape(self.height, 1), self.row_numel))

    def create_lod_tensor(self, scope, place, var_name):
        var = scope.var(var_name)
        w_tensor = var.get_tensor()
        w_array = self._get_array([i for i in range(self.height)],
                                  self.row_numel)
        w_tensor.set(w_array, place)
        return var


#----------- test fp16 -----------
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFP16SumOp(TestSumOp):

    def init_kernel_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_output_with_place(place, atol=2e-2)

    # FIXME: Because of the precision fp16, max_relative_error
    # should be 0.15 here.
    def test_check_grad(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_grad(['x0'], 'Out', max_relative_error=0.15)


def create_test_sum_fp16_class(parent):

    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestSumFp16Case(parent):

        def init_kernel_type(self):
            self.dtype = np.float16

        def test_w_is_selected_rows(self):
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                for inplace in [True, False]:
                    self.check_with_place(place, inplace)

    cls_name = "{0}_{1}".format(parent.__name__, "SumFp16Test")
    TestSumFp16Case.__name__ = cls_name
    globals()[cls_name] = TestSumFp16Case


#----------- test bf16 -----------
class TestSumBF16Op(OpTest):

    def setUp(self):
        self.op_type = "sum"
        self.init_kernel_type()
        x0 = np.random.random((3, 40)).astype(np.float32)
        x1 = np.random.random((3, 40)).astype(np.float32)
        x2 = np.random.random((3, 40)).astype(np.float32)
        y = x0 + x1 + x2
        self.inputs = {
            "X": [("x0", convert_float_to_uint16(x0)),
                  ("x1", convert_float_to_uint16(x1)),
                  ("x2", convert_float_to_uint16(x2))]
        }
        self.outputs = {'Out': convert_float_to_uint16(y)}

    def init_kernel_type(self):
        self.dtype = np.uint16

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out', numeric_grad_delta=0.5)


class API_Test_Add_n(unittest.TestCase):

    def test_api(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input0 = fluid.layers.fill_constant(shape=[2, 3],
                                                dtype='int64',
                                                value=5)
            input1 = fluid.layers.fill_constant(shape=[2, 3],
                                                dtype='int64',
                                                value=3)
            expected_result = np.empty((2, 3))
            expected_result.fill(8)
            sum_value = paddle.add_n([input0, input1])
            exe = fluid.Executor(fluid.CPUPlace())
            result = exe.run(fetch_list=[sum_value])

            self.assertEqual((result == expected_result).all(), True)

        with fluid.dygraph.guard():
            input0 = paddle.ones(shape=[2, 3], dtype='float32')
            expected_result = np.empty((2, 3))
            expected_result.fill(2)
            sum_value = paddle.add_n([input0, input0])

            self.assertEqual((sum_value.numpy() == expected_result).all(), True)

    def test_dygraph_api(self):
        with fluid.dygraph.guard():
            with _test_eager_guard():
                input0 = paddle.ones(shape=[2, 3], dtype='float32')
                input1 = paddle.ones(shape=[2, 3], dtype='float32')
                input0.stop_gradient = False
                input1.stop_gradient = False
                expected_result = np.empty((2, 3))
                expected_result.fill(2)
                sum_value = paddle.add_n([input0, input1])
                self.assertEqual((sum_value.numpy() == expected_result).all(),
                                 True)

                expected_grad_result = np.empty((2, 3))
                expected_grad_result.fill(1)
                sum_value.backward()
                self.assertEqual(
                    (input0.grad.numpy() == expected_grad_result).all(), True)
                self.assertEqual(
                    (input1.grad.numpy() == expected_grad_result).all(), True)

    def test_add_n_and_add_and_grad(self):
        with fluid.dygraph.guard():
            np_x = np.array([[1, 2, 3], [4, 5, 6]])
            np_y = [[7, 8, 9], [10, 11, 12]]
            np_z = [[1, 1, 1], [1, 1, 1]]
            x = paddle.to_tensor(np_x, dtype='float32', stop_gradient=False)
            y = paddle.to_tensor(np_y, dtype='float32', stop_gradient=False)
            z = paddle.to_tensor(np_z, dtype='float32')

            out1 = x + z
            out2 = y + z
            out = paddle.add_n([out1, out2])

            dx, dy = paddle.grad([out], [x, y], create_graph=True)

            expected_out = np.array([[10., 12., 14.], [16., 18., 20.]])
            expected_dx = np.array([[1, 1, 1], [1, 1, 1]])
            expected_dy = np.array([[1, 1, 1], [1, 1, 1]])

            np.testing.assert_allclose(out, expected_out, rtol=1e-05)
            np.testing.assert_allclose(dx, expected_dx, rtol=1e-05)
            np.testing.assert_allclose(dy, expected_dy, rtol=1e-05)


class TestRaiseSumError(unittest.TestCase):

    def test_errors(self):

        def test_type():
            fluid.layers.sum([11, 22])

        self.assertRaises(TypeError, test_type)

        def test_dtype():
            data1 = fluid.data(name="input1", shape=[10], dtype="int8")
            data2 = fluid.data(name="input2", shape=[10], dtype="int8")
            fluid.layers.sum([data1, data2])

        self.assertRaises(TypeError, test_dtype)

        def test_dtype1():
            data1 = fluid.data(name="input1", shape=[10], dtype="int8")
            fluid.layers.sum(data1)

        self.assertRaises(TypeError, test_dtype1)


class TestRaiseSumsError(unittest.TestCase):

    def test_errors(self):

        def test_type():
            fluid.layers.sums([11, 22])

        self.assertRaises(TypeError, test_type)

        def test_dtype():
            data1 = fluid.data(name="input1", shape=[10], dtype="int8")
            data2 = fluid.data(name="input2", shape=[10], dtype="int8")
            fluid.layers.sums([data1, data2])

        self.assertRaises(TypeError, test_dtype)

        def test_dtype1():
            data1 = fluid.data(name="input1", shape=[10], dtype="int8")
            fluid.layers.sums(data1)

        self.assertRaises(TypeError, test_dtype1)

        def test_out_type():
            data1 = fluid.data(name="input1", shape=[10], dtype="flaot32")
            data2 = fluid.data(name="input2", shape=[10], dtype="float32")
            fluid.layers.sums([data1, data2], out=[10])

        self.assertRaises(TypeError, test_out_type)

        def test_out_dtype():
            data1 = fluid.data(name="input1", shape=[10], dtype="flaot32")
            data2 = fluid.data(name="input2", shape=[10], dtype="float32")
            out = fluid.data(name="out", shape=[10], dtype="int8")
            fluid.layers.sums([data1, data2], out=out)

        self.assertRaises(TypeError, test_out_dtype)


class TestSumOpError(unittest.TestCase):

    def test_errors(self):

        def test_empty_list_input():
            with fluid.dygraph.guard():
                fluid._legacy_C_ops.sum([])

        def test_list_of_none_input():
            with fluid.dygraph.guard():
                fluid._legacy_C_ops.sum([None])

        self.assertRaises(Exception, test_empty_list_input)
        self.assertRaises(Exception, test_list_of_none_input)


create_test_sum_fp16_class(TestSelectedRowsSumOp)
create_test_sum_fp16_class(TestLoDTensorAndSelectedRowsOp)


class TestReduceOPTensorAxisBase(unittest.TestCase):

    def setUp(self):
        paddle.disable_static()
        paddle.seed(2022)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.save_path = os.path.join(self.temp_dir.name, 'reduce_tensor_axis')
        self.place = paddle.CUDAPlace(
            0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
        self.keepdim = False
        self.init_data()

    def tearDwon(self):
        self.temp_dir.cleanup()

    def init_data(self):
        self.pd_api = paddle.sum
        self.np_api = np.sum
        self.x = paddle.randn([10, 5, 9, 9], dtype='float64')
        self.np_axis = np.array((1, 2), dtype='int64')
        self.tensor_axis = paddle.to_tensor(self.np_axis, dtype='int64')

    def test_dygraph(self):
        self.x.stop_gradient = False
        pd_out = self.pd_api(self.x, self.tensor_axis)
        np_out = self.np_api(self.x.numpy(), tuple(self.np_axis))
        np.testing.assert_allclose(
            pd_out.numpy() if pd_out.size > 1 else pd_out.item(), np_out)
        pd_out.backward()
        self.assertEqual(self.x.gradient().shape, tuple(self.x.shape))

    def test_static_and_infer(self):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        starup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, starup_prog):
            # run static
            x = paddle.static.data(shape=self.x.shape,
                                   name='x',
                                   dtype='float32')
            if isinstance(self.tensor_axis, paddle.Tensor):
                axis = paddle.assign(self.np_axis)
            else:
                axis = []
                for i, item in enumerate(self.tensor_axis):
                    if isinstance(item, int):
                        axis.append(item)
                    else:
                        axis.append(paddle.full([1], self.np_axis[i], 'int64'))

            linear = paddle.nn.Linear(x.shape[-1], 5)
            linear_out = linear(x)
            out = self.pd_api(linear_out, axis, keepdim=self.keepdim)

            sgd = paddle.optimizer.SGD(learning_rate=0.)
            sgd.minimize(paddle.mean(out))
            exe = paddle.static.Executor(self.place)
            exe.run(starup_prog)
            static_out = exe.run(feed={'x': self.x.numpy().astype('float32')},
                                 fetch_list=[out])

            # run infer
            paddle.static.save_inference_model(self.save_path, [x], [out], exe)
            config = paddle_infer.Config(self.save_path + '.pdmodel',
                                         self.save_path + '.pdiparams')
            if paddle.is_compiled_with_cuda():
                config.enable_use_gpu(100, 0)
            else:
                config.disable_gpu()
            predictor = paddle_infer.create_predictor(config)
            input_names = predictor.get_input_names()
            input_handle = predictor.get_input_handle(input_names[0])
            fake_input = self.x.numpy().astype('float32')
            input_handle.reshape(self.x.shape)
            input_handle.copy_from_cpu(fake_input)
            predictor.run()
            output_names = predictor.get_output_names()
            output_handle = predictor.get_output_handle(output_names[0])
            infer_out = output_handle.copy_to_cpu()
            np.testing.assert_allclose(static_out[0], infer_out)


class TestSumWithTensorAxis1(TestReduceOPTensorAxisBase):

    def init_data(self):
        self.pd_api = paddle.sum
        self.np_api = np.sum
        self.x = paddle.randn([10, 5, 9, 9], dtype='float64')
        self.np_axis = np.array([0, 1, 2], dtype='int64')
        self.tensor_axis = [
            0,
            paddle.to_tensor([1], 'int64'),
            paddle.to_tensor([2], 'int64')
        ]


class TestAddNDoubleGradCheck(unittest.TestCase):

    def add_n_wrapper(self, x):
        return paddle.add_n(x)

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        eps = 0.005
        dtype = np.float32

        data1 = layers.data('data1', [3, 4, 5], False, dtype)
        data1.persistable = True
        data2 = layers.data('data2', [3, 4, 5], False, dtype)
        data2.persistable = True
        out = paddle.add_n([data1, data2])
        data1_arr = np.random.uniform(-1, 1, data1.shape).astype(dtype)
        data2_arr = np.random.uniform(-1, 1, data1.shape).astype(dtype)

        gradient_checker.double_grad_check([data1, data2],
                                           out,
                                           x_init=[data1_arr, data2_arr],
                                           place=place,
                                           eps=eps)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        gradient_checker.double_grad_check_for_dygraph(
            self.add_n_wrapper, [data1, data2],
            out,
            x_init=[data1_arr, data2_arr],
            place=place)

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestAddNTripleGradCheck(unittest.TestCase):

    def add_n_wrapper(self, x):
        return paddle.add_n(x)

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        eps = 0.005
        dtype = np.float32

        data1 = layers.data('data1', [3, 4, 5], False, dtype)
        data1.persistable = True
        data2 = layers.data('data2', [3, 4, 5], False, dtype)
        data2.persistable = True
        out = paddle.add_n([data1, data2])
        data1_arr = np.random.uniform(-1, 1, data1.shape).astype(dtype)
        data2_arr = np.random.uniform(-1, 1, data1.shape).astype(dtype)

        gradient_checker.triple_grad_check([data1, data2],
                                           out,
                                           x_init=[data1_arr, data2_arr],
                                           place=place,
                                           eps=eps)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        gradient_checker.triple_grad_check_for_dygraph(
            self.add_n_wrapper, [data1, data2],
            out,
            x_init=[data1_arr, data2_arr],
            place=place)

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestSumDoubleGradCheck(unittest.TestCase):

    def sum_wrapper(self, x):
        return paddle.sum(x[0], axis=1, keepdim=True)

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        eps = 0.005
        dtype = np.float32

        data = layers.data('data', [2, 4], False, dtype)
        data.persistable = True
        out = paddle.sum(data, axis=1, keepdim=True)
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

        gradient_checker.double_grad_check([data],
                                           out,
                                           x_init=[data_arr],
                                           place=place,
                                           eps=eps)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        gradient_checker.double_grad_check_for_dygraph(self.sum_wrapper, [data],
                                                       out,
                                                       x_init=[data_arr],
                                                       place=place)

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestSumTripleGradCheck(unittest.TestCase):

    def sum_wrapper(self, x):
        return paddle.sum(x[0], axis=1, keepdim=True)

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        eps = 0.005
        dtype = np.float32

        data = layers.data('data', [2, 4], False, dtype)
        data.persistable = True
        out = paddle.sum(data, axis=1, keepdim=True)
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

        gradient_checker.triple_grad_check([data],
                                           out,
                                           x_init=[data_arr],
                                           place=place,
                                           eps=eps)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        gradient_checker.triple_grad_check_for_dygraph(self.sum_wrapper, [data],
                                                       out,
                                                       x_init=[data_arr],
                                                       place=place)

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


if __name__ == "__main__":
    enable_static()
    unittest.main()
