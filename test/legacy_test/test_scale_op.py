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

import gradient_checker
import numpy as np
from decorator_helper import prog_scope
from op import Operator
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle import base
from paddle.base import core


class TestScaleOp(OpTest):
    def setUp(self):
        self.op_type = "scale"
        self.python_api = paddle.scale
        self.dtype = np.float32
        self.init_dtype_type()
        self.public_python_api = paddle.scale
        self.prim_op_type = "prim"
        self.inputs = {'X': np.random.random((10, 10)).astype(self.dtype)}
        self.attrs = {'scale': -2.3}
        self.outputs = {
            'Out': self.inputs['X'] * self.dtype(self.attrs['scale'])
        }

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output(check_cinn=True, check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestScaleOpFP64(TestScaleOp):
    def init_dtype_type(self):
        self.dtype = np.float64
        # NOTE(dev): Scalar.to<float> has diff with double.
        self.rev_comp_atol = 1e-7


class TestScaleOpScaleVariable(OpTest):
    def setUp(self):
        self.op_type = "scale"
        self.python_api = paddle.scale
        self.dtype = np.float64
        self.init_dtype_type()
        self.scale = -2.3
        self.inputs = {
            'X': np.random.random((10, 10)).astype(self.dtype),
            'ScaleTensor': np.array([self.scale]).astype('float64'),
        }
        self.attrs = {}
        self.outputs = {'Out': self.inputs['X'] * self.dtype(self.scale)}

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output(check_cinn=True, check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True)


class TestScaleOpSelectedRows(unittest.TestCase):
    def init_dtype_type(self):
        pass

    def check_with_place(self, place, in_name, out_name):
        scope = core.Scope()

        self.dtype = np.float64
        self.init_dtype_type()

        # create and initialize Grad Variable
        in_height = 10
        in_rows = [0, 4, 7]
        in_row_numel = 12
        scale = 2.0

        in_selected_rows = scope.var(in_name).get_selected_rows()
        in_selected_rows.set_height(in_height)
        in_selected_rows.set_rows(in_rows)
        in_array = np.random.random((len(in_rows), in_row_numel)).astype(
            self.dtype
        )

        in_tensor = in_selected_rows.get_tensor()
        in_tensor.set(in_array, place)

        # create and initialize Param Variable
        out_selected_rows = scope.var(out_name).get_selected_rows()
        out_tensor = out_selected_rows.get_tensor()
        out_tensor._set_dims(in_tensor._get_dims())

        # create and run sgd operator
        scale_op = Operator("scale", X=in_name, Out=out_name, scale=scale)
        scale_op.run(scope, place)

        # get and compare result
        out_height = out_selected_rows.height()
        out_rows = out_selected_rows.rows()
        result_array = np.array(out_tensor)

        assert (in_array * scale == result_array).all()
        assert in_height == out_height
        assert in_rows == out_rows

    def test_scale_selected_rows(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(core.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_with_place(place, 'in', 'out')

    def test_scale_selected_rows_inplace(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(core.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_with_place(place, 'in', 'in')


class TestScaleRaiseError(unittest.TestCase):

    def test_errors(self):
        paddle.enable_static()

        def test_type():
            paddle.scale([10])

        self.assertRaises(TypeError, test_type)


# Add FP16 test
@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestScaleFp16Op(TestScaleOp):
    def init_dtype_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output(check_cinn=True, check_pir=True)

    def test_check_grad(self):
        self.check_grad(["X"], "Out", check_pir=True, check_prim_pir=True)


@unittest.skipIf(
    not paddle.is_compiled_with_cuda() or paddle.is_compiled_with_rocm(),
    "BFP16 test runs only on CUDA",
)
class TestScaleBF16Op(OpTest):
    def setUp(self):
        self.op_type = "scale"
        self.python_api = paddle.scale
        self.public_python_api = paddle.scale
        self.prim_op_type = "prim"
        self.dtype = np.uint16
        self.attrs = {'scale': -2.3}
        x = np.random.random((10, 10)).astype(np.float32)
        out = x * np.float32(self.attrs['scale'])
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            numeric_grad_delta=0.8,
            check_pir=True,
            check_prim_pir=True,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestScaleFp16OpSelectedRows(TestScaleOpSelectedRows):
    def init_dtype_type(self):
        self.dtype = np.float16

    def test_scale_selected_rows(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_with_place(place, 'in', 'out')

    def test_scale_selected_rows_inplace(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_with_place(place, 'in', 'in')


class TestScaleApiStatic(unittest.TestCase):
    def _executed_api(self, x, scale=1.0, bias=0.0):
        return paddle.scale(x, scale, bias)

    def test_api(self):
        paddle.enable_static()
        input = np.random.random([2, 25]).astype("float32")
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, paddle.static.Program()):
            x = paddle.static.data(name="x", shape=[2, 25], dtype="float32")
            out = self._executed_api(x, scale=2.0, bias=3.0)

        exe = paddle.static.Executor(place=paddle.CPUPlace())
        out = exe.run(main_prog, feed={"x": input}, fetch_list=[out])
        np.testing.assert_array_equal(out[0], input * 2.0 + 3.0)


class TestScaleInplaceApiStatic(TestScaleApiStatic):
    def _executed_api(self, x, scale=1.0, bias=0.0):
        return x.scale_(scale, bias)


class TestScaleApiDygraph(unittest.TestCase):
    def _executed_api(self, x, scale=1.0, bias=0.0):
        return paddle.scale(x, scale, bias)

    def test_api(self):
        paddle.disable_static()
        input = np.random.random([2, 25]).astype("float32")
        x = paddle.to_tensor(input)
        out = self._executed_api(x, scale=2.0, bias=3.0)
        np.testing.assert_array_equal(out.numpy(), input * 2.0 + 3.0)
        paddle.enable_static()


class TestScaleInplaceApiDygraph(TestScaleApiDygraph):
    def _executed_api(self, x, scale=1.0, bias=0.0):
        return x.scale_(scale, bias)


class TestScaleDoubleGradCheck(unittest.TestCase):
    def scale_wrapper(self, x):
        return paddle.scale(x[0], scale=2.0)

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not include -1.
        eps = 0.005
        dtype = np.float32

        data = paddle.static.data('data', [2, 3], dtype)
        data.persistable = True
        out = paddle.scale(data, 2.0)
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

        gradient_checker.double_grad_check(
            [data], out, x_init=[data_arr], place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.scale_wrapper, [data], out, x_init=[data_arr], place=place
        )

    def test_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestScaleTripleGradCheck(unittest.TestCase):
    def scale_wrapper(self, x):
        return paddle.scale(x[0], scale=2.0)

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not include -1.
        eps = 0.005
        dtype = np.float32

        data = paddle.static.data('data', [2, 3], dtype)
        data.persistable = True
        out = paddle.scale(data, 2.0)
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

        gradient_checker.triple_grad_check(
            [data], out, x_init=[data_arr], place=place, eps=eps
        )
        gradient_checker.triple_grad_check_for_dygraph(
            self.scale_wrapper, [data], out, x_init=[data_arr], place=place
        )

    def test_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestScaleOpZeroNumelVariable(unittest.TestCase):
    def test_check_zero_numel_cpu(self):
        with paddle.pir_utils.OldIrGuard():
            paddle.set_device('cpu')
            data = paddle.ones([0, 1])
            out = paddle.scale(data, 2)
            self.assertEqual(out, data)

            if paddle.is_compiled_with_cuda():
                paddle.set_device('gpu')
                data = paddle.ones([0, 1])
                out = paddle.scale(data, 2)
                self.assertEqual(out, data)


if __name__ == "__main__":
    unittest.main()
