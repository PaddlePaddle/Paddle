#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, get_device_place

import paddle
from paddle.pir_utils import OldIrGuard


def ref_var(x, axis=None, unbiased=True, keepdim=False):
    ddof = 1 if unbiased else 0
    if isinstance(axis, int):
        axis = (axis,)
    if axis is not None:
        axis = tuple(axis)
    return np.var(x, axis=axis, ddof=ddof, keepdims=keepdim)


class TestVarAPI(unittest.TestCase):
    def setUp(self):
        self.dtype = 'float64'
        self.shape = [1, 3, 4, 10]
        self.axis = [1, 3]
        self.keepdim = False
        self.unbiased = True
        self.set_attrs()
        self.x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.place = get_device_place()

    def set_attrs(self):
        pass

    def static(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape, self.dtype)
            out = paddle.var(x, self.axis, self.unbiased, self.keepdim)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x}, fetch_list=[out])
        return res[0]

    def dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        out = paddle.var(x, self.axis, self.unbiased, self.keepdim)
        paddle.enable_static()
        return out.numpy()

    def test_api(self):
        out_ref = ref_var(self.x, self.axis, self.unbiased, self.keepdim)
        out_dygraph = self.dygraph()

        np.testing.assert_allclose(out_ref, out_dygraph, rtol=1e-05)
        self.assertTrue(np.equal(out_ref.shape, out_dygraph.shape).all())

        def test_static_or_pir_mode():
            out_static = self.static()
            np.testing.assert_allclose(out_ref, out_static, rtol=1e-05)
            self.assertTrue(np.equal(out_ref.shape, out_static.shape).all())

        test_static_or_pir_mode()


class TestVarAPI2(OpTest):
    def setUp(self):
        self.python_api = paddle.var
        self.op_type = "var"
        self.prim_op_type = "prim"
        self.init_dtype_type()
        self.attrs = {
            'axis': self.axis,
            'unbiased': self.unbiased,
            'keepdim': self.keepdim,
        }
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = ref_var(
            x, axis=self.axis, unbiased=self.unbiased, keepdim=self.keepdim
        )
        self.inputs = {'x': x}
        self.outputs = {'out': out}

        def var_wrapper(x):
            return paddle.var(
                x, axis=self.axis, unbiased=self.unbiased, keepdim=self.keepdim
            )

        self.python_api = var_wrapper
        self.public_python_api = var_wrapper

    def init_dtype_type(self):
        self.dtype = 'float64'
        self.shape = [1, 3, 4, 10]
        self.axis = [1, 3]
        self.keepdim = False
        self.unbiased = True

    def test_check_output(self):
        self.check_output_with_place(
            paddle.CPUPlace(),
            check_prim=True,
            check_pir=True,
            check_symbol_infer=True,
            check_prim_pir=True,
        )
        if paddle.is_compiled_with_cuda():
            self.check_output_with_place(
                paddle.CUDAPlace(0),
                check_prim=True,
                check_pir=True,
                check_symbol_infer=True,
                check_prim_pir=True,
            )

    def test_check_grad_normal(self):
        self.check_grad_with_place(
            paddle.CPUPlace(),
            ['x'],
            'out',
            check_prim=False,
            check_pir=True,
            check_prim_pir=False,
        )
        if paddle.core.is_compiled_with_cuda():
            self.check_grad_with_place(
                paddle.CUDAPlace(0),
                ['x'],
                'out',
                check_prim=False,
                check_pir=True,
                check_prim_pir=False,
            )


class TestVarAPI_dtype(TestVarAPI):
    def set_attrs(self):
        self.dtype = 'float32'


class TestVarAPI_axis_int(TestVarAPI):
    def set_attrs(self):
        self.axis = 2


class TestVarAPI_axis_list(TestVarAPI):
    def set_attrs(self):
        self.axis = [1, 2]


class TestVarAPI_axis_tuple(TestVarAPI):
    def set_attrs(self):
        self.axis = (1, 3)


class TestVarAPI_keepdim(TestVarAPI):
    def set_attrs(self):
        self.keepdim = False


class TestVarAPI_unbiased(TestVarAPI):
    def set_attrs(self):
        self.unbiased = False


class TestVarAPI_alias(unittest.TestCase):
    def test_alias(self):
        paddle.disable_static()
        x = paddle.to_tensor(np.array([10, 12], 'float32'))
        out1 = paddle.var(x).numpy()
        out2 = paddle.tensor.var(x).numpy()
        out3 = paddle.tensor.stat.var(x).numpy()
        np.testing.assert_allclose(out1, out2, rtol=1e-05)
        np.testing.assert_allclose(out1, out3, rtol=1e-05)
        paddle.enable_static()


class TestVarError(unittest.TestCase):
    def test_error(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', [2, 3, 4], 'int32')
            self.assertRaises(TypeError, paddle.var, x)


class TestVarAPI_ZeroSize(unittest.TestCase):
    def init_data(self):
        self.x_shape = [10, 0]

    def test_zerosize(self):
        self.init_data()
        paddle.disable_static()
        x = paddle.to_tensor(np.random.random(self.x_shape))
        out1 = paddle.var(x).numpy()
        out2 = np.var(x.numpy())
        np.testing.assert_allclose(out1, out2, equal_nan=True)
        paddle.enable_static()


class TestVarAPI_ZeroSize1(unittest.TestCase):
    def init_data(self):
        self.x_shape = [0]
        self.dtype = 'float64'
        self.expact_out = np.nan
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(self.dtype)

    def test_zerosize(self):
        self.init_data()
        paddle.disable_static()
        x = paddle.to_tensor(np.random.random(self.x_shape))
        out1 = paddle.var(x).numpy()
        np.testing.assert_allclose(out1, self.expact_out, equal_nan=True)
        paddle.enable_static()

    def test_static_zero(self):
        paddle.enable_static()
        self.init_data()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.x_shape, self.dtype)
            out = paddle.var(x)
            exe = paddle.static.Executor(paddle.CPUPlace())
            res = exe.run(feed={'X': self.x}, fetch_list=[out])
            np.testing.assert_allclose(self.expact_out, res[0], rtol=1e-05)
        paddle.disable_static()


class TestVarAPI_UnBiased1(unittest.TestCase):
    def init_data(self):
        self.x_shape = [1]
        # x = torch.randn([1])
        # res= torch.var(x,correction=0)     Here, res is 0.
        self.expact_out = 0.0

    def test_api(self):
        self.init_data()
        paddle.disable_static()
        x = paddle.to_tensor(np.random.random(self.x_shape))
        out1 = paddle.var(x, unbiased=False).numpy()
        np.testing.assert_allclose(out1, self.expact_out, equal_nan=True)
        paddle.enable_static()


class TestVarAPI_UnBiased2(unittest.TestCase):
    def init_data(self):
        self.x_shape = [1]
        # x = torch.randn([1])
        # res= torch.var(x,correction=1)     Here, res is 0.
        self.expact_out = np.nan

    def test_api(self):
        self.init_data()
        paddle.disable_static()
        x = paddle.to_tensor(np.random.random(self.x_shape))
        out1 = paddle.var(x, unbiased=True).numpy()
        np.testing.assert_allclose(out1, self.expact_out, equal_nan=True)
        paddle.enable_static()


def ref_var_with_correction(x, axis=None, correction=1, keepdim=False):
    if isinstance(axis, int):
        axis = (axis,)
    if axis is not None:
        axis = tuple(axis)
    return np.var(x, axis=axis, ddof=correction, keepdims=keepdim)


class TestVarAPI_Correction(TestVarAPI):
    def set_attrs(self):
        self.correction = 0
        self.use_correction = True

    def static(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape, self.dtype)
            if self.use_correction:
                out = paddle.var(
                    x,
                    self.axis,
                    keepdim=self.keepdim,
                    correction=self.correction,
                )
            else:
                out = paddle.var(x, self.axis, self.unbiased, self.keepdim)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x}, fetch_list=[out])
        return res[0]

    def dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        if self.use_correction:
            out = paddle.var(
                x, self.axis, keepdim=self.keepdim, correction=self.correction
            )
        else:
            out = paddle.var(x, self.axis, self.unbiased, self.keepdim)
        paddle.enable_static()
        return out.numpy()

    def test_api(self):
        if self.use_correction:
            out_ref = ref_var_with_correction(
                self.x, self.axis, self.correction, self.keepdim
            )
        else:
            out_ref = ref_var(self.x, self.axis, self.unbiased, self.keepdim)
        out_dygraph = self.dygraph()

        np.testing.assert_allclose(out_ref, out_dygraph, rtol=1e-05)
        self.assertTrue(np.equal(out_ref.shape, out_dygraph.shape).all())

        def test_static_or_pir_mode():
            out_static = self.static()
            np.testing.assert_allclose(out_ref, out_static, rtol=1e-05)
            self.assertTrue(np.equal(out_ref.shape, out_static.shape).all())

        test_static_or_pir_mode()


class TestVarAPI_Correction2(TestVarAPI_Correction):
    def set_attrs(self):
        self.correction = 2
        self.use_correction = True


class TestVarAPI_CorrectionFloat(TestVarAPI_Correction):
    def set_attrs(self):
        self.correction = 1.5
        self.use_correction = True


class TestVarAPI_CorrectionWithAxis(TestVarAPI_Correction):
    def set_attrs(self):
        self.correction = 0
        self.axis = [1, 2]
        self.use_correction = True


class TestVarAPI_OutParameter(unittest.TestCase):
    def setUp(self):
        self.dtype = 'float64'
        self.shape = [2, 3, 4]
        self.x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.place = get_device_place()

    def test_out_parameter_dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)

        out = paddle.empty(self.shape, dtype=self.dtype)
        result = paddle.var(x, out=out)

        self.assertTrue(paddle.equal_all(result, out))

        expected = paddle.var(x)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-05)

        paddle.enable_static()

    def test_out_parameter_with_axis(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        axis = 1

        expected_shape = list(self.shape)
        expected_shape.pop(axis)

        out = paddle.empty(expected_shape, dtype=self.dtype)
        result = paddle.var(x, axis=axis, out=out)

        self.assertTrue(paddle.equal_all(result, out))

        expected = paddle.var(x, axis=axis)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-05)

        paddle.enable_static()

    def test_out_parameter_with_keepdim(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        axis = 1

        expected_shape = list(self.shape)
        expected_shape[axis] = 1

        out = paddle.empty(expected_shape, dtype=self.dtype)
        result = paddle.var(x, axis=axis, keepdim=True, out=out)

        self.assertTrue(paddle.equal_all(result, out))

        expected = paddle.var(x, axis=axis, keepdim=True)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-05)

        paddle.enable_static()

    def test_out_parameter_none(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)

        result1 = paddle.var(x, out=None)
        result2 = paddle.var(x)

        np.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-05)

        paddle.enable_static()


class TestVarAPI_CorrectionAndOut(unittest.TestCase):
    def setUp(self):
        self.dtype = 'float64'
        self.shape = [2, 3, 4]
        self.x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)

    def test_correction_and_out_combination(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        correction = 0

        out = paddle.empty([], dtype=self.dtype)
        result = paddle.var(x, correction=correction, out=out)

        self.assertTrue(paddle.equal_all(result, out))

        expected = paddle.var(x, correction=correction)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-05)

        expected_np = np.var(self.x, ddof=correction)
        np.testing.assert_allclose(result.numpy(), expected_np, rtol=1e-05)

        paddle.enable_static()

    def test_correction_and_out_with_axis(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        correction = 2
        axis = 1

        expected_shape = list(self.shape)
        expected_shape.pop(axis)

        out = paddle.empty(expected_shape, dtype=self.dtype)
        result = paddle.var(x, axis=axis, correction=correction, out=out)

        self.assertTrue(paddle.equal_all(result, out))

        expected = paddle.var(x, axis=axis, correction=correction)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-05)

        expected_np = np.var(self.x, axis=axis, ddof=correction)
        np.testing.assert_allclose(result.numpy(), expected_np, rtol=1e-05)

        paddle.enable_static()


class TestVarAPI_ParamAlias(unittest.TestCase):
    def setUp(self):
        self.dtype = 'float64'
        self.shape = [2, 3, 4]
        self.x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)

    def test_input_alias(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)

        result1 = paddle.var(x=x)
        result2 = paddle.var(input=x)

        np.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-05)

        paddle.enable_static()

    def test_dim_alias(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        axis_val = 1

        result1 = paddle.var(x, axis=axis_val)
        result2 = paddle.var(x, dim=axis_val)

        np.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-05)

        paddle.enable_static()

    def test_all_aliases_combination(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        axis_val = [1, 2]

        result1 = paddle.var(x=x, axis=axis_val, unbiased=False, keepdim=True)
        result2 = paddle.var(
            input=x, dim=axis_val, unbiased=False, keepdim=True
        )

        np.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-05)

        paddle.enable_static()

    def test_alias_with_new_params(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        correction = 0

        expected_shape = []
        out = paddle.empty(expected_shape, dtype=self.dtype)

        result = paddle.var(input=x, correction=correction, out=out)

        expected = paddle.var(x, correction=correction)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-05)

        paddle.enable_static()

    def test_static_mode_aliases(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape, self.dtype)

            out = paddle.var(input=x, dim=1)

            exe = paddle.static.Executor(get_device_place())
            res = exe.run(feed={'X': self.x}, fetch_list=[out])

            expected = np.var(self.x, axis=1, ddof=1)
            np.testing.assert_allclose(res[0], expected, rtol=1e-05)


class TestVarAPI_CorrectionEdgeCases(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def tearDown(self):
        paddle.enable_static()

    def test_correction_larger_than_sample_size(self):
        x = paddle.to_tensor([1.0, 2.0, 3.0])

        result = paddle.var(x, correction=3)
        self.assertTrue(paddle.isinf(result) or paddle.isnan(result))

        result = paddle.var(x, correction=4)
        self.assertTrue(paddle.isinf(result) or paddle.isnan(result))

    def test_correction_negative(self):
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])

        result = paddle.var(x, correction=-1)
        expected_np = np.var(x.numpy(), ddof=-1)
        np.testing.assert_allclose(result.numpy(), expected_np, rtol=1e-05)

    def test_correction_zero(self):
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])

        result1 = paddle.var(x, correction=0)
        result2 = paddle.var(x, unbiased=False)

        np.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-05)


class TestVarAPI_NewParamsAlias(TestVarAPI_alias):
    def test_alias_with_new_parameters(self):
        paddle.disable_static()
        x = paddle.to_tensor(np.array([1, 2, 3, 4], 'float32'))

        out1 = paddle.var(x, correction=0).numpy()
        out2 = paddle.tensor.var(x, correction=0).numpy()
        out3 = paddle.tensor.stat.var(x, correction=0).numpy()
        np.testing.assert_allclose(out1, out2, rtol=1e-05)
        np.testing.assert_allclose(out1, out3, rtol=1e-05)

        out_tensor = paddle.empty([], dtype='float32')
        paddle.var(x, out=out_tensor)
        result1 = out_tensor.numpy()

        out_tensor2 = paddle.empty([], dtype='float32')
        paddle.tensor.var(x, out=out_tensor2)
        result2 = out_tensor2.numpy()

        np.testing.assert_allclose(result1, result2, rtol=1e-05)

        paddle.enable_static()


class TestVarAPI_Backward1(unittest.TestCase):
    def test_api(self):
        paddle.disable_static()
        self.shape = []
        self.axis = []
        self.x = np.random.uniform(-1, 1, self.shape).astype('float64')
        paddle.set_device(paddle.CPUPlace())

        out_ref = ref_var(self.x, self.axis, True, False)
        x = paddle.to_tensor(self.x)
        x.stop_gradient = False
        out = paddle.var(x, self.axis, True, False)

        out.sum().backward()
        paddle.enable_static()


class TestVarAPI_Backward2(unittest.TestCase):
    def test_api(self):
        paddle.disable_static()
        self.shape = [2]
        self.axis = []
        self.x = np.random.uniform(-1, 1, self.shape).astype('float64')
        paddle.set_device(paddle.CPUPlace())

        out_ref = ref_var(self.x, self.axis, True, False)
        x = paddle.to_tensor(self.x)
        x.stop_gradient = False
        out = paddle.var(x, self.axis, True, False)

        out.sum().backward()
        paddle.enable_static()


class TestVarAPI_Backward_ZeroSize1(unittest.TestCase):
    def test_api(self):
        paddle.disable_static()
        self.shape = [1, 3, 0, 10]
        self.axis = [1, 3]
        self.x = np.random.uniform(-1, 1, self.shape).astype('float64')
        paddle.set_device(paddle.CPUPlace())

        out_ref = ref_var(self.x, self.axis, True, False)
        x = paddle.to_tensor(self.x)
        x.stop_gradient = False
        out = paddle.var(x, self.axis, True, False)

        out.sum().backward()
        paddle.enable_static()


class TestVarEmptyAxisBackward(unittest.TestCase):
    """axis=[] and axis=None must not share a backward divisor.

    torch.var divides the gradient by _safe_size(sizes, dim), which is numel
    for dim=None and 1 for dim=[], while the forward reduces everything either
    way. The var_grad kernel and both composite var_grad rules have to agree
    on this.
    """

    def setUp(self):
        paddle.disable_static()
        paddle.set_device('cpu')
        self.x = np.array([1.0, 1.7], dtype='float64')

    def tearDown(self):
        paddle.enable_static()

    def _grad(self, axis):
        t = paddle.to_tensor(self.x)
        t.stop_gradient = False
        out = paddle.var(t, axis, True, False)
        out.sum().backward()
        return np.asarray(out.numpy()), t.grad.numpy()

    def test_axis_none_uses_numel(self):
        out, grad = self._grad(None)
        np.testing.assert_allclose(out, ref_var(self.x))
        # 2 * (x - mean) / (numel - correction) = 2 * (+-0.35) / 1
        np.testing.assert_allclose(grad, [-0.7, 0.7])

    def test_empty_axis_uses_one(self):
        out, grad = self._grad([])
        # The forward still reduces everything.
        np.testing.assert_allclose(out, ref_var(self.x))
        # The backward divides by 1 - 1 = 0, so it saturates to +-inf.
        self.assertTrue(np.all(np.isinf(grad)), f'grad = {grad}')

    def test_composite_rule_matches_kernel(self):
        eager_grad = self._grad([])[1]
        paddle.enable_static()
        try:
            from paddle.framework import core

            core._set_prim_backward_enabled(True)
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                x = paddle.static.data('x', self.x.shape, self.x.dtype)
                x.stop_gradient = False
                out = paddle.var(x, [], True, False)
                grads = paddle.static.gradients(out, x)
            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(startup)
            (prim_grad,) = exe.run(main, feed={'x': self.x}, fetch_list=grads)
        finally:
            core._set_prim_backward_enabled(False)
            paddle.disable_static()
        np.testing.assert_array_equal(prim_grad, eager_grad)


class TestVarStdNativeRegression(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        paddle.set_device('cpu')

    def tearDown(self):
        paddle.enable_static()

    def test_nonfinite_values_preserve_nan_semantics(self):
        for dtype in ('float32', 'float64'):
            for values in (
                [np.nan],
                [np.inf],
                [-np.inf],
                [1.0, np.inf],
                [1.0, np.nan],
            ):
                x = paddle.to_tensor(np.asarray(values, dtype=dtype))
                for op_name in ('var', 'std'):
                    result = getattr(paddle, op_name)(x, correction=1).numpy()
                    self.assertTrue(
                        np.isnan(result),
                        msg=f'{op_name}({dtype}, {values}) returned {result}',
                    )

    def test_correction_beyond_sample_count_matches_torch(self):
        for values, expected_is_inf in (
            ([1.0, 2.0, 4.0], True),
            ([2.0, 2.0, 2.0], False),
        ):
            x = paddle.to_tensor(np.asarray(values, dtype='float32'))
            for correction in (3.0, 4.0):
                for op_name in ('var', 'std'):
                    result = getattr(paddle, op_name)(
                        x, correction=correction
                    ).numpy()
                    if expected_is_inf:
                        self.assertTrue(
                            np.isposinf(result),
                            msg=f'{op_name} correction={correction}: {result}',
                        )
                    else:
                        self.assertTrue(
                            np.isnan(result),
                            msg=f'{op_name} correction={correction}: {result}',
                        )

    def test_correction_beyond_sample_count_backward_matches_torch(self):
        axis = None
        for values, is_constant in (
            ([1.0, 2.0], False),
            ([2.0, 2.0], True),
        ):
            for correction in (2.0, 3.0):
                for op_name in ('var', 'std'):
                    x = paddle.to_tensor(
                        np.asarray(values, dtype='float32'),
                        stop_gradient=False,
                    )
                    out = getattr(paddle, op_name)(
                        x, axis=axis, correction=correction
                    )
                    out.backward()
                    grad = x.grad.numpy()
                    if op_name == 'var' and not is_constant:
                        self.assertTrue(
                            np.all(np.isposinf(grad)),
                            msg=(
                                f'{op_name} axis={axis} '
                                f'correction={correction}: {grad}'
                            ),
                        )
                    else:
                        self.assertTrue(
                            np.all(np.isnan(grad)),
                            msg=(
                                f'{op_name} axis={axis} '
                                f'correction={correction}: {grad}'
                            ),
                        )

    def test_singular_backward_nonfinite_values_match_torch(self):
        cases = (
            ([np.inf], 1.0, True),
            ([-np.inf], 1.0, True),
            ([np.inf, np.inf], 2.0, True),
            ([-np.inf, -np.inf], 2.0, True),
            ([np.inf, -np.inf], 2.0, False),
            ([np.nan], 1.0, False),
        )
        for values, correction, is_nan in cases:
            for op_name in ('var', 'std'):
                x = paddle.to_tensor(
                    np.asarray(values, dtype='float32'), stop_gradient=False
                )
                out = getattr(paddle, op_name)(x, correction=correction)
                out.backward()
                grad = x.grad.numpy()
                if is_nan or op_name == 'std':
                    self.assertTrue(
                        np.all(np.isnan(grad)), (op_name, values, grad)
                    )
                else:
                    self.assertTrue(
                        np.all(np.isposinf(grad)), (op_name, values, grad)
                    )

    def test_cpu_low_precision_is_rejected(self):
        values = np.asarray([[1.0, 2.0, 4.0], [3.0, 5.0, 8.0]], 'float32')
        for dtype in ('float16', 'bfloat16'):
            x = paddle.to_tensor(values).astype(dtype)
            for op_name in ('var', 'std'):
                with self.assertRaisesRegex(
                    ValueError, 'on CPU does not support float16 or bfloat16'
                ):
                    getattr(paddle, op_name)(x, axis=1, correction=0)

    def test_static_cpu_low_precision_is_rejected(self):
        paddle.enable_static()
        try:
            paddle.set_device('cpu')
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                for dtype in ('float16', 'bfloat16'):
                    x = paddle.static.data(f'x_{dtype}', [2, 3], dtype=dtype)
                    for op_name in ('var', 'std'):
                        with self.assertRaisesRegex(
                            ValueError,
                            'on CPU does not support float16 or bfloat16',
                        ):
                            getattr(paddle, op_name)(x, axis=1, correction=0)
        finally:
            paddle.disable_static()

    def test_legacy_static_gpu_bfloat16_is_supported(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest('Test requires CUDA support.')
        if paddle.device.cuda.device_count() < 1:
            self.skipTest('Test requires an available CUDA device.')
        paddle.enable_static()
        try:
            if paddle.framework.in_pir_mode():
                self.skipTest(
                    'Legacy static graph requires PIR to be disabled.'
                )
            paddle.set_device('gpu')
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                x = paddle.static.data('x', [2, 3], dtype='bfloat16')
                var_out = paddle.var(x, axis=1, correction=0)
                std_out = paddle.std(x, axis=1, correction=0)
            exe = paddle.static.Executor(paddle.CUDAPlace(0))
            exe.run(startup)
            values = np.asarray(
                [[1.0, 2.0, 4.0], [3.0, 5.0, 8.0]], dtype='float32'
            )
            from paddle.base.data_feeder import convert_float_to_uint16

            result_var, result_std = exe.run(
                main,
                feed={'x': convert_float_to_uint16(values)},
                fetch_list=[var_out, std_out],
            )
            from paddle.base.data_feeder import convert_uint16_to_float

            expected_var = np.var(values, axis=1).astype('float32')
            expected_std = np.sqrt(expected_var)
            np.testing.assert_allclose(
                convert_uint16_to_float(result_var),
                expected_var,
                rtol=1e-2,
                atol=1e-2,
            )
            np.testing.assert_allclose(
                convert_uint16_to_float(result_std),
                expected_std,
                rtol=1e-2,
                atol=1e-2,
            )
        finally:
            paddle.disable_static()
            paddle.set_device('cpu')

    def test_dynamic_path_does_not_use_python_composition(self):
        x = paddle.to_tensor(np.asarray([1.0, 2.0, 4.0], 'float32'))
        original_mean = paddle.mean
        try:
            paddle.mean = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError('var must dispatch to the native op')
            )
            result = paddle.var(x, correction=0).numpy()
        finally:
            paddle.mean = original_mean
        np.testing.assert_allclose(
            result, np.var([1.0, 2.0, 4.0]), rtol=1e-6, atol=1e-6
        )


class TestVarStdLegacyStaticGraph(unittest.TestCase):
    """var/std must work under the legacy (non-PIR) static graph.

    stat.py builds the var/std ops through _append_stat_op only on that
    path; PIR-mode programs dispatch through _C_ops just like eager mode.
    """

    def setUp(self):
        self.x = np.random.uniform(-1, 1, [2, 3, 4]).astype('float64')

    def _run_static(self, build):
        with OldIrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                x = paddle.static.data('x', self.x.shape, 'float64')
                out = build(x)
                exe = paddle.static.Executor(paddle.CPUPlace())
                exe.run(startup)
                (res,) = exe.run(main, feed={'x': self.x}, fetch_list=[out])
        return res

    def test_var_correction(self):
        for axis, correction in ((1, 0), (None, 2)):
            res = self._run_static(
                lambda x: paddle.var(x, axis=axis, correction=correction)
            )
            np.testing.assert_allclose(
                res, np.var(self.x, axis=axis, ddof=correction), rtol=1e-12
            )

    def test_var_unbiased(self):
        res = self._run_static(lambda x: paddle.var(x, axis=1, unbiased=False))
        np.testing.assert_allclose(
            res, np.var(self.x, axis=1, ddof=0), rtol=1e-12
        )

    def test_std_correction(self):
        res = self._run_static(lambda x: paddle.std(x, axis=1, correction=0))
        np.testing.assert_allclose(
            res, np.std(self.x, axis=1, ddof=0), rtol=1e-12
        )

    def test_std_unbiased(self):
        res = self._run_static(lambda x: paddle.std(x, axis=1, unbiased=False))
        np.testing.assert_allclose(
            res, np.std(self.x, axis=1, ddof=0), rtol=1e-12
        )


class TestVarStdUnbiasedCorrectionConflict(unittest.TestCase):
    """Only one of unbiased and correction may be given, for var and std."""

    def setUp(self):
        paddle.disable_static()

    def tearDown(self):
        paddle.enable_static()

    def test_conflict_raises(self):
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        for op_name in ('var', 'std'):
            with self.assertRaisesRegex(
                ValueError, 'Only one of unbiased and correction'
            ):
                getattr(paddle, op_name)(x, unbiased=True, correction=0)


if __name__ == '__main__':
    unittest.main()
