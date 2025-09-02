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
from op_test import get_device_place

import paddle


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
        self.x_shape = []
        # x = torch.tensor([])
        # res= torch.var(x)     Here, res is nan
        self.expact_out = np.nan

    def test_zerosize(self):
        self.init_data()
        paddle.disable_static()
        x = paddle.to_tensor(np.random.random(self.x_shape))
        out1 = paddle.var(x).numpy()
        np.testing.assert_allclose(out1, self.expact_out, equal_nan=True)
        paddle.enable_static()


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


if __name__ == '__main__':
    unittest.main()
