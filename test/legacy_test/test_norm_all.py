# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
    convert_float_to_uint16,
    get_device_place,
    is_custom_device,
)
from utils import static_guard

import paddle
from paddle import _C_ops, base
from paddle.base import core
from paddle.base.framework import in_dygraph_mode


# hack method for test p_norm final state
def p_norm_python_api(
    x, p=2.0, axis=-1, epsilon=1e-12, keepdim=False, as_vector=False
):
    if in_dygraph_mode():
        return _C_ops.p_norm(x, p, axis, epsilon, keepdim, as_vector)


def norm_public_python_api(
    x, p=2.0, axis=-1, epsilon=1e-12, keepdim=False, as_vector=False
):
    return paddle.linalg.norm(
        x,
        p,
        axis,
        keepdim,
    )


def np_linalg_vector_norm(x, axis, porder, keepdims=False):
    x_shape = list(x.shape)

    origin_axis = axis
    if origin_axis is None:
        pass
    elif isinstance(origin_axis, int):
        origin_axis = [origin_axis]
    else:
        origin_axis = list(origin_axis)

    if axis is None:
        x = x.ravel()
        axis = -1

    if not isinstance(axis, int) and len(axis) > 1:
        for i in range(len(axis)):
            if axis[i] < 0:
                axis[i] += len(x.shape)
        tmp_axis = []
        for i in range(len(axis)):
            tmp_axis.append(-1 - i)
        x = np.moveaxis(x, axis, tmp_axis)

        front_dim = x.shape[0 : len(x.shape) - len(axis)]
        back_dim = 1
        for i in range(len(x.shape) - len(axis), len(x.shape)):
            back_dim = back_dim * x.shape[i]
        front_dim = list(front_dim)
        front_dim.append(back_dim)
        x = x.reshape(front_dim)
        axis = -1
    if isinstance(axis, list):
        axis = tuple(axis)

    r = np.linalg.norm(x, ord=porder, axis=axis, keepdims=keepdims)

    r_shape = r.shape

    if keepdims:
        if origin_axis is None:
            r_shape = np.ones_like(x_shape)
        elif len(origin_axis) > 1:
            r_shape = x_shape
            for i in origin_axis:
                r_shape[i] = 1
    r = r.reshape(r_shape)
    return r


def np_linalg_matrix_norm(x, axis, porder, keepdims=False):
    axis = tuple(axis)
    r = np.linalg.norm(x, ord=porder, axis=axis, keepdims=keepdims)
    return r


def np_linalg_norm(x, axis, porder, keepdims=False):
    r = []
    if axis is None or isinstance(axis, (int, float)):
        r = np_linalg_vector_norm(x, axis, porder, keepdims)
    elif isinstance(axis, list) and len(axis) == 2:
        r = np_linalg_matrix_norm(x, axis, porder, keepdims)
    r = r.astype(x.dtype)

    return r


def numpy_frobenius_norm(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    if axis is None:
        axis = (-2, -1)
    r = np.linalg.norm(x, ord='fro', axis=axis, keepdims=keepdims).astype(
        x.dtype
    )
    return r


def numpy_nuclear_norm(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    r = np.linalg.norm(x, ord='nuc', axis=axis, keepdims=keepdims).astype(
        x.dtype
    )
    return r


def frobenius_norm(x, dim, keep_dim):
    return paddle.linalg.norm(x, p='fro', axis=dim, keepdim=keep_dim)


def nuclear_norm(x, dim, keep_dim):
    return paddle.linalg.norm(x, p='nuc', axis=dim, keepdim=keep_dim)


class TestFrobeniusNormOp(OpTest):
    def setUp(self):
        self.python_api = frobenius_norm
        self.op_type = "frobenius_norm"
        self.init_test_case()
        self.init_dtype()
        x = (np.random.random(self.shape) + 1.0).astype(self.dtype)
        norm = numpy_frobenius_norm(x, self.axis, self.keepdim)
        self.reduce_all = False
        self.inputs = {'X': x}
        self.attrs = {
            'dim': list(self.axis),
            'keep_dim': self.keepdim,
            'reduce_all': self.reduce_all,
        }
        self.outputs = {'Out': norm}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True)

    def init_test_case(self):
        self.shape = [2, 3, 4, 5]
        self.axis = (1, 2)
        self.keepdim = False

    def init_dtype(self):
        self.dtype = "float64"


class TestFrobeniusNormOp2(TestFrobeniusNormOp):
    def init_test_case(self):
        self.shape = [5, 5, 5]
        self.axis = (0, 1)
        self.keepdim = True

    def init_dtype(self):
        self.dtype = "float32"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True)


class TestFrobeniusNormOp3(TestFrobeniusNormOp):
    def init_test_case(self):
        self.shape = [5, 5, 5]
        self.axis = (0, 1)
        self.keepdim = True

    def init_dtype(self):
        self.dtype = "complex64"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True)


class TestFrobeniusNormOp4(TestFrobeniusNormOp):
    def init_test_case(self):
        self.shape = [5, 5, 5, 2]
        self.axis = (0, 1)
        self.keepdim = True

    def init_dtype(self):
        self.dtype = "complex128"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True)


class TestFrobeniusNormOpZeroSize(TestFrobeniusNormOp):
    def init_test_case(self):
        self.shape = [0, 20, 3]
        self.axis = (1, 2)
        self.keepdim = False

    def init_dtype(self):
        self.dtype = "float32"

    def test_check_output(self):
        places = (
            [paddle.CPUPlace(), get_device_place()]
            if (core.is_compiled_with_cuda() or is_custom_device())
            else [paddle.CPUPlace()]
        )
        for place in places:
            self.check_output_with_place(place)

    def test_check_grad(self):
        pass


class TestFrobeniusNormOpZeroSize2(TestFrobeniusNormOpZeroSize):
    def init_test_case(self):
        self.shape = [3, 0, 3]
        self.axis = (1, 2)
        self.keepdim = False


class TestFrobeniusNormOpZeroSize3(TestFrobeniusNormOpZeroSize):
    def init_test_case(self):
        self.shape = [0, 20, 3]
        self.axis = (0, 2)
        self.keepdim = False


class TestFrobeniusNormOpZeroSize4(TestFrobeniusNormOpZeroSize):
    def init_test_case(self):
        self.shape = [0, 20, 3]
        self.axis = (0, -1)
        self.keepdim = False


class TestPnormOp(OpTest):
    def setUp(self):
        self.op_type = "p_norm"
        self.python_api = p_norm_python_api
        self.public_python_api = norm_public_python_api
        self.prim_op_type = "comp"
        self.init_test_case()
        self.init_dtype()
        self.fw_comp_atol = 1e-6
        self.fw_comp_rtol = 1e-6
        self.rev_comp_atol = 1e-6
        self.rev_comp_rtol = 1e-6
        x = (np.random.random(self.shape) + 0.5).astype(self.dtype)
        norm = np_linalg_norm(x, self.axis, self.porder, self.keepdim)
        self.inputs = {'X': x}
        self.attrs = {
            'epsilon': self.epsilon,
            'axis': self.axis,
            'keepdim': self.keepdim,
            'porder': float(self.porder),
            'asvector': self.asvector,
        }
        self.outputs = {'Out': norm}
        self.gradient = self.calc_gradient()

    def test_check_output(self):
        self.check_output(check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_prim_pir=True)

    def init_test_case(self):
        self.shape = [2, 3, 4, 5]
        self.axis = 1
        self.epsilon = 1e-12
        self.porder = 2.0
        self.keepdim = False
        self.asvector = False

    def init_dtype(self):
        self.dtype = "float64"

    def calc_gradient(self):
        self.attrs = {
            'epsilon': self.epsilon,
            'axis': self.axis,
            'keepdim': self.keepdim,
            'porder': float(self.porder),
            'asvector': self.asvector,
        }
        x = self.inputs["X"]
        porder = self.attrs["porder"]
        axis = self.attrs["axis"]
        asvector = self.attrs["asvector"]
        x_dtype = x.dtype
        x = x.astype(np.float32) if x.dtype == np.float16 else x
        if porder == 0:
            grad = np.zeros(x.shape).astype(x.dtype)
        elif porder in [float("inf"), float("-inf")]:
            norm = np_linalg_norm(x, axis=axis, porder=porder, keepdims=True)
            x_abs = np.abs(x)
            grad = np.sign(x)
            grad[x_abs != norm] = 0.0
        else:
            norm = np_linalg_norm(x, axis=axis, porder=porder, keepdims=True)
            grad = (
                np.power(norm, 1 - porder)
                * np.power(np.abs(x), porder - 1)
                * np.sign(x)
            )

        numel = 1
        for s in x.shape:
            numel *= s
        divisor = numel if asvector else x.shape[axis]
        numel /= divisor
        return [grad.astype(x_dtype) * 1 / numel]


class TestPnormOp2(TestPnormOp):
    def init_test_case(self):
        self.shape = [3, 20, 3]
        self.axis = 2
        self.epsilon = 1e-12
        self.porder = 2.0
        self.keepdim = True
        self.asvector = False

    def init_dtype(self):
        self.dtype = "float32"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_prim_pir=True)


class TestPnormOp3(TestPnormOp):
    def init_test_case(self):
        self.shape = [3, 20, 3]
        self.axis = 2
        self.epsilon = 1e-12
        self.porder = np.inf
        self.keepdim = True
        self.asvector = False

    def init_dtype(self):
        self.dtype = "float32"

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', user_defined_grads=self.gradient, check_prim_pir=True
        )


class TestPnormOp4(TestPnormOp):
    def init_test_case(self):
        self.shape = [3, 20, 3]
        self.axis = 2
        self.epsilon = 1e-12
        self.porder = -np.inf
        self.keepdim = True
        self.asvector = False

    def init_dtype(self):
        self.dtype = "float32"

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', user_defined_grads=self.gradient, check_prim_pir=True
        )


class TestPnormOp5(TestPnormOp):
    def init_test_case(self):
        self.shape = [3, 20, 3]
        self.axis = 2
        self.epsilon = 1e-12
        self.porder = 0
        self.keepdim = True
        self.asvector = False

    def init_dtype(self):
        self.dtype = "float32"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', user_defined_grads=self.gradient)


class TestPnormOp6(TestPnormOp):
    def init_test_case(self):
        self.shape = [3, 20, 3]
        self.axis = -1
        self.epsilon = 1e-12
        self.porder = 2
        self.keepdim = False
        self.asvector = False

    def init_dtype(self):
        self.dtype = "float32"

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', user_defined_grads=self.gradient, check_prim_pir=True
        )


# Regression test: porder=2.3 (not exactly representable in float32) with float64
# input to verify that the GPU kernel uses double precision for porder without
# truncating it to float, which would cause forward output mismatch.
class TestPnormOp7(TestPnormOp):
    def init_test_case(self):
        self.shape = [3, 20, 3]
        self.axis = 2
        self.epsilon = 1e-12
        self.porder = 2.3
        self.keepdim = True
        self.asvector = False

    def init_dtype(self):
        self.dtype = "float64"

    def test_check_output(self):
        self.check_output(atol=1e-12, rtol=1e-12, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', user_defined_grads=self.gradient, check_prim_pir=True
        )


class TestPnormOpZeroSize(TestPnormOp):
    def init_test_case(self):
        self.shape = [0, 20, 3]
        self.axis = 1
        self.epsilon = 1e-12
        self.porder = 2
        self.keepdim = False
        self.asvector = False

    def init_dtype(self):
        self.dtype = "float32"

    def test_check_output(self):
        places = (
            [paddle.CPUPlace(), get_device_place()]
            if (core.is_compiled_with_cuda() or is_custom_device())
            else [paddle.CPUPlace()]
        )
        for place in places:
            self.check_output_with_place(place)

    def test_check_grad(self):
        pass

    def calc_gradient(self):
        pass


class TestPnormOpZeroSize2(TestPnormOpZeroSize):
    def init_test_case(self):
        self.shape = [3, 0, 3]
        self.axis = 1
        self.epsilon = 1e-12
        self.porder = 2
        self.keepdim = False
        self.asvector = False


class TestPnormOpZeroSize3(TestPnormOpZeroSize):
    def init_test_case(self):
        self.shape = [0, 20, 3]
        self.axis = 2
        self.epsilon = 1e-12
        self.porder = 2
        self.keepdim = False
        self.asvector = False


class TestPnormOpZeroSize4(TestPnormOpZeroSize):
    def init_test_case(self):
        self.shape = [0, 20, 3]
        self.axis = -1
        self.epsilon = 1e-12
        self.porder = 2
        self.keepdim = False
        self.asvector = False


def create_test_fp16_class(parent, max_relative_error=2e-3):
    @unittest.skipIf(
        not (core.is_compiled_with_cuda() or is_custom_device()),
        "core is not compiled with CUDA",
    )
    class TestPnormFP16Op(parent):
        def init_dtype(self):
            self.dtype = "float16"

        def test_check_output(self):
            place = get_device_place()
            if core.is_float16_supported(place):
                self.check_output_with_place(place)

        def test_check_grad(self):
            place = get_device_place()
            if core.is_float16_supported(place):
                self.check_grad_with_place(
                    place,
                    ['X'],
                    'Out',
                    user_defined_grads=self.gradient,
                    max_relative_error=max_relative_error,
                )

    cls_name = "{}_{}".format(parent.__name__, "Fp16")
    TestPnormFP16Op.__name__ = cls_name
    globals()[cls_name] = TestPnormFP16Op


create_test_fp16_class(TestPnormOp)
create_test_fp16_class(TestPnormOp2)
create_test_fp16_class(TestPnormOp3)
create_test_fp16_class(TestPnormOp4)
create_test_fp16_class(TestPnormOp5)
create_test_fp16_class(TestPnormOp6)


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA",
)
class TestPnormBF16Op(OpTest):
    def setUp(self):
        self.op_type = "p_norm"
        self.prim_op_type = "comp"
        self.python_api = p_norm_python_api
        self.public_python_api = norm_public_python_api
        self.init_test_case()
        self.x = (np.random.random(self.shape) + 0.5).astype(np.float32)
        self.norm = np_linalg_norm(self.x, self.axis, self.porder, self.keepdim)
        self.gradient = self.calc_gradient()
        self.inputs = {'X': convert_float_to_uint16(self.x)}
        self.attrs = {
            'epsilon': self.epsilon,
            'axis': self.axis,
            'keepdim': self.keepdim,
            'porder': float(self.porder),
            'asvector': self.asvector,
        }
        self.outputs = {'Out': convert_float_to_uint16(self.norm)}

    def test_check_output(self):
        place = get_device_place()
        self.check_output_with_place(place, atol=1e-3, check_prim_pir=True)

    def test_check_grad(self):
        place = get_device_place()
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            user_defined_grads=self.gradient,
            check_prim_pir=True,
        )

    def init_test_case(self):
        self.shape = [2, 3, 4, 5]
        self.axis = 1
        self.epsilon = 1e-12
        self.porder = 2.0
        self.keepdim = False
        self.asvector = False

    def init_dtype(self):
        self.dtype = np.uint16

    def calc_gradient(self):
        self.attrs = {
            'epsilon': self.epsilon,
            'axis': self.axis,
            'keepdim': self.keepdim,
            'porder': float(self.porder),
            'asvector': self.asvector,
        }
        x = self.x
        porder = self.attrs["porder"]
        axis = self.attrs["axis"]
        asvector = self.attrs["asvector"]
        x_dtype = x.dtype
        x = x.astype(np.float32) if x.dtype == np.float16 else x
        if porder == 0:
            grad = np.zeros(x.shape).astype(x.dtype)
        elif porder in [float("inf"), float("-inf")]:
            norm = np_linalg_norm(x, axis=axis, porder=porder, keepdims=True)
            x_abs = np.abs(x)
            grad = np.sign(x)
            grad[x_abs != norm] = 0.0
        else:
            norm = np_linalg_norm(x, axis=axis, porder=porder, keepdims=True)
            grad = (
                np.power(norm, 1 - porder)
                * np.power(np.abs(x), porder - 1)
                * np.sign(x)
            )

        numel = 1
        for s in x.shape:
            numel *= s
        divisor = numel if asvector else x.shape[axis]
        numel /= divisor
        return [grad.astype(x_dtype) * 1 / numel]


def check_fro_static(self, p, axis, shape_x, dtype, keep_dim, check_dim=False):
    with base.program_guard(base.Program()):
        data = paddle.static.data(name="X", shape=shape_x, dtype=dtype)
        out = paddle.norm(x=data, p=p, axis=axis, keepdim=keep_dim)
        place = base.CPUPlace()
        exe = base.Executor(place)
        np_input = (np.random.rand(*shape_x) + 1.0).astype(dtype)
        expected_result = numpy_frobenius_norm(
            np_input, axis=axis, keepdims=keep_dim
        )
        (result,) = exe.run(feed={"X": np_input}, fetch_list=[out])
    np.testing.assert_allclose(result, expected_result, rtol=1e-6, atol=1e-8)
    if keep_dim and check_dim:
        np.testing.assert_equal(result.shape, expected_result.shape)


def check_fro_dygraph(self, p, axis, shape_x, dtype, keep_dim, check_dim=False):
    x_numpy = (np.random.random(shape_x) + 1.0).astype(dtype)
    expected_result = numpy_frobenius_norm(x_numpy, axis, keep_dim)
    x_paddle = paddle.to_tensor(x_numpy)
    result = paddle.norm(x=x_paddle, p=p, axis=axis, keepdim=keep_dim)
    result = result.numpy()
    np.testing.assert_allclose(result, expected_result, rtol=1e-6, atol=1e-8)
    if keep_dim and check_dim:
        np.testing.assert_equal(result.shape, expected_result.shape)


def check_nuc_static(self, p, axis, shape_x, dtype, keep_dim, check_dim=False):
    with base.program_guard(base.Program()):
        data = paddle.static.data(name="X", shape=shape_x, dtype=dtype)
        out = paddle.norm(x=data, p=p, axis=axis, keepdim=keep_dim)
        place = base.CPUPlace()
        exe = base.Executor(place)
        np_input = (np.random.rand(*shape_x) + 1.0).astype(dtype)
        expected_result = numpy_nuclear_norm(
            np_input, axis=axis, keepdims=keep_dim
        )
        (result,) = exe.run(feed={"X": np_input}, fetch_list=[out])
    np.testing.assert_allclose(result, expected_result, rtol=1e-6, atol=1e-8)
    if keep_dim and check_dim:
        np.testing.assert_equal(result.shape, expected_result.shape)


def check_nuc_dygraph(self, p, axis, shape_x, dtype, keep_dim, check_dim=False):
    x_numpy = (np.random.random(shape_x) + 1.0).astype(dtype)
    expected_result = numpy_nuclear_norm(x_numpy, axis, keep_dim)
    x_paddle = paddle.to_tensor(x_numpy)
    result = paddle.norm(x=x_paddle, p=p, axis=axis, keepdim=keep_dim)
    result = result.numpy()
    np.testing.assert_allclose(result, expected_result, rtol=1e-6, atol=1e-8)
    if keep_dim and check_dim:
        np.testing.assert_equal(result.shape, expected_result.shape)


def check_linalg_norm_static(
    self, p, axis, shape_x, dtype, keep_dim, check_dim=False
):
    with base.program_guard(base.Program()):
        data = paddle.static.data(name="X", shape=shape_x, dtype=dtype)
        out = paddle.norm(x=data, p=p, axis=axis, keepdim=keep_dim)
        place = base.CPUPlace()
        exe = base.Executor(place)
        np_input = (np.random.rand(*shape_x) + 1.0).astype(dtype)
        expected_result = np_linalg_norm(
            np_input, porder=p, axis=axis, keepdims=keep_dim
        ).astype(dtype)
        (result,) = exe.run(feed={"X": np_input}, fetch_list=[out])
    np.testing.assert_allclose(result, expected_result, rtol=1e-6, atol=1e-8)
    if keep_dim and check_dim:
        np.testing.assert_equal(result.shape, expected_result.shape)


def check_linalg_norm_dygraph(
    self, p, axis, shape_x, dtype, keep_dim, check_dim=False
):
    x_numpy = (np.random.random(shape_x) + 1.0).astype(dtype)
    expected_result = np_linalg_norm(
        x_numpy, porder=p, axis=axis, keepdims=keep_dim
    )
    x_paddle = paddle.to_tensor(x_numpy)
    result = paddle.linalg.norm(x=x_paddle, p=p, axis=axis, keepdim=keep_dim)
    result = result.numpy()
    np.testing.assert_allclose(result, expected_result, rtol=1e-6, atol=1e-8)
    if keep_dim and check_dim:
        np.testing.assert_equal(result.shape, expected_result.shape)


def check_linalg_matrix_static(
    self, p, axis, shape_x, dtype, keep_dim, check_dim=False
):
    with base.program_guard(base.Program()):
        data = paddle.static.data(name="X", shape=shape_x, dtype=dtype)
        out = paddle.linalg.matrix_norm(
            x=data, p=p, axis=axis, keepdim=keep_dim
        )
        place = base.CPUPlace()
        exe = base.Executor(place)
        np_input = (np.random.rand(*shape_x) + 1.0).astype(dtype)
        expected_result = np_linalg_matrix_norm(
            np_input, porder=p, axis=axis, keepdims=keep_dim
        ).astype(dtype)
        (result,) = exe.run(feed={"X": np_input}, fetch_list=[out])
    np.testing.assert_allclose(result, expected_result, rtol=1e-6, atol=1e-8)
    if keep_dim and check_dim:
        np.testing.assert_equal(result.shape, expected_result.shape)


def check_linalg_matrix_dygraph(
    self, p, axis, shape_x, dtype, keep_dim, check_dim=False
):
    x_numpy = (np.random.random(shape_x) + 1.0).astype(dtype)
    expected_result = np_linalg_matrix_norm(
        x_numpy, porder=p, axis=axis, keepdims=keep_dim
    )
    x_paddle = paddle.to_tensor(x_numpy)
    result = paddle.linalg.matrix_norm(
        x=x_paddle, p=p, axis=axis, keepdim=keep_dim
    )
    result = result.numpy()
    np.testing.assert_allclose(result, expected_result, rtol=1e-6, atol=1e-8)
    if keep_dim and check_dim:
        np.testing.assert_equal(result.shape, expected_result.shape)


def check_linalg_vector_static(
    self, p, axis, shape_x, dtype, keep_dim, check_dim=False
):
    with base.program_guard(base.Program()):
        data = paddle.static.data(name="X", shape=shape_x, dtype=dtype)
        out = paddle.linalg.vector_norm(
            x=data, p=p, axis=axis, keepdim=keep_dim
        )
        place = base.CPUPlace()
        exe = base.Executor(place)
        np_input = np.array(np.random.rand(*shape_x) + 1.0).astype(dtype)
        expected_result = np_linalg_vector_norm(
            np_input, porder=p, axis=axis, keepdims=keep_dim
        ).astype(dtype)
        (result,) = exe.run(feed={"X": np_input}, fetch_list=[out])
    np.testing.assert_allclose(result, expected_result, rtol=1e-6, atol=1e-8)
    if keep_dim and check_dim:
        np.testing.assert_equal(result.shape, expected_result.shape)


def check_linalg_vector_dygraph(
    self, p, axis, shape_x, dtype, keep_dim, check_dim=False
):
    x_numpy = np.array(np.random.random(shape_x) + 1.0).astype(dtype)
    expected_result = np_linalg_vector_norm(
        x_numpy, porder=p, axis=axis, keepdims=keep_dim
    )
    x_paddle = paddle.to_tensor(x_numpy)
    result = paddle.linalg.vector_norm(
        x=x_paddle, p=p, axis=axis, keepdim=keep_dim
    )
    result = result.numpy()
    np.testing.assert_allclose(result, expected_result, rtol=1e-6, atol=1e-8)
    if keep_dim and check_dim:
        np.testing.assert_equal(result.shape, expected_result.shape)


class NormTestForNUCAndDtype(unittest.TestCase):
    def test_nuc_and_dtype(self):
        x = np.random.randn(10, 20).astype("float32")
        res_numpy = np.linalg.norm(x, ord='nuc')
        res_paddle = paddle.tensor(x).norm(p="nuc")
        np.testing.assert_allclose(
            res_numpy, res_paddle.numpy(), rtol=1e-6, atol=1e-6
        )
        res_numpy = np.linalg.norm(x.astype("float64"), ord="nuc")
        res_paddle = paddle.tensor(x).norm(p="nuc", dtype="float64")
        np.testing.assert_allclose(
            res_numpy, res_paddle.numpy(), rtol=1e-6, atol=1e-6
        )
        self.assertEqual(res_paddle.dtype, paddle.float64)

    def test_with_out(self):
        # matrix
        x = np.random.randn(10, 20).astype("float32")

        res_numpy = np.linalg.norm(x, ord='nuc')
        res_out = paddle.zeros(res_numpy.shape, dtype="float32")
        res_paddle = paddle.tensor(x).norm(p='nuc', out=res_out)
        np.testing.assert_allclose(
            res_numpy, res_out.numpy(), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            res_out.numpy(), res_paddle.numpy(), rtol=1e-6, atol=1e-6
        )

        res_numpy = np.linalg.norm(x, ord=2, axis=(0, 1))
        res_out = paddle.zeros(res_numpy.shape, dtype="float32")
        res_paddle = paddle.tensor(x).norm(p=2, axis=[0, 1], out=res_out)
        np.testing.assert_allclose(
            res_out.numpy(), res_paddle.numpy(), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            res_numpy, res_out.numpy(), rtol=1e-5, atol=1e-6
        )

        # vector
        x = np.random.randn(10).astype("float32")
        res_numpy = np.linalg.norm(x, ord=2, axis=0)
        res_out = paddle.zeros(res_numpy.shape, dtype="float32")
        res_paddle = paddle.tensor(x).norm(p='fro', axis=0, out=res_out)
        np.testing.assert_allclose(
            res_numpy, res_out.numpy(), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            res_out.numpy(), res_paddle.numpy(), rtol=1e-6, atol=1e-6
        )

        res_numpy = np.linalg.norm(x, ord=2, axis=0)
        res_out = paddle.zeros(res_numpy.shape, dtype="float32")
        res_paddle = paddle.tensor(x).norm(p=2, axis=0, out=res_out)
        np.testing.assert_allclose(
            res_numpy, res_out.numpy(), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            res_out.numpy(), res_paddle.numpy(), rtol=1e-6, atol=1e-6
        )


class TestVectorNormDtypeAndOut(unittest.TestCase):
    def test_alias_dtype_and_out(self):
        x = np.random.randn(10).astype("float16")
        dtype = "float32"
        except_numpy = np_linalg_vector_norm(x.astype(dtype), porder=2, axis=0)
        out_res = paddle.zeros(except_numpy.shape, dtype="float32")
        res = paddle.linalg.vector_norm(
            paddle.tensor(x), p=2, axis=0, dtype=dtype, out=out_res
        )
        res_alias = paddle.linalg.vector_norm(
            paddle.tensor(x), ord=2, dim=0, dtype=dtype, out=out_res
        )
        np.testing.assert_allclose(
            except_numpy, res.numpy(), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            except_numpy, out_res.numpy(), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            except_numpy, res_alias.numpy(), rtol=1e-6, atol=1e-6
        )
        self.assertEqual(res.dtype, res_alias.dtype)
        self.assertEqual(res.dtype, out_res.dtype)
        self.assertEqual(res.dtype, paddle.float32)


class API_NormTest(unittest.TestCase):
    def test_basic(self):
        with static_guard():
            keep_dims = {False, True}
            for keep in keep_dims:
                check_fro_static(
                    self,
                    p='fro',
                    axis=[-2, -1],
                    shape_x=[2, 3, 4],
                    dtype="float32",
                    keep_dim=keep,
                )
                check_fro_static(
                    self,
                    p='fro',
                    axis=[0, 1],
                    shape_x=[2, 3, 4],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_nuc_static(
                    self,
                    p='nuc',
                    axis=[0, 1],
                    shape_x=[2, 3, 4],
                    dtype='float64',
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_norm_static(
                    self,
                    p=2,
                    axis=None,
                    shape_x=[3, 4],
                    dtype="float32",
                    keep_dim=keep,
                )
                check_linalg_norm_static(
                    self,
                    p=2,
                    axis=1,
                    shape_x=[3, 4],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_norm_static(
                    self,
                    p=np.inf,
                    axis=0,
                    shape_x=[2, 3, 4],
                    dtype="float32",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_norm_static(
                    self,
                    p=np.inf,
                    axis=None,
                    shape_x=[2, 3, 4],
                    dtype="float32",
                    keep_dim=keep,
                )
                check_linalg_norm_static(
                    self,
                    p=-np.inf,
                    axis=0,
                    shape_x=[2, 3, 4],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_norm_static(
                    self,
                    p=-np.inf,
                    axis=None,
                    shape_x=[2, 3, 4],
                    dtype="float64",
                    keep_dim=keep,
                )
                check_linalg_norm_static(
                    self,
                    p=0,
                    axis=1,
                    shape_x=[3, 4],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )

                check_linalg_norm_static(
                    self,
                    p=1,
                    axis=1,
                    shape_x=[3, 4],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_norm_static(
                    self,
                    p=0,
                    axis=None,
                    shape_x=[3, 4],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_norm_static(
                    self,
                    p=2,
                    axis=[0, 1],
                    shape_x=[2, 3, 4],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_norm_static(
                    self,
                    p=2,
                    axis=-1,
                    shape_x=[2, 3, 4],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_norm_static(
                    self,
                    p=1,
                    axis=[0, 1],
                    shape_x=[2, 3, 4],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_norm_static(
                    self,
                    p=np.inf,
                    axis=[0, 1],
                    shape_x=[2, 3, 4],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_norm_static(
                    self,
                    p=-np.inf,
                    axis=[0, 1],
                    shape_x=[2, 3, 4],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )

                check_linalg_vector_static(
                    self,
                    p=2,
                    axis=None,
                    shape_x=[3, 4],
                    dtype="float32",
                    keep_dim=keep,
                )
                check_linalg_vector_static(
                    self,
                    p=4,
                    axis=1,
                    shape_x=[3, 4],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_vector_static(
                    self,
                    p=np.inf,
                    axis=0,
                    shape_x=[2, 3, 4],
                    dtype="float32",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_vector_static(
                    self,
                    p=np.inf,
                    axis=None,
                    shape_x=[2, 3, 4],
                    dtype="float32",
                    keep_dim=keep,
                )
                check_linalg_vector_static(
                    self,
                    p=-np.inf,
                    axis=0,
                    shape_x=[2, 3, 4],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_vector_static(
                    self,
                    p=-np.inf,
                    axis=None,
                    shape_x=[2, 3, 4],
                    dtype="float64",
                    keep_dim=keep,
                )
                check_linalg_vector_static(
                    self,
                    p=0,
                    axis=1,
                    shape_x=[3, 4],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_vector_static(
                    self,
                    p=1,
                    axis=1,
                    shape_x=[3, 4],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_vector_static(
                    self,
                    p=0,
                    axis=None,
                    shape_x=[3, 4],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_vector_static(
                    self,
                    p=2,
                    axis=[0, 1],
                    shape_x=[2, 3, 4],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_vector_static(
                    self,
                    p=2,
                    axis=-1,
                    shape_x=[2, 3, 4],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_vector_static(
                    self,
                    p=1,
                    axis=[0, 1],
                    shape_x=[2, 3, 4, 5],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_vector_static(
                    self,
                    p=np.inf,
                    axis=[0, 1],
                    shape_x=[2, 3, 4],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_vector_static(
                    self,
                    p=-np.inf,
                    axis=[0, 1, 2],
                    shape_x=[2, 3, 4],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_vector_static(
                    self,
                    p=2,
                    axis=None,
                    shape_x=[],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_vector_static(
                    self,
                    p=np.inf,
                    axis=None,
                    shape_x=[],
                    dtype="complex64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_vector_static(
                    self,
                    p=-np.inf,
                    axis=[0, 1, 2, 3],
                    shape_x=[1, 14, 5, 14],
                    dtype="complex128",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_vector_static(
                    self,
                    p=np.inf,
                    axis=2,
                    shape_x=[1, 14, 5, 14],
                    dtype="complex128",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_vector_static(
                    self,
                    p=0,
                    axis=[1, 3],
                    shape_x=[1, 14, 5, 14],
                    dtype="complex128",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_matrix_static(
                    self,
                    p=-np.inf,
                    axis=[0, 1],
                    shape_x=[2, 3, 4],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_matrix_static(
                    self,
                    p='fro',
                    axis=[0, 1],
                    shape_x=[2, 3, 4],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_matrix_static(
                    self,
                    p='nuc',
                    axis=[0, 1],
                    shape_x=[2, 3, 4],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_matrix_static(
                    self,
                    p=-2,
                    axis=[1, 2],
                    shape_x=[2, 3, 4, 5],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_matrix_static(
                    self,
                    p=-np.inf,
                    axis=[-2, -1],
                    shape_x=[0, 1, 2, 1],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_matrix_static(
                    self,
                    p="fro",
                    axis=[-2, -1],
                    shape_x=[0, 1, 2, 1],
                    dtype="float64",
                    keep_dim=keep,
                    check_dim=True,
                )
                check_linalg_matrix_static(
                    self,
                    p="fro",
                    axis=[-2, -1],
                    shape_x=[3, 2, 1],
                    dtype="complex64",
                    keep_dim=keep,
                )
                check_linalg_matrix_static(
                    self,
                    p="fro",
                    axis=[-2, -1],
                    shape_x=[3, 2, 1],
                    dtype="complex128",
                    keep_dim=keep,
                )

    def test_dygraph(self):
        paddle.disable_static()
        keep_dims = {False, True}
        for keep in keep_dims:
            check_fro_dygraph(
                self,
                p='fro',
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype='float64',
                keep_dim=keep,
                check_dim=True,
            )
            check_fro_dygraph(
                self,
                p='fro',
                axis=[1, 2],
                shape_x=[2, 3, 4, 5],
                dtype='float64',
                keep_dim=keep,
                check_dim=True,
            )

            check_nuc_dygraph(
                self,
                p='nuc',
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype='float64',
                keep_dim=keep,
                check_dim=True,
            )
            check_nuc_dygraph(
                self,
                p='nuc',
                axis=[1, 2],
                shape_x=[2, 3, 4, 5],
                dtype='float64',
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_dygraph(
                self,
                p=2,
                axis=None,
                shape_x=[3, 4],
                dtype="float32",
                keep_dim=keep,
            )
            check_linalg_norm_dygraph(
                self,
                p=2,
                axis=1,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_dygraph(
                self,
                p=np.inf,
                axis=0,
                shape_x=[2, 3, 4],
                dtype="float32",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_dygraph(
                self,
                p=np.inf,
                axis=None,
                shape_x=[2, 3, 4],
                dtype="float32",
                keep_dim=keep,
            )
            check_linalg_norm_dygraph(
                self,
                p=-np.inf,
                axis=0,
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_dygraph(
                self,
                p=-np.inf,
                axis=None,
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
            )
            check_linalg_norm_dygraph(
                self,
                p=0,
                axis=1,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )

            check_linalg_norm_dygraph(
                self,
                p=1,
                axis=1,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_dygraph(
                self,
                p=0,
                axis=None,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_dygraph(
                self,
                p=2,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_dygraph(
                self,
                p=2,
                axis=-1,
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_dygraph(
                self,
                p=1,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_dygraph(
                self,
                p=np.inf,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_dygraph(
                self,
                p=-np.inf,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )

            check_linalg_vector_dygraph(
                self,
                p=2,
                axis=None,
                shape_x=[3, 4],
                dtype="float32",
                keep_dim=keep,
            )
            check_linalg_vector_dygraph(
                self,
                p=2,
                axis=1,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=np.inf,
                axis=0,
                shape_x=[2, 3, 4],
                dtype="float32",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=np.inf,
                axis=None,
                shape_x=[2, 3, 4],
                dtype="float32",
                keep_dim=keep,
            )
            check_linalg_vector_dygraph(
                self,
                p=-np.inf,
                axis=0,
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=-np.inf,
                axis=None,
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
            )
            check_linalg_vector_dygraph(
                self,
                p=0,
                axis=1,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )

            check_linalg_vector_dygraph(
                self,
                p=1,
                axis=1,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=0,
                axis=None,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=2,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=2,
                axis=-1,
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=1,
                axis=[0, 1],
                shape_x=[2, 3, 4, 5],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=np.inf,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=-np.inf,
                axis=[0, 1, 2],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=2,
                axis=None,
                shape_x=(),
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=np.inf,
                axis=None,
                shape_x=[],
                dtype="complex64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=-np.inf,
                axis=[0, 1, 2, 3],
                shape_x=[1, 14, 5, 14],
                dtype="complex128",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=np.inf,
                axis=2,
                shape_x=[1, 14, 5, 14],
                dtype="complex128",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=0,
                axis=[1, 3],
                shape_x=[1, 14, 5, 14],
                dtype="complex128",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_matrix_dygraph(
                self,
                p=-np.inf,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_matrix_dygraph(
                self,
                p='fro',
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_matrix_dygraph(
                self,
                p='nuc',
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_matrix_dygraph(
                self,
                p=-2,
                axis=[1, 2],
                shape_x=[2, 3, 4, 5],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_matrix_dygraph(
                self,
                p=-np.inf,
                axis=[-2, -1],
                shape_x=[0, 1, 2, 1],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_matrix_dygraph(
                self,
                p="fro",
                axis=[-2, -1],
                shape_x=[0, 1, 2, 1],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_matrix_dygraph(
                self,
                p="fro",
                axis=[-2, -1],
                shape_x=[3, 2, 1],
                dtype="complex64",
                keep_dim=keep,
            )
            check_linalg_matrix_dygraph(
                self,
                p="fro",
                axis=[-2, -1],
                shape_x=[3, 2, 1],
                dtype="complex128",
                keep_dim=keep,
            )
        paddle.enable_static()

    def test_name(self):
        if not paddle.framework.use_pir_api():
            paddle.enable_static()
            with base.program_guard(base.Program()):
                x = paddle.static.data(
                    name="x", shape=[10, 10], dtype="float32"
                )
                y_1 = paddle.norm(
                    x, p='fro', axis=[-2, -1], name='frobenius_name'
                )
                y_2 = paddle.norm(x, p=2, name='pnorm_name')
                y_3 = paddle.norm(x, p='nuc', axis=[0, 1], name='nuclear_name')
                y_4 = paddle.norm(
                    x, p=2, axis=[0, 1], name='p_matrix_norm_name'
                )
                self.assertEqual(('frobenius_name' in y_1.name), True)
                self.assertEqual(('pnorm_name' in y_2.name), True)
                self.assertEqual(('nuclear_name' in y_3.name), True)
                self.assertEqual(('p_matrix_norm_name' in y_4.name), True)

    def test_errors(self):
        paddle.enable_static()
        with base.program_guard(base.Program(), base.Program()):

            def err_dtype(p, shape_x, xdtype, out=None):
                data = paddle.static.data(shape=shape_x, dtype=xdtype)
                paddle.norm(data, p=p, out=out)

            self.assertRaises(TypeError, err_dtype, "fro", [2, 2], "int64")
            self.assertRaises(ValueError, paddle.norm, "inf", [2], "int64")
            out = paddle.static.data(name="out", shape=[1], dtype="int64")
            self.assertRaises(
                TypeError, err_dtype, "fro", [2, 2], "float64", out
            )
            self.assertRaises(TypeError, err_dtype, 2, [10], "int64")
            self.assertRaises(TypeError, err_dtype, 2, [10], "float64", out)

            data = paddle.static.data(
                name="data_2d", shape=[2, 2], dtype="float64"
            )
            self.assertRaises(
                ValueError, paddle.norm, data, p="unsupported norm"
            )
            self.assertRaises(ValueError, paddle.norm, data, p=[1])
            self.assertRaises(ValueError, paddle.norm, data, p=[1], axis=-1)
            self.assertRaises(ValueError, paddle.norm, 0, [1, 0], "float64")
            data = paddle.static.data(
                name="data_3d", shape=[2, 2, 2], dtype="float64"
            )
            self.assertRaises(
                ValueError, paddle.norm, data, p='unspport', axis=[-3, -2, -1]
            )


class API_NormTest_ZeroSize(unittest.TestCase):
    def check_linalg_norm_dygraph_and_grad(
        self, p, axis, shape_x, dtype, keep_dim, check_dim=False
    ):
        x_numpy = (np.random.random(shape_x) + 1.0).astype(dtype)
        expected_result = np_linalg_norm(
            x_numpy, porder=p, axis=axis, keepdims=keep_dim
        )
        x_paddle = paddle.to_tensor(x_numpy)
        x_paddle.stop_gradient = False
        result1 = paddle.linalg.norm(
            x=x_paddle, p=p, axis=axis, keepdim=keep_dim
        )
        result = result1.numpy()
        np.testing.assert_allclose(
            result, expected_result, rtol=1e-6, atol=1e-8
        )
        if keep_dim and check_dim:
            np.testing.assert_equal(result.shape, expected_result.shape)
        loss = paddle.sum(result1)
        loss.backward()
        np.testing.assert_equal(x_paddle.grad.shape, x_paddle.shape)

    def check_linalg_vector_dygraph_and_grad(
        self, p, axis, shape_x, dtype, keep_dim, check_dim=False
    ):
        x_numpy = np.array(np.random.random(shape_x) + 1.0).astype(dtype)
        expected_result = np_linalg_vector_norm(
            x_numpy, porder=p, axis=axis, keepdims=keep_dim
        )
        x_paddle = paddle.to_tensor(x_numpy)
        x_paddle.stop_gradient = False
        result1 = paddle.linalg.vector_norm(
            x=x_paddle, p=p, axis=axis, keepdim=keep_dim
        )
        result = result1.numpy()
        np.testing.assert_allclose(
            result, expected_result, rtol=1e-6, atol=1e-8
        )
        if keep_dim and check_dim:
            np.testing.assert_equal(result.shape, expected_result.shape)
        loss = paddle.sum(result1)
        loss.backward()
        np.testing.assert_equal(x_paddle.grad.shape, x_paddle.shape)

    def check_linalg_matrix_dygraph_and_grad(
        self, p, axis, shape_x, dtype, keep_dim, check_dim=False
    ):
        x_numpy = (np.random.random(shape_x) + 1.0).astype(dtype)
        expected_result = np_linalg_matrix_norm(
            x_numpy, porder=p, axis=axis, keepdims=keep_dim
        )
        x_paddle = paddle.to_tensor(x_numpy)
        x_paddle.stop_gradient = False
        result1 = paddle.linalg.matrix_norm(
            x=x_paddle, p=p, axis=axis, keepdim=keep_dim
        )
        result = result1.numpy()
        np.testing.assert_allclose(
            result, expected_result, rtol=1e-6, atol=1e-8
        )
        if keep_dim and check_dim:
            np.testing.assert_equal(result.shape, expected_result.shape)
        loss = paddle.sum(result1)
        loss.backward()
        np.testing.assert_equal(x_paddle.grad.shape, x_paddle.shape)

    def test_dygraph(self):
        paddle.disable_static()
        keep_dims = {False, True}
        for keep in keep_dims:
            self.check_linalg_norm_dygraph_and_grad(
                p=1,
                axis=[0, 1],
                shape_x=[0, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )

            self.check_linalg_vector_dygraph_and_grad(
                p=np.inf,
                axis=[1],
                shape_x=[0, 3, 4],
                dtype="float32",
                keep_dim=keep,
            )

            self.check_linalg_matrix_dygraph_and_grad(
                p=np.inf,
                axis=[1, 2],
                shape_x=[0, 3, 4],
                dtype="float32",
                keep_dim=keep,
            )


class API_PnormGradTest_ZeroSize(unittest.TestCase):
    """Cover the 0-size early-return branch in p_norm_grad CPU/GPU kernels.

    The PR adds `if (out_dx->numel() == 0) return;` to both
    paddle/phi/kernels/cpu/p_norm_grad_kernel.cc and
    paddle/phi/kernels/gpu/p_norm_grad_kernel.cu. Running backward on a
    0-size input exercises that branch for every porder dispatch arm
    (porder == 0, 1, 2, <1, in (1,2), >2, +/-inf).
    """

    def _run_pnorm_backward(self, shape, axis, porder, dtype="float32"):
        x_np = (np.random.random(shape) + 1.0).astype(dtype)
        x = paddle.to_tensor(x_np)
        x.stop_gradient = False
        out = paddle.linalg.vector_norm(x=x, p=porder, axis=axis, keepdim=False)
        loss = paddle.sum(out)
        loss.backward()
        # 0-size input -> 0-size grad, with shape preserved.
        np.testing.assert_equal(x.grad.shape, x.shape)
        self.assertEqual(x.grad.numel().item(), 0)

    def test_dygraph_zero_size_grad(self):
        paddle.disable_static()
        # Trigger every porder branch in PNormGradKernel so each new
        # `out_dx->numel() == 0` early-return is exercised.
        porders = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, float("inf"), float("-inf")]
        shape = [0, 3, 4]
        for axis in [0, 1, -1]:
            for p in porders:
                self._run_pnorm_backward(shape, axis, p)


class TestPnormGradCompatKernel(unittest.TestCase):
    """Test the Compat=true path (FLAGS_use_accuracy_compatible_kernel=1).

    Verifies that the fused GPU kernel reproduces torch-compatible intermediate
    rounding behavior for fp16/bf16. The reference is computed by simulating
    torch's multi-kernel decomposition: each intermediate result is rounded
    back to storage dtype before the next operation.
    """

    def setUp(self):
        paddle.disable_static()
        paddle.set_flags({'FLAGS_use_accuracy_compatible_kernel': True})

    def tearDown(self):
        paddle.set_flags({'FLAGS_use_accuracy_compatible_kernel': False})

    @staticmethod
    def _round_to_storage(val, dtype):
        """Simulate RoundToStorage: round float32 intermediate to dtype and back."""
        if dtype == "bfloat16":
            # Use round-to-nearest-even (RNE) matching CUDA __float2bfloat16.
            # NOTE: convert_float_to_uint16 uses truncation which does NOT
            # match the kernel's rounding behavior.
            arr = np.asarray(val, dtype=np.float32)
            bits = arr.view(np.uint32)
            rounding_bias = np.uint32(0x7FFF) + (
                (bits >> np.uint32(16)) & np.uint32(1)
            )
            rounded_bits = bits + rounding_bias
            bf16_uint16 = (rounded_bits >> np.uint32(16)).astype(np.uint16)
            result_bits = bf16_uint16.astype(np.uint32) << np.uint32(16)
            result = result_bits.view(np.float32).copy()
            result[np.isnan(arr)] = np.float32(np.nan)
            result[np.isinf(arr)] = arr[np.isinf(arr)]
            result[arr == 0] = arr[arr == 0]
            return result
        return val.astype(dtype).astype("float32")

    def _pow_like_torch(self, val_f32, exponent, dtype):
        """Simulate torch's pow kernel for half/bf16: compute in float32,
        truncate result to storage dtype."""
        if exponent == 2.0:
            result = val_f32 * val_f32
        elif exponent == 3.0:
            tmp = self._round_to_storage(val_f32 * val_f32, dtype)
            result = tmp * val_f32
        elif exponent == -1.0:
            result = np.float32(1.0) / val_f32
        elif exponent == -2.0:
            sq = self._round_to_storage(val_f32 * val_f32, dtype)
            result = np.float32(1.0) / sq
        else:
            result = np.power(val_f32, np.float32(exponent))
        return self._round_to_storage(result, dtype)

    def _reference_p2_grad(self, x_np, norm_np, grad_np, dtype):
        """Reference for p=2: grad * (x / norm), masked where norm==0."""
        x_f = x_np.astype("float32")
        norm_f = norm_np.astype("float32")
        grad_f = grad_np.astype("float32")
        x_div_norm = self._round_to_storage(x_f / norm_f, dtype)
        dx = self._round_to_storage(grad_f * x_div_norm, dtype)
        # Broadcast norm_f == 0 mask to match dx shape
        mask = np.broadcast_to(norm_f == 0, dx.shape)
        dx[mask] = 0
        return dx

    def _reference_pgt2_grad(self, x_np, norm_np, grad_np, porder, dtype):
        """Reference for p>2: x * |x|^(p-2) * grad / norm^(p-1), masked."""
        x_f = x_np.astype("float32")
        norm_f = norm_np.astype("float32")
        grad_f = grad_np.astype("float32")
        abs_x = np.abs(x_f)

        abs_pow = self._pow_like_torch(abs_x, porder - 2, dtype)
        self_scaled = self._round_to_storage(x_f * abs_pow, dtype)
        norm_pow = self._pow_like_torch(norm_f, porder - 1, dtype)
        scale_v = self._round_to_storage(grad_f / norm_pow, dtype)
        dx = self._round_to_storage(self_scaled * scale_v, dtype)
        mask = np.broadcast_to(norm_f == 0, dx.shape)
        dx[mask] = 0
        return dx

    def _reference_p_between_1_and_2_grad(
        self, x_np, norm_np, grad_np, porder, dtype
    ):
        """Reference for 1<p<2: sign(x)*|x|^(p-1) * grad / norm^(p-1)."""
        x_f = x_np.astype("float32")
        norm_f = norm_np.astype("float32")
        grad_f = grad_np.astype("float32")
        abs_x = np.abs(x_f)
        sign_x = np.sign(x_f)

        abs_pow = self._pow_like_torch(abs_x, porder - 1, dtype)
        self_scaled = self._round_to_storage(sign_x * abs_pow, dtype)
        norm_pow = self._pow_like_torch(norm_f, porder - 1, dtype)
        scale_v = self._round_to_storage(grad_f / norm_pow, dtype)
        dx = self._round_to_storage(self_scaled * scale_v, dtype)
        mask = np.broadcast_to(norm_f == 0, dx.shape)
        dx[mask] = 0
        return dx

    def _reference_p_less_than_1_grad(
        self, x_np, norm_np, grad_np, porder, dtype
    ):
        """Reference for p<1: sign(x)*|x|^(p-1) * grad * norm^(1-p), masked x==0."""
        x_f = x_np.astype("float32")
        norm_f = norm_np.astype("float32")
        grad_f = grad_np.astype("float32")
        abs_x = np.abs(x_f)
        sign_x = np.sign(x_f)

        abs_pow = self._pow_like_torch(abs_x, porder - 1, dtype)
        self_scaled = self._round_to_storage(sign_x * abs_pow, dtype)
        temp1 = self._round_to_storage(self_scaled * grad_f, dtype)
        norm_pow = self._pow_like_torch(norm_f, 1.0 - porder, dtype)
        dx = self._round_to_storage(temp1 * norm_pow, dtype)
        # mask: x == 0 → dx = 0
        mask = np.broadcast_to(x_f == 0, dx.shape)
        dx[mask] = 0
        return dx

    def _run_compat_test(self, shape, porder, axis, dtype_str):
        """Run paddle backward and compare against reference with rounding."""
        np.random.seed(2024)
        x_f32 = ((np.random.random(shape) - 0.5) * 1.2).astype("float32")
        # Round to storage dtype for numpy reference
        x_np = self._round_to_storage(x_f32, dtype_str)

        # Create paddle tensor
        if dtype_str == "bfloat16":
            x = paddle.to_tensor(x_np, place=paddle.CUDAPlace(0)).cast(
                "bfloat16"
            )
        else:
            x_np_stored = x_np.astype(dtype_str)
            x = paddle.to_tensor(x_np_stored, place=paddle.CUDAPlace(0))
        x.stop_gradient = False
        out = paddle.linalg.vector_norm(x=x, p=porder, axis=axis, keepdim=True)

        np.random.seed(100)
        grad_f32 = ((np.random.random(out.shape) - 0.5) * 0.8).astype("float32")
        grad_np = self._round_to_storage(grad_f32, dtype_str)
        if dtype_str == "bfloat16":
            grad_tensor = paddle.to_tensor(
                grad_np, place=paddle.CUDAPlace(0)
            ).cast("bfloat16")
        else:
            grad_np_stored = grad_np.astype(dtype_str)
            grad_tensor = paddle.to_tensor(
                grad_np_stored, place=paddle.CUDAPlace(0)
            )

        paddle_grad = paddle.grad(
            outputs=[out], inputs=[x], grad_outputs=[grad_tensor]
        )
        if dtype_str == "bfloat16":
            paddle_dx = paddle_grad[0].cast("float32").numpy()
            norm_np = out.cast("float32").numpy()
        else:
            paddle_dx = paddle_grad[0].numpy().astype("float32")
            norm_np = out.numpy().astype("float32")

        # Round paddle results to storage precision for comparison
        paddle_dx = self._round_to_storage(paddle_dx, dtype_str)
        norm_np = self._round_to_storage(norm_np, dtype_str)

        # Build reference
        if porder == 2.0:
            ref_dx = self._reference_p2_grad(x_np, norm_np, grad_np, dtype_str)
        elif porder > 2.0:
            ref_dx = self._reference_pgt2_grad(
                x_np, norm_np, grad_np, porder, dtype_str
            )
        elif 1.0 < porder < 2.0:
            ref_dx = self._reference_p_between_1_and_2_grad(
                x_np, norm_np, grad_np, porder, dtype_str
            )
        elif 0 < porder < 1.0:
            ref_dx = self._reference_p_less_than_1_grad(
                x_np, norm_np, grad_np, porder, dtype_str
            )
        else:
            return  # skip unsupported ranges in this helper

        np.testing.assert_array_equal(
            paddle_dx,
            ref_dx,
            err_msg=f"Compat mismatch: p={porder}, dtype={dtype_str}, "
            f"shape={shape}, axis={axis}",
        )

    @unittest.skipIf(not core.is_compiled_with_cuda(), "CUDA not available")
    def test_p2_fp16_compat(self):
        self._run_compat_test((4, 8, 16), 2.0, -1, "float16")

    @unittest.skipIf(
        not core.is_compiled_with_cuda()
        or not core.is_bfloat16_supported(core.CUDAPlace(0))
        or core.is_compiled_with_rocm(),
        "CUDA bf16 not available or running on ROCm/DCU",
    )
    def test_p2_bf16_compat(self):
        self._run_compat_test((4, 8, 16), 2.0, -1, "bfloat16")

    @unittest.skipIf(not core.is_compiled_with_cuda(), "CUDA not available")
    def test_p4_fp16_compat(self):
        self._run_compat_test((4, 5, 32), 4.0, -1, "float16")

    @unittest.skipIf(
        not core.is_compiled_with_cuda()
        or not core.is_bfloat16_supported(core.CUDAPlace(0))
        or core.is_compiled_with_rocm(),
        "CUDA bf16 not available or running on ROCm/DCU",
    )
    def test_p4_bf16_compat(self):
        self._run_compat_test((4, 5, 32), 4.0, -1, "bfloat16")

    @unittest.skipIf(not core.is_compiled_with_cuda(), "CUDA not available")
    def test_p3_fp16_compat(self):
        self._run_compat_test((3, 6, 16), 3.0, -1, "float16")

    @unittest.skipIf(not core.is_compiled_with_cuda(), "CUDA not available")
    def test_p1_5_fp16_compat(self):
        self._run_compat_test((4, 8, 16), 1.5, -1, "float16")

    @unittest.skipIf(
        not core.is_compiled_with_cuda()
        or not core.is_bfloat16_supported(core.CUDAPlace(0))
        or core.is_compiled_with_rocm(),
        "CUDA bf16 not available or running on ROCm/DCU",
    )
    def test_p1_5_bf16_compat(self):
        self._run_compat_test((4, 8, 16), 1.5, -1, "bfloat16")

    @unittest.skipIf(not core.is_compiled_with_cuda(), "CUDA not available")
    def test_p0_5_fp16_compat(self):
        self._run_compat_test((4, 8, 16), 0.5, -1, "float16")

    @unittest.skipIf(
        not core.is_compiled_with_cuda()
        or not core.is_bfloat16_supported(core.CUDAPlace(0))
        or core.is_compiled_with_rocm(),
        "CUDA bf16 not available or running on ROCm/DCU",
    )
    def test_p0_5_bf16_compat(self):
        self._run_compat_test((4, 8, 16), 0.5, -1, "bfloat16")

    @unittest.skipIf(not core.is_compiled_with_cuda(), "CUDA not available")
    def test_p2_fp16_axis0_compat(self):
        self._run_compat_test((8, 4, 16), 2.0, 0, "float16")

    @unittest.skipIf(not core.is_compiled_with_cuda(), "CUDA not available")
    def test_p4_fp16_reduce_all_compat(self):
        """reduce_all path with Compat."""
        np.random.seed(2024)
        dtype_str = "float16"
        shape = (4, 5, 6)
        x_np = ((np.random.random(shape) - 0.5) * 1.2).astype(dtype_str)

        x = paddle.to_tensor(x_np, place=paddle.CUDAPlace(0))
        x.stop_gradient = False
        out = paddle.linalg.vector_norm(x=x, p=4.0, keepdim=True)

        np.random.seed(100)
        grad_np = ((np.random.random(out.shape) - 0.5) * 0.8).astype(dtype_str)
        grad_tensor = paddle.to_tensor(grad_np, place=paddle.CUDAPlace(0))

        paddle_grad = paddle.grad(
            outputs=[out], inputs=[x], grad_outputs=[grad_tensor]
        )
        paddle_dx = paddle_grad[0].numpy()

        norm_np = out.numpy()
        ref_dx = self._reference_pgt2_grad(
            x_np, norm_np, grad_np, 4.0, dtype_str
        )
        np.testing.assert_array_equal(paddle_dx, ref_dx)


class API_NormTest_Alias(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def test_alias(self):
        """
        Test the alias of norm function.
        ``norm(x=x, axis=1)`` is equivalent to ``norm(input=x, dim=1)``
        """
        shape_cases = [
            [2, 3, 4],
            [3, 4, 5],
        ]
        p_cases = [2, 'fro', 'nuc', np.inf, -np.inf, 1, -1]
        axis_cases = [None, 1, [0, 1], [-2, -1]]

        for shape in shape_cases:
            x = paddle.rand(shape)
            for p in p_cases:
                for axis in axis_cases:
                    # Skip invalid combinations
                    if p == 'fro' and (axis is None or isinstance(axis, int)):
                        continue
                    if p == 'nuc' and (axis is None or isinstance(axis, int)):
                        continue

                    # Test x/input alias
                    kwargs1 = {'x': x, 'p': p, 'axis': axis}
                    kwargs2 = {'input': x, 'p': p, 'axis': axis}

                    out1 = paddle.norm(**kwargs1).numpy()
                    out2 = paddle.norm(**kwargs2).numpy()
                    np.testing.assert_allclose(out1, out2, rtol=1e-6, atol=1e-8)

                    # Test axis/dim alias
                    kwargs3 = {'x': x, 'p': p, 'dim': axis}
                    out3 = paddle.norm(**kwargs3).numpy()
                    np.testing.assert_allclose(out1, out3, rtol=1e-6, atol=1e-8)

                    # Test both aliases together
                    kwargs4 = {'input': x, 'p': p, 'dim': axis}
                    out4 = paddle.norm(**kwargs4).numpy()
                    np.testing.assert_allclose(out1, out4, rtol=1e-6, atol=1e-8)

    def test_static_alias(self):
        """
        Test alias in static mode
        """
        paddle.enable_static()
        with base.program_guard(base.Program()):
            x = paddle.static.data(name='x', shape=[2, 3, 4], dtype='float32')

            # Test x/input alias
            out1 = paddle.norm(x=x, p=2, axis=1)
            out2 = paddle.norm(input=x, p=2, axis=1)

            # Test axis/dim alias
            out3 = paddle.norm(x=x, p=2, dim=1)
            out4 = paddle.norm(input=x, p=2, dim=1)

            place = base.CPUPlace()
            exe = base.Executor(place)
            x_np = np.random.random([2, 3, 4]).astype('float32')
            res1, res2, res3, res4 = exe.run(
                feed={'x': x_np}, fetch_list=[out1, out2, out3, out4]
            )

            np.testing.assert_allclose(res1, res2, rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(res1, res3, rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(res1, res4, rtol=1e-6, atol=1e-8)
        paddle.disable_static()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
