# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import unittest
from os.path import dirname

import numpy as np

import paddle
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))
import utils


def tril(x):
    return paddle.tril(x)


def reciprocal(x):
    return paddle.reciprocal(x)


def isinf(x):
    return paddle.isinf(x)


def isfinite(x):
    return paddle.isfinite(x)


def isnan(x):
    return paddle.isnan(x)


def tril_diag_neg(x):
    return paddle.tril(x, -1)


def tril_diag_pos(x):
    return paddle.tril(x, 1)


def isclose(x, y):
    return paddle.isclose(
        x, y, rtol=1e-05, atol=1e-08, equal_nan=False, name=None
    )


def reverse(x):
    return paddle.flip(x, axis=[1])


class CINNSubGraphNet(paddle.nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        out = self.fn(x)
        return out


class CINNSubGraphNetBinary(paddle.nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, y):
        out = self.fn(x, y)
        return out


class TestCinnSubGrapTril(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.x_shape = [32, 32]
        self.x = paddle.randn(self.x_shape, dtype="float32")
        self.x.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)

    def eval_symbolic(self, use_cinn):
        paddle.seed(2022)
        net = CINNSubGraphNet(tril)
        input_spec = [
            InputSpec(shape=[None, 32], dtype='float32'),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval_symbolic(self):
        cinn_out = self.eval_symbolic(use_cinn=True)
        dy_out = self.eval_symbolic(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class TestCinnSubGrapTrilBoolGE2Dim(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.x_shape = [32, 32, 64]
        self.x = paddle.randint(0, 2, self.x_shape)
        self.x.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)

    def eval_symbolic(self, use_cinn):
        paddle.seed(2022)
        net = CINNSubGraphNet(tril)
        input_spec = [
            InputSpec(shape=[None, 32, 64], dtype='bool'),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval_symbolic(self):
        cinn_out = self.eval_symbolic(use_cinn=True)
        dy_out = self.eval_symbolic(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class TestCinnSubGrapTrilDiagNeg(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.x_shape = [32, 32]
        self.x = paddle.randn(self.x_shape, dtype="float32")
        self.x.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)

    def eval_symbolic(self, use_cinn):
        paddle.seed(2022)
        net = CINNSubGraphNet(tril_diag_neg)
        input_spec = [
            InputSpec(shape=[None, 32], dtype='float32'),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval_symbolic(self):
        cinn_out = self.eval_symbolic(use_cinn=True)
        dy_out = self.eval_symbolic(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class TestCinnSubGrapTrilDiagPos(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.x_shape = [64, 128]
        self.x = paddle.randn(self.x_shape, dtype="float32")
        self.x.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)

    def eval_symbolic(self, use_cinn):
        paddle.seed(2022)
        net = CINNSubGraphNet(tril_diag_pos)
        input_spec = [
            InputSpec(shape=[None, 128], dtype='float32'),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval_symbolic(self):
        cinn_out = self.eval_symbolic(use_cinn=True)
        dy_out = self.eval_symbolic(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class TestCinnSubGrapIsInf(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.x_shape = [32, 32]
        self.x = paddle.randn(self.x_shape, dtype="float32")
        self.x.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)

    def eval_symbolic(self, use_cinn):
        paddle.seed(2022)
        net = CINNSubGraphNet(isinf)
        input_spec = [
            InputSpec(shape=[None, 32], dtype='float32'),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval_symbolic(self):
        cinn_out = self.eval_symbolic(use_cinn=True)
        dy_out = self.eval_symbolic(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class TestCinnSubGrapIsFinite(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.x_shape = [32, 32]
        self.x = paddle.randn(self.x_shape, dtype="float32")
        self.x.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)

    def eval_symbolic(self, use_cinn):
        paddle.seed(2022)
        net = CINNSubGraphNet(isfinite)
        input_spec = [
            InputSpec(shape=[None, 32], dtype='float32'),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval_symbolic(self):
        cinn_out = self.eval_symbolic(use_cinn=True)
        dy_out = self.eval_symbolic(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class TestCinnSubGrapIsNan(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.x_shape = [32, 32]
        self.x = paddle.randn(self.x_shape, dtype="float32")
        self.x.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)

    def eval_symbolic(self, use_cinn):
        paddle.seed(2022)
        net = CINNSubGraphNet(isnan)
        input_spec = [
            InputSpec(shape=[None, 32], dtype='float32'),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval_symbolic(self):
        cinn_out = self.eval_symbolic(use_cinn=True)
        dy_out = self.eval_symbolic(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class TestCinnSubGraphIscloseFalse(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        x_shape = [32, 32]
        y_shape = [32, 32]
        tensor_x = np.random.random(x_shape).astype("float32")
        tensor_y = np.random.random(y_shape).astype("float32")
        self.x = paddle.to_tensor(tensor_x)
        self.y = paddle.to_tensor(tensor_y)
        self.x.stop_gradient = False
        self.y.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)

    def eval_symbolic(self, use_cinn):
        paddle.seed(2022)
        net = CINNSubGraphNetBinary(isclose)
        input_spec = [
            InputSpec(shape=[None, 32], dtype='float32'),
            InputSpec(shape=[None, 32], dtype='float32'),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x, self.y)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval_symbolic(self):
        cinn_out = self.eval_symbolic(use_cinn=True)
        dy_out = self.eval_symbolic(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class TestCinnSubGraphIscloseTrue(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        x_shape = [32, 32]
        y_shape = [32, 32]
        tensor_x = np.random.random(x_shape).astype("float32")
        tensor_y = np.random.random(y_shape).astype("float32")
        tensor_y[0] = tensor_x[0]
        self.x = paddle.to_tensor(tensor_x)
        self.y = paddle.to_tensor(tensor_y)
        self.x.stop_gradient = False
        self.y.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)

    def eval_symbolic(self, use_cinn):
        paddle.seed(2022)
        net = CINNSubGraphNetBinary(isclose)
        input_spec = [
            InputSpec(shape=[None, 32], dtype='float32'),
            InputSpec(shape=[None, 32], dtype='float32'),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x, self.y)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval_symbolic(self):
        cinn_out = self.eval_symbolic(use_cinn=True)
        dy_out = self.eval_symbolic(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class TestCinnSubGrapReciprocal(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.x_shape = [32, 32]
        self.x = paddle.randn(self.x_shape, dtype="float32")
        self.x.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)

    def eval_symbolic(self, use_cinn):
        paddle.seed(2022)
        net = CINNSubGraphNet(reciprocal)
        input_spec = [
            InputSpec(shape=[None, 32], dtype='float32'),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval_symbolic(self):
        cinn_out = self.eval_symbolic(use_cinn=True)
        dy_out = self.eval_symbolic(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class TestCinnSubGraphFlipOrReverse(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.x_shape = [32, 32]
        self.x = paddle.randn(self.x_shape, dtype="float32")
        self.x.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)

    def eval_symbolic(self, use_cinn):
        paddle.seed(2022)
        net = CINNSubGraphNet(reverse)
        input_spec = [
            InputSpec(shape=[None, 32], dtype='float32'),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval_symbolic(self):
        cinn_out = self.eval_symbolic(use_cinn=True)
        dy_out = self.eval_symbolic(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
