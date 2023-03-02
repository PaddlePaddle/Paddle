#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid.core as core
import paddle.nn.functional as F

np.random.seed(100)


def ref_poisson_nll_loss(
    input,
    label,
    log_input=True,
    full=False,
    epsilon=1e-8,
    reduction="mean",
):
    if epsilon <= 0:
        raise ValueError(
            "The value of `epsilon` in PoissonNLLLoss should be positve, but received %f, which is not allowed"
            % epsilon
        )

    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in SoftMarginLoss should be 'sum', 'mean' or 'none', but "
            "received %s, which is not allowed." % reduction
        )
    loss_out = 0
    if log_input:
        loss_out = np.exp(input) - label * input
    else:
        loss_out = input - label * np.log(input + epsilon)
    if full:
        stirling_approx = (
            label * np.log(label) - label + 0.5 * np.log(2 * np.pi * label)
        )
        loss_out += np.where(stirling_approx <= 1, 0, stirling_approx)

    if reduction == 'none':
        return loss_out
    elif reduction == 'sum':
        return [np.sum(loss_out)]
    elif reduction == 'mean':
        return [np.mean(loss_out)]


class TestPoissonNLLLossBasicCase(unittest.TestCase):
    def setUp(self, dtype="float32"):
        self.shape = [10, 2]
        self.dtype = dtype
        self.input_np = np.random.random(self.shape).astype(self.dtype)
        self.label_np = np.random.random(self.shape).astype(self.dtype)
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_api(self):
        pass


class TestPoissonNLLLossStaticCase(TestPoissonNLLLossBasicCase):
    def setUp(self, dtype="float32"):
        super().setUp(dtype)

    def test_static_case(
        self,
        dtype="float32",
        log_input=True,
        full=False,
        epsilon=1e-8,
        reduction="mean",
    ):
        self.setUp(dtype)
        paddle.enable_static()
        prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(prog, startup_prog):
            input = paddle.static.data('input', self.shape, dtype)
            label = paddle.static.data('label', self.shape, dtype)
            input.desc.set_need_check_feed(False)
            label.desc.set_need_check_feed(False)
            out1 = F.poisson_nll_loss(
                input,
                label,
                log_input=log_input,
                full=full,
                epsilon=epsilon,
                reduction=reduction,
            )
            poisson_nll_loss = paddle.nn.PoissonNLLLoss(
                log_input=log_input,
                full=full,
                epsilon=epsilon,
                reduction=reduction,
            )
            out2 = poisson_nll_loss(input, label)
        exe = paddle.static.Executor(self.place)
        exe.run(startup_prog)
        res = exe.run(
            prog,
            feed={'input': self.input_np, 'label': self.label_np},
            fetch_list=[out1, out2],
        )
        out_ref = ref_poisson_nll_loss(
            self.input_np,
            self.label_np,
            log_input=log_input,
            full=full,
            epsilon=epsilon,
            reduction=reduction,
        )
        for r in res:
            np.allclose(out_ref, r, rtol=1e-5)


class TestPoissonNLLLossDynamicCase(TestPoissonNLLLossBasicCase):
    def setUp(self, dtype="float32"):
        super().setUp(dtype)

    def test_dynamic_case(
        self,
        dtype="float32",
        log_input=True,
        full=False,
        epsilon=1e-8,
        reduction="mean",
        type=None,
    ):
        self.setUp(dtype)
        paddle.disable_static(self.place)

        input_x = paddle.to_tensor(self.input_np)
        label = paddle.to_tensor(self.label_np)
        out_ref = ref_poisson_nll_loss(
            self.input_np,
            self.label_np,
            log_input=log_input,
            full=full,
            epsilon=epsilon,
            reduction=reduction,
        )
        out1 = F.poisson_nll_loss(
            input_x,
            label,
            log_input=log_input,
            full=full,
            epsilon=epsilon,
            reduction=reduction,
        )
        if type == 'test_err_reduction':
            self.assertRaises(
                ValueError,
                paddle.nn.functional.poisson_nll_loss,
                input=input_x,
                label=label,
                log_input=log_input,
                full=full,
                epsilon=epsilon,
                reduction="unsupport reduction",
            )
        elif type == 'test_err_epsilon':
            self.assertRaises(
                ValueError,
                paddle.nn.functional.poisson_nll_loss,
                input=input_x,
                label=label,
                log_input=log_input,
                full=full,
                epsilon=-1,
                reduction="mean",
            )
        poisson_nll_loss = paddle.nn.PoissonNLLLoss(
            log_input=log_input, full=full, epsilon=epsilon, reduction=reduction
        )
        out2 = poisson_nll_loss(input_x, label)

        for r in [out1, out2]:
            np.allclose(out_ref, r.numpy(), rtol=1e-5)
        paddle.enable_static()


class TestPoissonNLLLossErrReductionCase(TestPoissonNLLLossDynamicCase):
    def test_api(self):
        self.test_dynamic_case(type="test_err_reduction")


class TestPoissonNLLLossErrEpsilonCase(TestPoissonNLLLossDynamicCase):
    def test_api(self):
        self.test_dynamic_case(type="test_err_epsilon")


class TestPoissonNLLLossDynamicFloat32Case(TestPoissonNLLLossDynamicCase):
    def test_api(self):
        self.test_dynamic_case(dtype="float32")


class TestPoissonNLLLossDynamicFloat64Case(TestPoissonNLLLossDynamicCase):
    def test_api(self):
        self.test_dynamic_case(dtype="float64")


class TestPoissonNLLLossStaticFloat32Case(TestPoissonNLLLossStaticCase):
    def test_api(self):
        self.test_static_case(dtype="float32")


class TestPoissonNLLLossStaticFloat64Case(TestPoissonNLLLossStaticCase):
    def test_api(self):
        self.test_static_case(dtype="float64")


class TestPoissonNLLLossDynamicNoLoginputCase(TestPoissonNLLLossDynamicCase):
    def test_api(self):
        self.test_dynamic_case(log_input=False)


class TestPoissonNLLLossStaticNoLoginputCase(TestPoissonNLLLossStaticCase):
    def test_api(self):
        self.test_static_case(log_input=False)


class TestPoissonNLLLossDynamicFulllossCase(TestPoissonNLLLossDynamicCase):
    def test_api(self):
        self.test_dynamic_case(full=True)


class TestPoissonNLLLossStaticFulllossCase(TestPoissonNLLLossStaticCase):
    def test_api(self):
        self.test_static_case(full=True)


class TestPoissonNLLLossDynamicSumReductionCase(TestPoissonNLLLossDynamicCase):
    def test_api(self):
        self.test_dynamic_case(reduction="sum")


class TestPoissonNLLLossStaticSumReductionCase(TestPoissonNLLLossStaticCase):
    def test_api(self):
        self.test_static_case(reduction="sum")


if __name__ == "__main__":
    unittest.main()
