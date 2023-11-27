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
import paddle.nn.functional as F
from paddle.base import core

np.random.seed(10)


def ref_gaussian_nll_loss(
    input, label, variance, full=False, eps=1e-6, reduction='none'
):
    if variance.shape != input.shape:
        if input.shape[:-1] == variance.shape:
            variance = np.expand_dims(variance, -1)
        elif (
            input.shape[:-1] == variance.shape[:-1] and variance.shape[-1] == 1
        ):
            pass
        else:
            raise ValueError("variance is of incorrect size")
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    if np.any(variance < 0):
        raise ValueError("var has negative entry/entries")

    variance = variance.copy()
    variance = np.clip(variance, a_min=eps, a_max=None)

    loss = 0.5 * (np.log(variance) + (input - label) ** 2 / variance)
    if full:
        loss += 0.5 * np.log(2 * np.pi)

    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return [np.sum(loss)]
    elif reduction == 'mean':
        return [np.mean(loss)]


class TestGaussianNLLLossAPI(unittest.TestCase):
    # test paddle.nn.functional.gaussian_nll_loss, paddle.nn.gaussian_nll_loss

    def setUp(self, type=None):
        self.shape = [10, 2]
        if type in ['float16', 'float64', 'int32', 'int64']:
            dtype = np.dtype(type)
            self.input_np = np.random.random(self.shape).astype(dtype)
            self.label_np = np.random.random(self.shape).astype(dtype)
            self.variance_np = np.ones(self.shape).astype(dtype)
        elif type == 'broadcast1':
            self.shape = [10, 2, 3]
            self.broadcast_shape = [10, 2]
            self.input_np = np.random.random(self.shape).astype(np.float32)
            self.label_np = np.random.random(self.shape).astype(np.float32)
            self.variance_np = np.ones(self.broadcast_shape).astype(np.float32)
        elif type == 'broadcast2':
            self.shape = [10, 2, 3]
            self.broadcast_shape = [10, 2, 1]
            self.input_np = np.random.random(self.shape).astype(np.float32)
            self.label_np = np.random.random(self.shape).astype(np.float32)
            self.variance_np = np.ones(self.broadcast_shape).astype(np.float32)
        else:
            dtype = np.dtype('float32')
            self.input_np = np.random.random(self.shape).astype(dtype)
            self.label_np = np.random.random(self.shape).astype(dtype)
            self.variance_np = np.ones(self.shape).astype(dtype)
        if type == 'test_err':
            self.variance_np = -np.ones(self.shape).astype(np.float32)

        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_dynamic_case(self, type=None, full=False, reduction='none'):
        self.setUp(type)
        paddle.disable_static(self.place)

        input_x = paddle.to_tensor(self.input_np)
        label = paddle.to_tensor(self.label_np)
        variance = paddle.to_tensor(self.variance_np)
        if type in ['test_err', 'int32', 'int64']:
            self.assertRaises(
                ValueError,
                paddle.nn.functional.gaussian_nll_loss,
                input=input_x,
                label=label,
                variance=variance,
            )
        else:
            out_ref = ref_gaussian_nll_loss(
                self.input_np,
                self.label_np,
                self.variance_np,
                full=full,
                reduction=reduction,
            )
            out1 = F.gaussian_nll_loss(
                input_x, label, variance, full=full, reduction=reduction
            )
            gaussian_nll_loss = paddle.nn.GaussianNLLLoss(
                full, reduction=reduction
            )
            out2 = gaussian_nll_loss(input_x, label, variance)

            for r in [out1, out2]:
                np.allclose(out_ref, r.numpy(), rtol=1e-5, atol=1e-5)
        paddle.enable_static()

    def test_static_case(self, type=None, full=False, reduction='none'):
        self.setUp(type)
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            if type in ['int32', 'int64', 'float64']:
                input_x = paddle.static.data('Input_x', self.shape, type)
                label = paddle.static.data('Label', self.shape, type)
                variance = paddle.static.data('Variance', self.shape, type)
            elif type in ['broadcast1', 'broadcast2']:
                input_x = paddle.static.data('Input_x', self.shape)
                label = paddle.static.data('Label', self.shape)
                variance = paddle.static.data('Variance', self.broadcast_shape)
            else:
                input_x = paddle.static.data('Input_x', self.shape, 'float32')
                label = paddle.static.data('Label', self.shape, 'float32')
                variance = paddle.static.data('Variance', self.shape, 'float32')
            out1 = F.gaussian_nll_loss(
                input_x, label, variance, full=full, reduction=reduction
            )
            gaussian_nll_loss = paddle.nn.GaussianNLLLoss(
                full, reduction=reduction
            )
            out2 = gaussian_nll_loss(input_x, label, variance)
            exe = paddle.static.Executor(self.place)
            if type not in ['test_err', 'int32', 'int64']:
                out_ref = ref_gaussian_nll_loss(
                    self.input_np,
                    self.label_np,
                    self.variance_np,
                    full=full,
                    reduction=reduction,
                )
                res = exe.run(
                    feed={
                        'Input_x': self.input_np,
                        'Label': self.label_np,
                        'Variance': self.variance_np,
                    },
                    fetch_list=[out1, out2],
                )
                for r in res:
                    np.allclose(out_ref, r, rtol=1e-5, atol=1e-5)
            else:
                try:
                    res = exe.run(
                        feed={
                            'Input_x': self.input_np,
                            'Label': self.label_np,
                            'Variance': self.variance_np,
                        },
                        fetch_list=[out1, out2],
                    )
                except ValueError:
                    pass

    def test_api(self):
        self.test_dynamic_case()
        self.test_static_case()

    def test_float64(self):
        self.test_dynamic_case('float64')
        self.test_static_case('float64')

    def test_broadcast(self):
        self.test_dynamic_case('broadcast1')
        self.test_static_case('broadcast1')

    def test_broadcast_with_same_dim(self):
        self.test_dynamic_case('broadcast2')
        self.test_static_case('broadcast2')

    def test_reduction(self):
        self.test_dynamic_case(full=True, reduction='mean')
        self.test_dynamic_case(full=True, reduction='sum')
        self.test_static_case(full=True, reduction='mean')

    def test_error(self):
        self.test_dynamic_case('test_err')
        self.test_static_case('test_err')

    def test_int(self):
        self.test_dynamic_case('int64')
        self.test_dynamic_case('int32')


if __name__ == "__main__":
    unittest.main()
