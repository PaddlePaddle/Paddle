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

import paddle
from paddle import base


def smooth_l1_loss_forward(val, delta):
    abs_val = abs(val)
    if abs_val <= delta:
        return 0.5 * val * val
    else:
        return delta * (abs_val - 0.5 * delta)


def smooth_l1_loss_np(input, label, reduction='mean', delta=1.0):
    diff = input - label
    out = np.vectorize(smooth_l1_loss_forward)(diff, delta)
    if reduction == 'sum':
        return np.sum(out)
    elif reduction == 'mean':
        return np.mean(out)
    elif reduction == 'none':
        return out


class SmoothL1Loss(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)

    def test_smooth_l1_loss_mean(self):
        input_np = np.random.random([100, 200]).astype(np.float32)
        label_np = np.random.random([100, 200]).astype(np.float32)

        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )

        expected = smooth_l1_loss_np(input_np, label_np, reduction='mean')

        def test_static():
            prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(prog, startup_prog):
                input = paddle.static.data(
                    name='input', shape=[100, 200], dtype='float32'
                )
                label = paddle.static.data(
                    name='label', shape=[100, 200], dtype='float32'
                )
                smooth_l1_loss = paddle.nn.loss.SmoothL1Loss()
                ret = smooth_l1_loss(input, label)

                exe = paddle.static.Executor(place)
                (static_ret,) = exe.run(
                    feed={
                        'input': input_np,
                        'label': label_np,
                    },
                    fetch_list=[ret],
                )
                self.assertIsNotNone(static_ret)
                np.testing.assert_allclose(static_ret, expected, rtol=1e-05)

        with base.dygraph.guard():
            smooth_l1_loss = paddle.nn.loss.SmoothL1Loss()
            dy_ret = smooth_l1_loss(
                paddle.to_tensor(input_np),
                paddle.to_tensor(label_np),
            )
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)

        test_static()
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_smooth_l1_loss_sum(self):
        input_np = np.random.random([100, 200]).astype(np.float32)
        label_np = np.random.random([100, 200]).astype(np.float32)

        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        expected = smooth_l1_loss_np(input_np, label_np, reduction='sum')

        def test_static():
            prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(prog, startup_prog):
                input = paddle.static.data(
                    name='input', shape=[100, 200], dtype='float32'
                )
                label = paddle.static.data(
                    name='label', shape=[100, 200], dtype='float32'
                )
                smooth_l1_loss = paddle.nn.loss.SmoothL1Loss(reduction='sum')
                ret = smooth_l1_loss(input, label)

                exe = paddle.static.Executor(place)
                (static_ret,) = exe.run(
                    feed={
                        'input': input_np,
                        'label': label_np,
                    },
                    fetch_list=[ret],
                )
                self.assertIsNotNone(static_ret)
                np.testing.assert_allclose(static_ret, expected, rtol=1e-05)

        with base.dygraph.guard():
            smooth_l1_loss = paddle.nn.loss.SmoothL1Loss(reduction='sum')
            dy_ret = smooth_l1_loss(
                paddle.to_tensor(input_np),
                paddle.to_tensor(label_np),
            )
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)

        test_static()
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_smooth_l1_loss_none(self):
        input_np = np.random.random([100, 200]).astype(np.float32)
        label_np = np.random.random([100, 200]).astype(np.float32)

        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        expected = smooth_l1_loss_np(input_np, label_np, reduction='none')

        def test_static():
            prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(prog, startup_prog):
                input = paddle.static.data(
                    name='input', shape=[100, 200], dtype='float32'
                )
                label = paddle.static.data(
                    name='label', shape=[100, 200], dtype='float32'
                )
                smooth_l1_loss = paddle.nn.loss.SmoothL1Loss(reduction='none')
                ret = smooth_l1_loss(input, label)

                exe = paddle.static.Executor(place)
                (static_ret,) = exe.run(
                    feed={
                        'input': input_np,
                        'label': label_np,
                    },
                    fetch_list=[ret],
                )
                self.assertIsNotNone(static_ret)
                np.testing.assert_allclose(static_ret, expected, rtol=1e-05)

        with base.dygraph.guard():
            smooth_l1_loss = paddle.nn.loss.SmoothL1Loss(reduction='none')
            dy_ret = smooth_l1_loss(
                paddle.to_tensor(input_np),
                paddle.to_tensor(label_np),
            )
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)

        test_static()
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_smooth_l1_loss_delta(self):
        input_np = np.random.random([100, 200]).astype(np.float32)
        label_np = np.random.random([100, 200]).astype(np.float32)
        delta = np.random.rand()

        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        expected = smooth_l1_loss_np(input_np, label_np, delta=delta)

        def test_static():
            prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(prog, startup_prog):
                input = paddle.static.data(
                    name='input', shape=[100, 200], dtype='float32'
                )
                label = paddle.static.data(
                    name='label', shape=[100, 200], dtype='float32'
                )
                smooth_l1_loss = paddle.nn.loss.SmoothL1Loss(delta=delta)
                ret = smooth_l1_loss(input, label)

                exe = paddle.static.Executor(place)
                (static_ret,) = exe.run(
                    feed={
                        'input': input_np,
                        'label': label_np,
                    },
                    fetch_list=[ret],
                )
                self.assertIsNotNone(static_ret)
                np.testing.assert_allclose(static_ret, expected, rtol=1e-05)

        with base.dygraph.guard():
            smooth_l1_loss = paddle.nn.loss.SmoothL1Loss(delta=delta)
            dy_ret = smooth_l1_loss(
                paddle.to_tensor(input_np),
                paddle.to_tensor(label_np),
            )
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)

        test_static()
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
