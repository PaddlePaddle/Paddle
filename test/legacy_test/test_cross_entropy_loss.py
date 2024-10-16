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

import sys
import unittest

import numpy as np

sys.path.append("../deprecated/legacy_test")
from test_softmax_op import stable_softmax
from test_softmax_with_cross_entropy_op import cross_entropy

import paddle
from paddle import base
from paddle.base import Program, program_guard


def label_smooth(label, C, epsilon, is_onehot=True):
    """
    Smooths the labels, commonly used in machine learning and deep learning to address label noise issues.

    Args:
        label: Labels, of type np.ndarray, with shape (batch_size, C)
        C: Number of classes
        epsilon: Smoothing factor, should be greater than or equal to 0 and less than or equal to 1
        is_onehot: Whether the labels are in one-hot encoding, default is True

    Returns:
        smooth_labels: Smoothed labels, of type np.ndarray, with the same shape as label
    """
    assert epsilon >= 0.0 and epsilon <= 1.0, "epsilon should be in [0.0, 1.0]"
    confidence = 1.0 - epsilon

    # one-hot label
    if label.shape[-1] == C and is_onehot is True:
        smooth_labels = label * confidence + epsilon / C

    # index label
    else:
        # Convert index labels to one-hot labels
        eye_matrix = np.eye(C)
        onehot_labels = eye_matrix[label]
        smooth_labels = onehot_labels * confidence + epsilon / C
    return smooth_labels.astype(np.float64)


def log_softmax(x, axis=-1):
    softmax_out = np.apply_along_axis(stable_softmax, axis, x)
    return np.log(softmax_out)


def cross_entropy_loss_1d(
    input, label, weight=None, reduction='mean', ignore_index=-100
):
    log_softmax_out = log_softmax(input)
    input_shape = log_softmax_out.shape
    N = input_shape[0]
    C = input_shape[1]
    out = np.zeros_like(label).astype(np.float64)
    total_weight = 0
    # 1. compute softmax cross_entropy (with weight)
    #    Note: only support hard labels.
    for i in range(N):
        cur_target = label[i]
        if cur_target == ignore_index:
            out[i] = 0
            continue
        cur_weight = weight[cur_target] if weight is not None else 1
        total_weight += cur_weight
        out[i] = -log_softmax_out[i][cur_target] * cur_weight

    # 2. deal with reduction
    if reduction == 'sum':
        return np.sum(out), np.array([total_weight]).astype('float64')
    elif reduction == 'mean':
        out = out.sum() / total_weight if total_weight != 0 else out.sum()
        return out, np.array([total_weight]).astype('float64')
    elif reduction == 'none':
        return out


def cross_entropy_loss_2d(
    input, label, weight=None, reduction='mean', ignore_index=-100
):
    log_softmax_out = log_softmax(input)
    input_shape = log_softmax_out.shape
    N = input_shape[0]
    H = input_shape[1]
    W = input_shape[2]

    out = np.zeros_like(label).astype(np.float64)
    total_weight = 0
    for i in range(N):
        for h in range(H):
            for w in range(W):
                cur_target = label[i][h][w]
                if cur_target == ignore_index:
                    out[i][h][w] = 0
                    continue
                cur_weight = weight[cur_target] if weight is not None else 1
                total_weight += cur_weight
                out[i][h][w] = (
                    -log_softmax_out[i][h][w][cur_target] * cur_weight
                )
    if reduction == 'sum':
        return np.sum(out), np.array([total_weight]).astype('float64')
    elif reduction == 'mean':
        out = out.sum() / total_weight if total_weight != 0 else out.sum()
        return out, np.array([total_weight]).astype('float64')
    elif reduction == 'none':
        return out


def cross_entropy_soft(
    softmax, label, axis, N, weight=None, reduction='mean', ignore_index=-100
):
    # 1.loss
    loss = cross_entropy(
        softmax, label, True, axis, ignore_index  # soft_label,
    )

    if weight is None and reduction == 'none':
        return loss

    # 2.weight
    weighted_loss = loss
    total_weight = N  # for weight is None
    if weight is not None:
        weighted_loss = np.zeros_like(loss).astype(np.float64)
        total_weight = 0
        for i in range(N):
            cur_soft_label = label[i]
            cur_weight = np.dot(weight, cur_soft_label)
            total_weight += cur_weight
            weighted_loss[i] = loss[i] * cur_weight

    # 3.reduce
    if reduction == 'none':
        return weighted_loss

    elif reduction == 'mean':
        weighted_loss_sum = np.sum(weighted_loss)
        weighted_loss_mean = weighted_loss_sum / total_weight
        return weighted_loss_mean

    else:
        weighted_loss_sum = np.sum(weighted_loss)
        return weighted_loss_sum


def cross_entropy_soft_2d(
    softmax,
    label,
    axis,
    N,
    H,
    W,
    weight=None,
    reduction='mean',
    ignore_index=-100,
):
    # 1.loss
    loss = cross_entropy(
        softmax, label, True, axis, ignore_index  # soft_label,
    )

    if weight is None and reduction == 'none':
        return loss

    # 2.weight
    weighted_loss = loss
    total_weight = N  # for weight is None
    if weight is not None:
        weighted_loss = np.zeros_like(loss).astype(np.float64)
        total_weight = 0
        for i in range(N):
            for h in range(H):
                for w in range(W):
                    cur_soft_label = label[i][h][w]
                    cur_weight = np.dot(weight, cur_soft_label)
                    total_weight += cur_weight
                    weighted_loss[i][h][w] = loss[i][h][w] * cur_weight

    # 3.reduce
    if reduction == 'none':
        return weighted_loss

    elif reduction == 'mean':
        weighted_loss_sum = np.sum(weighted_loss)
        weighted_loss_mean = weighted_loss_sum / total_weight
        return weighted_loss_mean

    else:
        weighted_loss_sum = np.sum(weighted_loss)
        return weighted_loss_sum


class CrossEntropyLoss(unittest.TestCase):
    def setUp(self):
        self.dtype = (
            'float32' if base.core.is_compiled_with_rocm() else 'float64'
        )

    # test for deprecated softmax_with_cross_entropy
    def test_softmax_with_cross_entropy(self):
        self.numeric_stable_mode = False
        self.soft_label = True
        self.dtype = (
            'float32' if base.core.is_compiled_with_rocm() else 'float64'
        )
        self.axis = -1
        self.ignore_index = -100  # should not be changed
        self.N = 4
        self.C = 3
        self.shape = [self.N, self.C]
        self.use_softmax = True
        self.reduction = 'none'
        self.weight = None
        self.logits = getattr(
            self,
            "logits",
            np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype),
        )
        softmax = np.apply_along_axis(stable_softmax, self.axis, self.logits)

        self.labels = np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype)
        self.labels /= np.sum(self.labels, axis=self.axis, keepdims=True)

        expected = cross_entropy_soft(
            softmax,
            self.labels,
            self.axis,
            self.N,
            weight=self.weight,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )

        paddle.set_device("cpu")

        paddle.disable_static()
        paddle_loss_swce = paddle.nn.functional.softmax_with_cross_entropy(
            paddle.to_tensor(self.logits),
            paddle.to_tensor(self.labels),
            soft_label=True,
            axis=self.axis,
        )

        paddle_loss_ce = paddle.nn.functional.cross_entropy(
            paddle.to_tensor(self.logits),
            paddle.to_tensor(self.labels),
            soft_label=True,
            axis=self.axis,
            weight=(
                paddle.to_tensor(self.weight)
                if self.weight is not None
                else None
            ),
            reduction=self.reduction,
        )

        np.testing.assert_allclose(
            paddle_loss_swce.numpy(), expected, rtol=1e-05
        )
        np.testing.assert_allclose(paddle_loss_ce.numpy(), expected, rtol=1e-05)

    # soft_label test start
    # soft_label test 1

    def test_cross_entropy_loss_soft_1d(self):
        self.numeric_stable_mode = False
        self.soft_label = True
        self.dtype = (
            'float32' if base.core.is_compiled_with_rocm() else 'float64'
        )
        self.axis = -1
        self.ignore_index = -100  # should not be changed
        self.N = 4
        self.C = 3
        self.shape = [self.N, self.C]
        self.use_softmax = True
        self.reduction = 'none'
        self.weight = None
        self.logits = getattr(
            self,
            "logits",
            np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype),
        )
        softmax = np.apply_along_axis(stable_softmax, self.axis, self.logits)

        self.labels = np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype)
        self.labels /= np.sum(self.labels, axis=self.axis, keepdims=True)

        expected = cross_entropy_soft(
            softmax,
            self.labels,
            self.axis,
            self.N,
            weight=self.weight,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )

        paddle.set_device("cpu")

        # 2. dygraph
        paddle.disable_static()
        paddle_loss_none_weight = paddle.nn.functional.cross_entropy(
            paddle.to_tensor(self.logits),
            paddle.to_tensor(self.labels),
            soft_label=True,
            axis=self.axis,
            weight=(
                paddle.to_tensor(self.weight)
                if self.weight is not None
                else None
            ),
            reduction=self.reduction,
        )
        dy_ret_value = paddle_loss_none_weight.numpy()

        # 3. static
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[self.N, self.C], dtype=self.dtype
            )
            label = paddle.static.data(
                name='label', shape=[self.N, self.C], dtype=self.dtype
            )

            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction=self.reduction, soft_label=True
            )
            ret = cross_entropy_loss(input, label)

            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': self.logits,
                    'label': self.labels,
                },
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)
        paddle.disable_static()

        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    # soft_label test 2

    def test_cross_entropy_loss_soft_1d_weight(self):
        self.numeric_stable_mode = False
        self.soft_label = True
        self.dtype = (
            'float32' if base.core.is_compiled_with_rocm() else 'float64'
        )
        self.axis = -1
        self.ignore_index = -100  # should not be changed
        self.N = 4
        self.C = 3
        self.shape = [self.N, self.C]
        self.use_softmax = True
        self.reduction = 'none'
        self.weight = np.random.uniform(0.1, 1.0, self.C).astype(self.dtype)
        self.logits = getattr(
            self,
            "logits",
            np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype),
        )
        softmax = np.apply_along_axis(stable_softmax, self.axis, self.logits)

        if self.soft_label:
            self.labels = np.random.uniform(0.1, 1.0, self.shape).astype(
                self.dtype
            )
            self.labels /= np.sum(self.labels, axis=self.axis, keepdims=True)
        else:
            axis_dim = self.shape[self.axis]
            self.shape[self.axis] = 1
            self.labels = np.random.randint(
                0, axis_dim, self.shape, dtype="int64"
            )

        # 1. numpy
        expected = cross_entropy_soft(
            softmax,
            self.labels,
            self.axis,
            self.N,
            weight=self.weight,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )

        paddle.set_device("cpu")

        # 2. dygraph
        paddle.disable_static()
        paddle_loss_none_weight = paddle.nn.functional.cross_entropy(
            paddle.to_tensor(self.logits),
            paddle.to_tensor(self.labels),
            soft_label=True,
            axis=self.axis,
            weight=paddle.to_tensor(self.weight),
            reduction=self.reduction,
        )
        dy_ret_value = paddle_loss_none_weight.numpy()

        # 3.static
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[self.N, self.C], dtype=self.dtype
            )
            label = paddle.static.data(
                name='label', shape=[self.N, self.C], dtype=self.dtype
            )
            weight = paddle.static.data(
                name='weight', shape=[self.C], dtype=self.dtype
            )

            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight, reduction=self.reduction, soft_label=True
            )
            ret = cross_entropy_loss(input, label)

            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': self.logits,
                    'label': self.labels,
                    "weight": self.weight,
                },
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)
        paddle.disable_static()

        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    # soft_label test 3

    def test_cross_entropy_loss_soft_1d_mean(self):
        self.numeric_stable_mode = False
        self.soft_label = True
        self.dtype = (
            'float32' if base.core.is_compiled_with_rocm() else 'float64'
        )
        self.axis = -1
        self.ignore_index = -100  # should not be changed
        self.N = 4
        self.C = 3
        self.shape = [self.N, self.C]
        self.use_softmax = True
        self.reduction = 'mean'
        self.weight = None
        self.logits = getattr(
            self,
            "logits",
            np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype),
        )
        softmax = np.apply_along_axis(stable_softmax, self.axis, self.logits)

        self.labels = np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype)
        self.labels /= np.sum(self.labels, axis=self.axis, keepdims=True)

        # 1. numpy
        expected = cross_entropy_soft(
            softmax,
            self.labels,
            self.axis,
            self.N,
            weight=self.weight,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )

        paddle.set_device("cpu")

        # 2 dygraph
        paddle.disable_static()
        paddle_loss_mean = paddle.nn.functional.cross_entropy(
            paddle.to_tensor(self.logits),
            paddle.to_tensor(self.labels),
            soft_label=True,
            axis=self.axis,
            weight=self.weight,
            reduction=self.reduction,
        )
        dy_ret_value = paddle_loss_mean.numpy()

        # 3. static
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[self.N, self.C], dtype=self.dtype
            )
            label = paddle.static.data(
                name='label', shape=[self.N, self.C], dtype=self.dtype
            )

            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction=self.reduction, soft_label=True
            )
            ret = cross_entropy_loss(input, label)

            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={'input': self.logits, 'label': self.labels},
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)
        paddle.disable_static()

        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    # soft_label test 4

    def test_cross_entropy_loss_soft_1d_weight_mean(self):
        self.numeric_stable_mode = False
        self.soft_label = True
        self.dtype = (
            'float32' if base.core.is_compiled_with_rocm() else 'float64'
        )
        self.axis = -1
        self.ignore_index = -100  # should not be changed
        self.N = 4
        self.C = 3
        self.shape = [self.N, self.C]
        self.use_softmax = True
        self.reduction = 'mean'
        self.weight = np.random.uniform(0.1, 1.0, self.C).astype(self.dtype)
        self.logits = getattr(
            self,
            "logits",
            np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype),
        )
        softmax = np.apply_along_axis(stable_softmax, self.axis, self.logits)

        self.labels = np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype)
        self.labels /= np.sum(self.labels, axis=self.axis, keepdims=True)

        # 1. numpy
        expected = cross_entropy_soft(
            softmax,
            self.labels,
            self.axis,
            self.N,
            weight=self.weight,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )

        paddle.set_device("cpu")
        paddle.disable_static()

        # 2. dygraph
        paddle_loss_none_weight = paddle.nn.functional.cross_entropy(
            paddle.to_tensor(self.logits),
            paddle.to_tensor(self.labels),
            soft_label=True,
            axis=self.axis,
            weight=paddle.to_tensor(self.weight),
            reduction=self.reduction,
        )
        dy_ret_value = paddle_loss_none_weight.numpy()

        # 3. static
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[self.N, self.C], dtype=self.dtype
            )
            label = paddle.static.data(
                name='label', shape=[self.N, self.C], dtype=self.dtype
            )
            weight = paddle.static.data(
                name='weight', shape=[self.C], dtype=self.dtype
            )

            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight, reduction=self.reduction, soft_label=True
            )
            ret = cross_entropy_loss(input, label)
            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': self.logits,
                    'label': self.labels,
                    "weight": self.weight,
                },
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)
        paddle.disable_static()

        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    # soft_label test 5

    def test_cross_entropy_loss_soft_2d(self):
        def inner_cross_entropy_loss_soft_2d(soft_label):
            self.numeric_stable_mode = False
            self.soft_label = soft_label
            self.dtype = (
                'float32' if base.core.is_compiled_with_rocm() else 'float64'
            )
            self.axis = -1
            self.ignore_index = -100  # should not be changed
            self.N = 3
            self.H = 2
            self.W = 2
            self.C = 5
            self.shape = [self.N, self.H, self.W, self.C]
            self.use_softmax = True
            self.reduction = 'none'
            self.weight = None
            self.logits = getattr(
                self,
                "logits",
                np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype),
            )
            softmax = np.apply_along_axis(
                stable_softmax, self.axis, self.logits
            )

            self.labels = np.random.uniform(0.1, 1.0, self.shape).astype(
                self.dtype
            )
            self.labels /= np.sum(self.labels, axis=self.axis, keepdims=True)

            # 1. numpy
            expected = cross_entropy_soft_2d(
                softmax,
                self.labels,
                self.axis,
                self.N,
                self.H,
                self.W,
                weight=self.weight,
                reduction=self.reduction,
                ignore_index=self.ignore_index,
            )

            paddle.set_device("cpu")
            paddle.disable_static()

            # 2. dygraph
            paddle_loss_none_weight = paddle.nn.functional.cross_entropy(
                paddle.to_tensor(self.logits),
                paddle.to_tensor(self.labels),
                soft_label=True,
                axis=self.axis,
                weight=(
                    paddle.to_tensor(self.weight)
                    if self.weight is not None
                    else None
                ),
                reduction=self.reduction,
            )
            dy_ret_value = paddle_loss_none_weight.numpy()

            # 3. static
            paddle.enable_static()
            prog = base.Program()
            startup_prog = base.Program()
            place = (
                base.CUDAPlace(0)
                if base.core.is_compiled_with_cuda()
                else base.CPUPlace()
            )
            with base.program_guard(prog, startup_prog):
                input = paddle.static.data(
                    name='input',
                    shape=[self.N, self.H, self.W, self.C],
                    dtype=self.dtype,
                )
                label = paddle.static.data(
                    name='label',
                    shape=[self.N, self.H, self.W, self.C],
                    dtype=self.dtype,
                )

                cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                    reduction=self.reduction, soft_label=True
                )
                ret = cross_entropy_loss(input, label)
                exe = base.Executor(place)
                static_ret = exe.run(
                    prog,
                    feed={
                        'input': self.logits,
                        'label': self.labels,
                    },
                    fetch_list=[ret],
                )
                self.assertIsNotNone(static_ret)
            paddle.disable_static()

            np.testing.assert_allclose(static_ret[0], dy_ret_value, rtol=1e-05)
            np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
            np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

        inner_cross_entropy_loss_soft_2d(True)
        inner_cross_entropy_loss_soft_2d(False)

    # soft_label test 6

    def test_cross_entropy_loss_soft_2d_weight_mean(self):
        self.numeric_stable_mode = False
        self.soft_label = True
        self.dtype = (
            'float32' if base.core.is_compiled_with_rocm() else 'float64'
        )
        self.axis = -1
        self.ignore_index = -100  # should not be changed
        self.N = 3
        self.H = 2
        self.W = 2
        self.C = 5
        self.shape = [self.N, self.H, self.W, self.C]
        self.use_softmax = True
        self.reduction = 'mean'
        self.weight = np.random.uniform(0.1, 1.0, self.C).astype(self.dtype)
        self.logits = getattr(
            self,
            "logits",
            np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype),
        )
        softmax = np.apply_along_axis(stable_softmax, self.axis, self.logits)

        self.labels = np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype)
        self.labels /= np.sum(self.labels, axis=self.axis, keepdims=True)

        # 1. numpy
        expected = cross_entropy_soft_2d(
            softmax,
            self.labels,
            self.axis,
            self.N,
            self.H,
            self.W,
            weight=self.weight,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )

        paddle.set_device("cpu")
        paddle.disable_static()

        # 2. dygraph
        paddle_loss_none_weight = paddle.nn.functional.cross_entropy(
            paddle.to_tensor(self.logits),
            paddle.to_tensor(self.labels),
            soft_label=True,
            axis=self.axis,
            weight=paddle.to_tensor(self.weight),
            reduction=self.reduction,
        )
        dy_ret_value = paddle_loss_none_weight.numpy()

        # 3. static
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input',
                shape=[self.N, self.H, self.W, self.C],
                dtype=self.dtype,
            )
            label = paddle.static.data(
                name='label',
                shape=[self.N, self.H, self.W, self.C],
                dtype=self.dtype,
            )
            weight = paddle.static.data(
                name='weight', shape=[self.C], dtype=self.dtype
            )

            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight, reduction=self.reduction, soft_label=True
            )
            ret = cross_entropy_loss(input, label)
            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': self.logits,
                    'label': self.labels,
                    "weight": self.weight,
                },
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)
        paddle.disable_static()

        np.testing.assert_allclose(static_ret[0], dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    # soft_label test end

    # label_smoothing test 1

    def test_cross_entropy_loss_onehot_label_smoothing_1d(self):
        self.numeric_stable_mode = False
        self.soft_label = True
        self.label_smoothing = np.random.uniform(0.1, 1.0)
        self.dtype = (
            'float32' if base.core.is_compiled_with_rocm() else 'float64'
        )
        self.axis = -1
        self.ignore_index = -100  # should not be changed
        self.N = 4
        self.C = 3
        self.shape = [self.N, self.C]
        self.use_softmax = True
        self.reduction = 'none'
        self.weight = None
        self.logits = getattr(
            self,
            "logits",
            np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype),
        )
        softmax = np.apply_along_axis(stable_softmax, self.axis, self.logits)

        self.labels = np.zeros(self.shape, dtype=self.dtype)
        indices = np.random.randint(0, self.C, self.N)
        self.labels[np.arange(self.N), indices] = 1.0
        self.soft_labels = label_smooth(
            self.labels, self.C, epsilon=self.label_smoothing
        )

        expected = cross_entropy_soft(
            softmax,
            self.soft_labels,
            self.axis,
            self.N,
            weight=self.weight,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )

        paddle.set_device("cpu")

        # 2. dygraph
        paddle.disable_static()
        paddle_loss_none_weight = paddle.nn.functional.cross_entropy(
            paddle.to_tensor(self.logits),
            paddle.to_tensor(self.labels),
            soft_label=True,
            label_smoothing=self.label_smoothing,
            axis=self.axis,
            weight=(
                paddle.to_tensor(self.weight)
                if self.weight is not None
                else None
            ),
            reduction=self.reduction,
        )
        dy_ret_value = paddle_loss_none_weight.numpy()

        # 3. static
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[self.N, self.C], dtype=self.dtype
            )
            label = paddle.static.data(
                name='label', shape=[self.N, self.C], dtype=self.dtype
            )

            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction=self.reduction,
                soft_label=True,
                label_smoothing=self.label_smoothing,
            )
            ret = cross_entropy_loss(input, label)

            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': self.logits,
                    'label': self.labels,
                },
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)
        paddle.disable_static()

        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)
        paddle.enable_static()

    # label_smoothing test 2

    def test_cross_entropy_loss_onehot_label_smoothing_1d_weight_mean(self):
        self.numeric_stable_mode = False
        self.soft_label = True
        self.label_smoothing = np.random.uniform(0.1, 1.0)
        self.dtype = (
            'float32' if base.core.is_compiled_with_rocm() else 'float64'
        )
        self.axis = -1
        self.ignore_index = -100  # should not be changed
        self.N = 4
        self.C = 3
        self.shape = [self.N, self.C]
        self.use_softmax = True
        self.reduction = 'mean'
        self.weight = np.random.uniform(0.1, 1.0, self.C).astype(self.dtype)
        self.logits = getattr(
            self,
            "logits",
            np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype),
        )
        softmax = np.apply_along_axis(stable_softmax, self.axis, self.logits)

        self.labels = np.zeros(self.shape, dtype=self.dtype)
        indices = np.random.randint(0, self.C, self.N)
        self.labels[np.arange(self.N), indices] = 1.0
        self.soft_labels = label_smooth(
            self.labels, self.C, epsilon=self.label_smoothing
        )

        # 1. numpy
        expected = cross_entropy_soft(
            softmax,
            self.soft_labels,
            self.axis,
            self.N,
            weight=self.weight,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )

        paddle.set_device("cpu")
        paddle.disable_static()

        # 2. dygraph
        paddle_loss_none_weight = paddle.nn.functional.cross_entropy(
            paddle.to_tensor(self.logits),
            paddle.to_tensor(self.labels),
            soft_label=True,
            label_smoothing=self.label_smoothing,
            axis=self.axis,
            weight=paddle.to_tensor(self.weight),
            reduction=self.reduction,
        )
        dy_ret_value = paddle_loss_none_weight.numpy()

        # 3. static
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[self.N, self.C], dtype=self.dtype
            )
            label = paddle.static.data(
                name='label', shape=[self.N, self.C], dtype=self.dtype
            )
            weight = paddle.static.data(
                name='weight', shape=[self.C], dtype=self.dtype
            )

            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight,
                reduction=self.reduction,
                soft_label=True,
                label_smoothing=self.label_smoothing,
            )
            ret = cross_entropy_loss(input, label)
            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': self.logits,
                    'label': self.labels,
                    "weight": self.weight,
                },
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)
        paddle.disable_static()

        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    # label_smoothing test 3

    def test_cross_entropy_loss_onehot_label_smoothing_2d(self):
        self.numeric_stable_mode = False
        self.soft_label = True
        self.label_smoothing = np.random.uniform(0.1, 1.0)
        self.dtype = (
            'float32' if base.core.is_compiled_with_rocm() else 'float64'
        )
        self.axis = -1
        self.ignore_index = -100  # should not be changed
        self.N = 3
        self.H = 2
        self.W = 2
        self.C = 5
        self.shape = [self.N, self.H, self.W, self.C]
        self.use_softmax = True
        self.reduction = 'none'
        self.weight = None
        self.logits = getattr(
            self,
            "logits",
            np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype),
        )
        softmax = np.apply_along_axis(stable_softmax, self.axis, self.logits)

        self.labels = np.zeros(self.shape, dtype=self.dtype)
        indices = np.random.randint(0, self.C, self.shape[:-1])
        for i in range(self.N):
            self.labels[i, np.arange(self.H), np.arange(self.W), indices[i]] = (
                1.0
            )
        self.soft_labels = label_smooth(
            self.labels, self.C, epsilon=self.label_smoothing
        )

        # 1. numpy
        expected = cross_entropy_soft_2d(
            softmax,
            self.soft_labels,
            self.axis,
            self.N,
            self.H,
            self.W,
            weight=self.weight,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )

        paddle.set_device("cpu")
        paddle.disable_static()

        # 2. dygraph
        paddle_loss_none_weight = paddle.nn.functional.cross_entropy(
            paddle.to_tensor(self.logits),
            paddle.to_tensor(self.labels),
            soft_label=True,
            label_smoothing=self.label_smoothing,
            axis=self.axis,
            weight=(
                paddle.to_tensor(self.weight)
                if self.weight is not None
                else None
            ),
            reduction=self.reduction,
        )
        dy_ret_value = paddle_loss_none_weight.numpy()

        # 3. static
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input',
                shape=[self.N, self.H, self.W, self.C],
                dtype=self.dtype,
            )
            label = paddle.static.data(
                name='label',
                shape=[self.N, self.H, self.W, self.C],
                dtype=self.dtype,
            )

            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction=self.reduction,
                soft_label=True,
                label_smoothing=self.label_smoothing,
            )
            ret = cross_entropy_loss(input, label)
            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': self.logits,
                    'label': self.labels,
                },
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)
        paddle.disable_static()

        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    # label_smoothing test 4

    def test_cross_entropy_loss_onehot_label_smoothing_2d_weight_mean(self):
        self.numeric_stable_mode = False
        self.soft_label = True
        self.label_smoothing = np.random.uniform(0.1, 1.0)
        self.dtype = (
            'float32' if base.core.is_compiled_with_rocm() else 'float64'
        )
        self.axis = -1
        self.ignore_index = -100  # should not be changed
        self.N = 3
        self.H = 2
        self.W = 2
        self.C = 5
        self.shape = [self.N, self.H, self.W, self.C]
        self.use_softmax = True
        self.reduction = 'mean'
        self.weight = np.random.uniform(0.1, 1.0, self.C).astype(self.dtype)
        self.logits = getattr(
            self,
            "logits",
            np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype),
        )
        softmax = np.apply_along_axis(stable_softmax, self.axis, self.logits)

        self.labels = np.zeros(self.shape, dtype=self.dtype)
        indices = np.random.randint(0, self.C, self.shape[:-1])
        for i in range(self.N):
            self.labels[i, np.arange(self.H), np.arange(self.W), indices[i]] = (
                1.0
            )
        self.soft_labels = label_smooth(
            self.labels, self.C, epsilon=self.label_smoothing
        )

        # 1. numpy
        expected = cross_entropy_soft_2d(
            softmax,
            self.soft_labels,
            self.axis,
            self.N,
            self.H,
            self.W,
            weight=self.weight,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )

        paddle.set_device("cpu")
        paddle.disable_static()

        # 2. dygraph
        paddle_loss_none_weight = paddle.nn.functional.cross_entropy(
            paddle.to_tensor(self.logits),
            paddle.to_tensor(self.labels),
            soft_label=True,
            label_smoothing=self.label_smoothing,
            axis=self.axis,
            weight=paddle.to_tensor(self.weight),
            reduction=self.reduction,
        )
        dy_ret_value = paddle_loss_none_weight.numpy()

        # 3. static
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input',
                shape=[self.N, self.H, self.W, self.C],
                dtype=self.dtype,
            )
            label = paddle.static.data(
                name='label',
                shape=[self.N, self.H, self.W, self.C],
                dtype=self.dtype,
            )
            weight = paddle.static.data(
                name='weight', shape=[self.C], dtype=self.dtype
            )

            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight,
                reduction=self.reduction,
                soft_label=True,
                label_smoothing=self.label_smoothing,
            )
            ret = cross_entropy_loss(input, label)
            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': self.logits,
                    'label': self.labels,
                    "weight": self.weight,
                },
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)
        paddle.disable_static()

        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    # label_smoothing test 5

    def test_cross_entropy_loss_integer_label_smoothing_1d(self):
        self.numeric_stable_mode = False
        self.soft_label = True
        self.label_smoothing = np.random.uniform(0.1, 1.0)
        self.input_dtype = (
            'float32' if base.core.is_compiled_with_rocm() else 'float64'
        )
        self.label_dtype = 'int64'
        self.axis = -1
        self.ignore_index = -100  # should not be changed
        self.N = 4
        self.C = 3
        self.shape = [self.N, self.C]
        self.use_softmax = True
        self.reduction = 'none'
        self.weight = None
        self.logits = getattr(
            self,
            "logits",
            np.random.uniform(0.1, 1.0, self.shape).astype(self.input_dtype),
        )
        softmax = np.apply_along_axis(stable_softmax, self.axis, self.logits)
        self.labels = np.random.randint(0, self.C, size=self.N, dtype=np.int64)
        self.soft_labels = label_smooth(
            self.labels,
            self.C,
            epsilon=self.label_smoothing,
            is_onehot=False,
        )

        expected = cross_entropy_soft(
            softmax,
            self.soft_labels,
            self.axis,
            self.N,
            weight=self.weight,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )

        paddle.set_device("cpu")

        # 2. dygraph
        paddle.disable_static()
        paddle_loss_none_weight = paddle.nn.functional.cross_entropy(
            paddle.to_tensor(self.logits),
            paddle.to_tensor(self.labels),
            soft_label=self.soft_label,
            label_smoothing=self.label_smoothing,
            axis=self.axis,
            weight=(
                paddle.to_tensor(self.weight)
                if self.weight is not None
                else None
            ),
            reduction=self.reduction,
        )
        dy_ret_value = paddle_loss_none_weight.numpy()

        # 3. static
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[self.N, self.C], dtype=self.input_dtype
            )
            label = paddle.static.data(
                name='label', shape=[self.N], dtype=self.label_dtype
            )

            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction=self.reduction,
                soft_label=True,
                label_smoothing=self.label_smoothing,
            )
            ret = cross_entropy_loss(input, label)

            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': self.logits,
                    'label': self.labels,
                },
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)
        paddle.disable_static()

        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    # label_smoothing test 6

    def test_cross_entropy_loss_integer_label_smoothing_1d_weight_mean(self):
        self.numeric_stable_mode = False
        self.soft_label = True
        self.label_smoothing = np.random.uniform(0.1, 1.0)
        self.input_dtype = (
            'float32' if base.core.is_compiled_with_rocm() else 'float64'
        )
        self.label_dtype = 'int64'
        self.axis = -1
        self.ignore_index = -100  # should not be changed
        self.N = 4
        self.C = 3
        self.shape = [self.N, self.C]
        self.use_softmax = True
        self.reduction = 'mean'
        self.weight = np.random.uniform(0.1, 1.0, self.C).astype(self.dtype)
        self.logits = getattr(
            self,
            "logits",
            np.random.uniform(0.1, 1.0, self.shape).astype(self.input_dtype),
        )
        softmax = np.apply_along_axis(stable_softmax, self.axis, self.logits)
        self.labels = np.random.randint(0, self.C, size=self.N, dtype=np.int64)
        self.soft_labels = label_smooth(
            self.labels,
            self.C,
            epsilon=self.label_smoothing,
            is_onehot=False,
        )

        expected = cross_entropy_soft(
            softmax,
            self.soft_labels,
            self.axis,
            self.N,
            weight=self.weight,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )

        paddle.set_device("cpu")

        # 2. dygraph
        paddle.disable_static()
        paddle_loss_none_weight = paddle.nn.functional.cross_entropy(
            paddle.to_tensor(self.logits),
            paddle.to_tensor(self.labels),
            soft_label=self.soft_label,
            label_smoothing=self.label_smoothing,
            axis=self.axis,
            weight=(
                paddle.to_tensor(self.weight)
                if self.weight is not None
                else None
            ),
            reduction=self.reduction,
        )
        dy_ret_value = paddle_loss_none_weight.numpy()

        # 3. static
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[self.N, self.C], dtype=self.input_dtype
            )
            label = paddle.static.data(
                name='label', shape=[self.N], dtype=self.label_dtype
            )
            weight = paddle.static.data(
                name='weight', shape=[self.C], dtype=self.dtype
            )

            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight,
                reduction=self.reduction,
                soft_label=True,
                label_smoothing=self.label_smoothing,
            )
            ret = cross_entropy_loss(input, label)

            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': self.logits,
                    'label': self.labels,
                    'weight': self.weight,
                },
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)
        paddle.disable_static()

        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    # label_smoothing test 7

    def test_cross_entropy_loss_integer_label_smoothing_2d(self):
        self.numeric_stable_mode = False
        self.soft_label = True
        self.label_smoothing = np.random.uniform(0.1, 1.0)
        self.input_dtype = (
            'float32' if base.core.is_compiled_with_rocm() else 'float64'
        )
        self.label_dtype = 'int64'
        self.axis = -1
        self.ignore_index = -100  # should not be changed
        self.N = 3
        self.H = 2
        self.W = 2
        self.C = 5
        self.shape = [self.N, self.H, self.W, self.C]
        self.use_softmax = True
        self.reduction = 'none'
        self.weight = None
        self.logits = getattr(
            self,
            "logits",
            np.random.uniform(0.1, 1.0, self.shape).astype(self.input_dtype),
        )
        softmax = np.apply_along_axis(stable_softmax, self.axis, self.logits)

        self.labels = np.random.randint(0, self.C, self.shape[:-1]).astype(
            self.label_dtype
        )
        self.soft_labels = label_smooth(
            self.labels, self.C, epsilon=self.label_smoothing, is_onehot=False
        )

        # 1. numpy
        expected = cross_entropy_soft_2d(
            softmax,
            self.soft_labels,
            self.axis,
            self.N,
            self.H,
            self.W,
            weight=self.weight,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )

        paddle.set_device("cpu")
        paddle.disable_static()

        # 2. dygraph
        paddle_loss_none_weight = paddle.nn.functional.cross_entropy(
            paddle.to_tensor(self.logits),
            paddle.to_tensor(self.labels),
            soft_label=True,
            label_smoothing=self.label_smoothing,
            axis=self.axis,
            weight=(
                paddle.to_tensor(self.weight)
                if self.weight is not None
                else None
            ),
            reduction=self.reduction,
        )
        dy_ret_value = paddle_loss_none_weight.numpy()

        # 3. static
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input',
                shape=[self.N, self.H, self.W, self.C],
                dtype=self.input_dtype,
            )
            label = paddle.static.data(
                name='label',
                shape=[self.N, self.H, self.W],
                dtype=self.label_dtype,
            )

            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction=self.reduction,
                soft_label=True,
                label_smoothing=self.label_smoothing,
            )
            ret = cross_entropy_loss(input, label)
            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': self.logits,
                    'label': self.labels,
                },
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)
        paddle.disable_static()

        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    # label_smoothing test 8

    def test_cross_entropy_loss_integer_label_smoothing_2d_weight_mean(self):
        self.numeric_stable_mode = False
        self.soft_label = True
        self.label_smoothing = np.random.uniform(0.1, 1.0)
        self.input_dtype = (
            'float32' if base.core.is_compiled_with_rocm() else 'float64'
        )
        self.label_dtype = 'int64'
        self.axis = -1
        self.ignore_index = -100  # should not be changed
        self.N = 3
        self.H = 2
        self.W = 2
        self.C = 5
        self.shape = [self.N, self.H, self.W, self.C]
        self.use_softmax = True
        self.reduction = 'mean'
        self.weight = np.random.uniform(0.1, 1.0, self.C).astype(self.dtype)
        self.logits = getattr(
            self,
            "logits",
            np.random.uniform(0.1, 1.0, self.shape).astype(self.input_dtype),
        )
        softmax = np.apply_along_axis(stable_softmax, self.axis, self.logits)

        self.labels = np.random.randint(0, self.C, self.shape[:-1]).astype(
            self.label_dtype
        )
        self.soft_labels = label_smooth(
            self.labels, self.C, epsilon=self.label_smoothing, is_onehot=False
        )

        # 1. numpy
        expected = cross_entropy_soft_2d(
            softmax,
            self.soft_labels,
            self.axis,
            self.N,
            self.H,
            self.W,
            weight=self.weight,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )

        paddle.set_device("cpu")
        paddle.disable_static()

        # 2. dygraph
        paddle_loss_none_weight = paddle.nn.functional.cross_entropy(
            paddle.to_tensor(self.logits),
            paddle.to_tensor(self.labels),
            soft_label=True,
            label_smoothing=self.label_smoothing,
            axis=self.axis,
            weight=(
                paddle.to_tensor(self.weight)
                if self.weight is not None
                else None
            ),
            reduction=self.reduction,
        )
        dy_ret_value = paddle_loss_none_weight.numpy()

        # 3. static
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input',
                shape=[self.N, self.H, self.W, self.C],
                dtype=self.input_dtype,
            )
            label = paddle.static.data(
                name='label',
                shape=[self.N, self.H, self.W],
                dtype=self.label_dtype,
            )
            weight = paddle.static.data(
                name='weight', shape=[self.C], dtype=self.dtype
            )

            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight,
                reduction=self.reduction,
                soft_label=True,
                label_smoothing=self.label_smoothing,
            )
            ret = cross_entropy_loss(input, label)
            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': self.logits,
                    'label': self.labels,
                    'weight': self.weight,
                },
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)
        paddle.disable_static()

        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    # label_smoothing test end

    def test_cross_entropy_loss_1d_with_mean_ignore(self):
        input_np = np.random.random([2, 4]).astype(self.dtype)
        label_np = np.random.randint(0, 4, size=(2)).astype(np.int64)
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[2, 4], dtype=self.dtype
            )
            label = paddle.static.data(name='label', shape=[2], dtype='int64')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(ignore_index=0)
            ret = cross_entropy_loss(input, label)

            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': input_np,
                    'label': label_np,
                },
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)
        expected = cross_entropy_loss_1d(input_np, label_np)[0]

        with base.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                axis=1, ignore_index=0
            )
            dy_ret = cross_entropy_loss(
                paddle.to_tensor(input_np),
                paddle.to_tensor(label_np),
            )
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_1d(input_np, label_np, ignore_index=0)[0]
        np.testing.assert_allclose(static_ret[0], dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_cross_entropy_loss_1d_with_mean_ignore_negative(self):
        N = 100
        C = 200
        input_np = np.random.random([N, C]).astype(self.dtype)
        label_np = -np.ones(N).astype(np.int64)
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[N, C], dtype=self.dtype
            )
            label = paddle.static.data(name='label', shape=[N], dtype='int64')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                ignore_index=-1
            )
            ret = cross_entropy_loss(input, label)
            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': input_np,
                    'label': label_np,
                },
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)

        with base.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                axis=1, ignore_index=-1
            )
            dy_ret = cross_entropy_loss(
                paddle.to_tensor(input_np),
                paddle.to_tensor(label_np),
            )
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_1d(input_np, label_np, ignore_index=-1)[0]

        np.testing.assert_allclose(static_ret[0], dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_cross_entropy_loss_1d_with_weight_mean_ignore(self):
        N = 100
        C = 200
        input_np = np.random.random([N, C]).astype(self.dtype)
        label_np = np.random.randint(0, C, size=(N)).astype(np.int64)
        weight_np = np.random.random([C]).astype(self.dtype)
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[N, C], dtype=self.dtype
            )
            label = paddle.static.data(name='label', shape=[N], dtype='int64')
            weight = paddle.static.data(
                name='weight', shape=[C], dtype=self.dtype
            )  # weight for each class
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight, ignore_index=0
            )
            ret = cross_entropy_loss(input, label)

            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': input_np,
                    'label': label_np,
                    "weight": weight_np,
                },
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)

        with base.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=paddle.to_tensor(weight_np),
                axis=1,
                ignore_index=0,
            )
            dy_ret = cross_entropy_loss(
                paddle.to_tensor(input_np),
                paddle.to_tensor(label_np),
            )
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_1d(
            input_np, label_np, weight=weight_np, ignore_index=0
        )[0]

        np.testing.assert_allclose(static_ret[0], dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_cross_entropy_loss_1d_with_weight_mean_ignore_exceedlabel(self):
        N = 100
        C = 200
        input_np = np.random.random([N, C]).astype(self.dtype)
        label_np = np.random.randint(0, C, size=(N)).astype(np.int64)
        label_np[0] = 255
        weight_np = np.random.random([C]).astype(self.dtype)

        with base.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=paddle.to_tensor(weight_np), ignore_index=255
            )
            dy_ret = cross_entropy_loss(
                paddle.to_tensor(input_np),
                paddle.to_tensor(label_np),
            )
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_1d(
            input_np, label_np, weight=weight_np, ignore_index=255
        )[0]

        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_cross_entropy_loss_1d_with_weight_mean(self):
        input_np = np.random.random([2, 4]).astype(self.dtype)
        label_np = np.random.randint(0, 4, size=(2)).astype(np.int64)
        weight_np = np.random.random([4]).astype(self.dtype)  # shape:C
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[2, 4], dtype=self.dtype
            )
            label = paddle.static.data(name='label', shape=[2], dtype='int64')
            weight = paddle.static.data(
                name='weight', shape=[4], dtype=self.dtype
            )  # weight for each class
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(weight=weight)
            ret = cross_entropy_loss(input, label)

            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': input_np,
                    'label': label_np,
                    "weight": weight_np,
                },
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)
        expected = cross_entropy_loss_1d(input_np, label_np, weight=weight_np)[
            0
        ]

        with base.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=paddle.to_tensor(weight_np), axis=1
            )
            dy_ret = cross_entropy_loss(
                paddle.to_tensor(input_np),
                paddle.to_tensor(label_np),
            )
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_1d(input_np, label_np, weight=weight_np)[
            0
        ]
        np.testing.assert_allclose(static_ret[0], dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_cross_entropy_loss_1d_with_weight_sum(self):
        input_np = np.random.random([100, 200]).astype(self.dtype)  # N,C
        label_np = np.random.randint(0, 100, size=(100)).astype(np.int64)  # N,1
        weight_np = np.random.random([200]).astype(self.dtype)  # C
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[100, 200], dtype=self.dtype
            )
            label = paddle.static.data(name='label', shape=[100], dtype='int64')
            weight = paddle.static.data(
                name='weight', shape=[200], dtype=self.dtype
            )
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight, reduction='sum'
            )
            ret = cross_entropy_loss(input, label)

            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': input_np,
                    'label': label_np,
                    "weight": weight_np,
                },
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)
        with base.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=paddle.to_tensor(weight_np), reduction='sum'
            )
            dy_ret = cross_entropy_loss(
                paddle.to_tensor(input_np),
                paddle.to_tensor(label_np),
            )
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_1d(
            input_np, label_np, weight=weight_np, reduction='sum'
        )[0]
        np.testing.assert_allclose(static_ret[0], dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_cross_entropy_loss_1d_with_weight_none(self):
        input_np = np.random.random([100, 200]).astype(self.dtype)  # N,C
        label_np = np.random.randint(0, 100, size=(100)).astype(np.int64)  # N,1
        weight_np = np.random.random([200]).astype(self.dtype)  # C

        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[100, 200], dtype=self.dtype
            )
            label = paddle.static.data(name='label', shape=[100], dtype='int64')
            weight = paddle.static.data(
                name='weight', shape=[200], dtype=self.dtype
            )

            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight, reduction='none'
            )
            ret = cross_entropy_loss(input, label)

            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': input_np,
                    'label': label_np,
                    "weight": weight_np,
                },
                fetch_list=[ret],
            )
            static_ret = np.squeeze(static_ret)
            self.assertIsNotNone(static_ret)
        with base.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=paddle.to_tensor(weight_np), reduction='none'
            )
            dy_ret = cross_entropy_loss(
                paddle.to_tensor(input_np),
                paddle.to_tensor(label_np),
            )
            dy_ret_value = dy_ret.numpy()
            dy_ret_value = np.squeeze(dy_ret_value)
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_1d(
            input_np, label_np, weight=weight_np, reduction='none'
        )
        np.testing.assert_allclose(static_ret, dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret, expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_cross_entropy_loss_1d_with_weight_none_func(self):
        input_np = np.random.random([100, 200]).astype(self.dtype)  # N,C
        label_np = np.random.randint(0, 100, size=(100)).astype(np.int64)  # N
        weight_np = np.random.random([200]).astype(self.dtype)  # C
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[100, 200], dtype=self.dtype
            )
            label = paddle.static.data(name='label', shape=[100], dtype='int64')
            weight = paddle.static.data(
                name='weight', shape=[200], dtype=self.dtype
            )
            ret = paddle.nn.functional.cross_entropy(
                input, label, weight=weight, reduction='none'
            )

            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': input_np,
                    'label': label_np,
                    "weight": weight_np,
                },
                fetch_list=[ret],
            )
            static_ret = np.squeeze(static_ret)
            self.assertIsNotNone(static_ret)
        with base.dygraph.guard():
            dy_ret = paddle.nn.functional.cross_entropy(
                paddle.to_tensor(input_np),
                paddle.to_tensor(label_np),
                weight=paddle.to_tensor(weight_np),
                reduction='none',
            )
            dy_ret_value = dy_ret.numpy()
            dy_ret_value = np.squeeze(dy_ret_value)
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_1d(
            input_np, label_np, weight=weight_np, reduction='none'
        )
        np.testing.assert_allclose(static_ret, dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret, expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_cross_entropy_loss_1d_mean(self):
        input_np = np.random.random([100, 200]).astype(self.dtype)  # N,C
        label_np = np.random.randint(0, 100, size=(100)).astype(np.int64)  # N,1
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[100, 200], dtype=self.dtype
            )
            label = paddle.static.data(name='label', shape=[100], dtype='int64')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss()
            ret = cross_entropy_loss(input, label)
            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={'input': input_np, 'label': label_np},
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)
        with base.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss()
            dy_ret = cross_entropy_loss(
                paddle.to_tensor(input_np),
                paddle.to_tensor(label_np),
            )
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_1d(input_np, label_np)[0]
        np.testing.assert_allclose(static_ret[0], dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_cross_entropy_loss_1d_sum(self):
        input_np = np.random.random([100, 200]).astype(self.dtype)  # N,C
        label_np = np.random.randint(0, 100, size=(100)).astype(np.int64)  # N,1
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[100, 200], dtype=self.dtype
            )
            label = paddle.static.data(name='label', shape=[100], dtype='int64')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='sum'
            )
            ret = cross_entropy_loss(input, label)
            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={'input': input_np, 'label': label_np},
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)
        with base.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='sum'
            )
            dy_ret = cross_entropy_loss(
                paddle.to_tensor(input_np),
                paddle.to_tensor(label_np),
            )
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_1d(input_np, label_np, reduction='sum')[0]
        np.testing.assert_allclose(static_ret[0], dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_cross_entropy_loss_1d_none(self):
        input_np = np.random.random([100, 200]).astype(self.dtype)  # N,C
        label_np = np.random.randint(0, 100, size=(100)).astype(np.int64)  # N,1
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[100, 200], dtype=self.dtype
            )
            label = paddle.static.data(name='label', shape=[100], dtype='int64')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='none'
            )
            ret = cross_entropy_loss(input, label)
            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={'input': input_np, 'label': label_np},
                fetch_list=[ret],
            )
            static_ret = np.squeeze(static_ret)
            self.assertIsNotNone(static_ret)
        with base.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='none'
            )
            dy_ret = cross_entropy_loss(
                paddle.to_tensor(input_np),
                paddle.to_tensor(label_np),
            )
            dy_ret_value = dy_ret.numpy()
            dy_ret_value = np.squeeze(dy_ret_value)
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_1d(input_np, label_np, reduction='none')
        np.testing.assert_allclose(static_ret, dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret, expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_cross_entropy_loss_2d_with_weight_none(self):
        input_np = np.random.random(size=(2, 2, 2, 3)).astype(
            self.dtype
        )  # NHWC
        label_np = np.random.randint(0, 3, size=(2, 2, 2)).astype(
            np.int64
        )  # NHW1
        weight_np = np.random.random(size=(3,)).astype(self.dtype)  # C

        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[2, 2, 2, 3], dtype=self.dtype
            )
            label = paddle.static.data(
                name='label', shape=[2, 2, 2], dtype='int64'
            )
            weight = paddle.static.data(
                name='weight', shape=[3], dtype=self.dtype
            )
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight, reduction='none'
            )
            ret = cross_entropy_loss(input, label)

            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': input_np,
                    'label': label_np,
                    "weight": weight_np,
                },
                fetch_list=[ret],
            )
            static_ret = np.squeeze(static_ret)
            self.assertIsNotNone(static_ret)
        with base.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=paddle.to_tensor(weight_np), reduction='none'
            )
            dy_ret = cross_entropy_loss(
                paddle.to_tensor(input_np),
                paddle.to_tensor(label_np),
            )
            dy_ret_value = dy_ret.numpy()
            dy_ret_value = np.squeeze(dy_ret_value)
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_2d(
            input_np, label_np, weight=weight_np, reduction='none'
        )
        np.testing.assert_allclose(static_ret, dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret, expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_cross_entropy_loss_2d_with_weight_axis_change_mean(self):
        input_np = np.random.random(size=(2, 3, 2, 2)).astype(
            self.dtype
        )  # NCHW
        label_np = np.random.randint(0, 3, size=(2, 2, 2)).astype(
            np.int64
        )  # NHW
        weight_np = np.random.random(size=(3,)).astype(self.dtype)  # C

        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[2, 3, 2, 2], dtype=self.dtype
            )
            label = paddle.static.data(
                name='label', shape=[2, 2, 2], dtype='int64'
            )
            weight = paddle.static.data(
                name='weight', shape=[3], dtype=self.dtype
            )
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight, reduction='mean', axis=1
            )
            # specify the class channels to axis 1
            ret = cross_entropy_loss(input, label)

            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': input_np,
                    'label': label_np,
                    "weight": weight_np,
                },
                fetch_list=[ret],
            )

            self.assertIsNotNone(static_ret)
        with base.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=paddle.to_tensor(weight_np),
                reduction='mean',
                axis=1,
            )
            dy_ret = cross_entropy_loss(
                paddle.to_tensor(input_np),
                paddle.to_tensor(label_np),
            )
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_2d(
            np.transpose(input_np, [0, 2, 3, 1]),
            label_np,
            weight=weight_np,
            reduction='mean',
        )[0]
        np.testing.assert_allclose(static_ret[0], dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_cross_entropy_loss_2d_with_weight_mean_ignore_exceedlabel(self):
        N = 4
        C = 3
        H = 512
        W = 512
        input_np = np.random.random([N, H, W, C]).astype(self.dtype)
        label_np = np.random.randint(0, C, size=(N, H, W)).astype(np.int64)
        label_np[0, 0, 0] = 255
        weight_np = np.random.random([C]).astype(self.dtype)
        with base.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=paddle.to_tensor(weight_np), ignore_index=255
            )
            dy_ret = cross_entropy_loss(
                paddle.to_tensor(input_np),
                paddle.to_tensor(label_np),
            )
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_2d(
            input_np, label_np, weight=weight_np, ignore_index=255
        )[0]
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_cross_entropy_loss_2d_with_weight_mean(self):
        input_np = np.random.random(size=(2, 2, 2, 3)).astype(
            self.dtype
        )  # NHWC
        label_np = np.random.randint(0, 3, size=(2, 2, 2)).astype(
            np.int64
        )  # NHW
        weight_np = np.random.random(size=(3,)).astype(self.dtype)  # C
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[2, 2, 2, 3], dtype=self.dtype
            )
            label = paddle.static.data(
                name='label', shape=[2, 2, 2], dtype='int64'
            )
            weight = paddle.static.data(
                name='weight', shape=[3], dtype=self.dtype
            )
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight, reduction='mean'
            )
            ret = cross_entropy_loss(input, label)

            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': input_np,
                    'label': label_np,
                    "weight": weight_np,
                },
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)
        with base.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=paddle.to_tensor(weight_np), reduction='mean'
            )
            dy_ret = cross_entropy_loss(
                paddle.to_tensor(input_np),
                paddle.to_tensor(label_np),
            )
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_2d(
            input_np, label_np, weight=weight_np, reduction='mean'
        )[0]
        np.testing.assert_allclose(static_ret[0], dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_cross_entropy_loss_2d_with_weight_sum(self):
        input_np = np.random.random(size=(2, 2, 2, 3)).astype(
            self.dtype
        )  # NHWC
        label_np = np.random.randint(0, 3, size=(2, 2, 2)).astype(
            np.int64
        )  # NHW
        weight_np = np.random.random(size=(3,)).astype(self.dtype)  # C
        paddle.enable_static()

        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[2, 2, 2, 3], dtype=self.dtype
            )
            label = paddle.static.data(
                name='label', shape=[2, 2, 2], dtype='int64'
            )
            weight = paddle.static.data(
                name='weight', shape=[3], dtype=self.dtype
            )
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight, reduction='sum'
            )
            ret = cross_entropy_loss(input, label)

            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': input_np,
                    'label': label_np,
                    "weight": weight_np,
                },
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)
        with base.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=paddle.to_tensor(weight_np), reduction='sum'
            )
            dy_ret = cross_entropy_loss(
                paddle.to_tensor(input_np),
                paddle.to_tensor(label_np),
            )
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_2d(
            input_np, label_np, weight=weight_np, reduction='sum'
        )[0]
        np.testing.assert_allclose(static_ret[0], dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_cross_entropy_loss_2d_none(self):
        input_np = np.random.random(size=(2, 2, 2, 3)).astype(
            self.dtype
        )  # NHWC
        label_np = np.random.randint(0, 3, size=(2, 2, 2)).astype(
            np.int64
        )  # NHW
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[2, 2, 2, 3], dtype=self.dtype
            )
            label = paddle.static.data(
                name='label', shape=[2, 2, 2], dtype='int64'
            )
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='none'
            )
            ret = cross_entropy_loss(input, label)
            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': input_np,
                    'label': label_np,
                },
                fetch_list=[ret],
            )
            static_ret = np.squeeze(static_ret)
            self.assertIsNotNone(static_ret)
        with base.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='none'
            )
            dy_ret = cross_entropy_loss(
                paddle.to_tensor(input_np),
                paddle.to_tensor(label_np),
            )
            dy_ret_value = dy_ret.numpy()
            dy_ret_value = np.squeeze(dy_ret_value)
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_2d(input_np, label_np, reduction='none')
        np.testing.assert_allclose(static_ret, dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret, expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_cross_entropy_loss_2d_mean(self):
        input_np = np.random.random(size=(2, 2, 2, 3)).astype(
            self.dtype
        )  # NHWC
        label_np = np.random.randint(0, 3, size=(2, 2, 2)).astype(
            np.int64
        )  # NHW
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[2, 2, 2, 3], dtype=self.dtype
            )
            label = paddle.static.data(
                name='label', shape=[2, 2, 2], dtype='int64'
            )
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='mean'
            )
            ret = cross_entropy_loss(input, label)

            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': input_np,
                    'label': label_np,
                },
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)
        with base.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='mean'
            )
            dy_ret = cross_entropy_loss(
                paddle.to_tensor(input_np),
                paddle.to_tensor(label_np),
            )
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_2d(input_np, label_np, reduction='mean')[
            0
        ]
        np.testing.assert_allclose(static_ret[0], dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)

    def test_cross_entropy_loss_2d_sum(self):
        input_np = np.random.random(size=(2, 2, 2, 3)).astype(
            self.dtype
        )  # NHWC
        label_np = np.random.randint(0, 3, size=(2, 2, 2)).astype(
            np.int64
        )  # NHW
        paddle.enable_static()
        prog = base.Program()
        startup_prog = base.Program()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        with base.program_guard(prog, startup_prog):
            input = paddle.static.data(
                name='input', shape=[2, 2, 2, 3], dtype=self.dtype
            )
            label = paddle.static.data(
                name='label', shape=[2, 2, 2], dtype='int64'
            )
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='sum'
            )
            ret = cross_entropy_loss(input, label)

            exe = base.Executor(place)
            static_ret = exe.run(
                prog,
                feed={
                    'input': input_np,
                    'label': label_np,
                },
                fetch_list=[ret],
            )
            self.assertIsNotNone(static_ret)
        with base.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='sum'
            )
            dy_ret = cross_entropy_loss(
                paddle.to_tensor(input_np),
                paddle.to_tensor(label_np),
            )
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_2d(input_np, label_np, reduction='sum')[0]
        np.testing.assert_allclose(static_ret[0], dy_ret_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret[0], expected, rtol=1e-05)
        np.testing.assert_allclose(dy_ret_value, expected, rtol=1e-05)


class TestCrossEntropyFAPIError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):

            def test_WeightLength_NotEqual():
                input_data = paddle.rand(shape=[20, 100])
                label_data = paddle.randint(
                    0, 100, shape=[20, 1], dtype="int64"
                )
                weight_data = paddle.rand([100 + 1])
                paddle.nn.functional.cross_entropy(
                    input=input_data,
                    label=label_data,
                    weight=weight_data,
                    ignore_index=-100,
                )

            self.assertRaises(ValueError, test_WeightLength_NotEqual)

            def test_LabelValue_ExceedMax():
                input_data = paddle.rand(shape=[20, 100])
                label_data = paddle.randint(
                    0, 100, shape=[20, 1], dtype="int64"
                )
                label_data[0] = 100
                weight_data = paddle.rand([100])
                paddle.nn.functional.cross_entropy(
                    input=input_data,
                    label=label_data,
                    weight=weight_data,
                    ignore_index=-100,
                )

            self.assertRaises(IndexError, test_LabelValue_ExceedMax)

            def test_LabelValue_ExceedMin():
                input_data = paddle.rand(shape=[20, 100])
                label_data = paddle.randint(
                    0, 100, shape=[20, 1], dtype="int64"
                )
                label_data[0] = -1
                weight_data = paddle.rand([100])
                paddle.nn.functional.cross_entropy(
                    input=input_data,
                    label=label_data,
                    weight=weight_data,
                    ignore_index=-100,
                )

            self.assertRaises(IndexError, test_LabelValue_ExceedMin)

            def static_test_WeightLength_NotEqual():
                input_np = np.random.random([2, 4]).astype('float32')
                label_np = np.random.randint(0, 4, size=(2)).astype(np.int64)
                weight_np = np.random.random([3]).astype('float32')
                paddle.enable_static()
                prog = base.Program()
                startup_prog = base.Program()
                place = (
                    base.CUDAPlace(0)
                    if base.core.is_compiled_with_cuda()
                    else base.CPUPlace()
                )
                with base.program_guard(prog, startup_prog):
                    input = paddle.static.data(
                        name='input', shape=[2, 4], dtype='float32'
                    )
                    label = paddle.static.data(
                        name='label', shape=[2], dtype='int64'
                    )
                    weight = paddle.static.data(
                        name='weight', shape=[3], dtype='float32'
                    )  # weight for each class
                    cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                        weight=weight
                    )
                    ret = cross_entropy_loss(input, label)

                    exe = base.Executor(place)
                    static_ret = exe.run(
                        prog,
                        feed={
                            'input': input_np,
                            'label': label_np,
                            "weight": weight_np,
                        },
                        fetch_list=[ret],
                    )
                    self.assertIsNotNone(static_ret)

            self.assertRaises(ValueError, static_test_WeightLength_NotEqual)


if __name__ == "__main__":
    unittest.main()
