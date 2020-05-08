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

from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy as np
import unittest


def stable_softmax(x):
    shiftx = (x - np.max(x)).clip(-64.)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def cross_entropy_loss_np(input,
                          label,
                          weight=None,
                          reduction='mean',
                          ignore_index=-100):
    softmax_out = np.apply_along_axis(stable_softmax, -1, input)
    log_softmax_out = np.log(softmax_out)
    input_shape = log_softmax_out.shape
    N = input_shape[0]
    C = input_shape[1]
    out = np.zeros_like(label).astype(np.float64)
    total_weight = 0
    for i in range(N):
        cur_target = label[i]
        if cur_target == ignore_index:
            out[i] = 0
            continue
        cur_weight = weight[cur_target] if weight is not None else 1
        total_weight += cur_weight
        out[i] = -log_softmax_out[i][cur_target] * cur_weight
    if reduction == 'sum':
        return np.sum(out), np.array([total_weight]).astype('float64')
    elif reduction == 'mean':
        return out.sum() / total_weight, np.array(
            [total_weight]).astype('float64')
    elif reduction == 'none':
        return out


class CrossEntropyLoss(unittest.TestCase):
    def test_cross_entropy_loss_with_weight_mean(self):
        input_np = np.random.random([100, 200]).astype(np.float64)
        label_np = np.random.randint(0, 100, size=(100, )).astype(np.int64)
        weight_np = np.random.random([200]).astype(np.float64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(name='input', shape=[100, 200], dtype='float64')
            label = fluid.data(name='label', shape=[100], dtype='int64')
            weight = fluid.data(name='weight', shape=[200], dtype='float64')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(weight=weight)
            ret = cross_entropy_loss(input, label)

            exe = fluid.Executor(place)
            static_ret = exe.run(prog,
                                 feed={
                                     'input': input_np,
                                     'label': label_np,
                                     "weight": weight_np
                                 },
                                 fetch_list=[ret])
            self.assertIsNotNone(static_ret)
        with fluid.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=fluid.dygraph.to_variable(weight_np))
            dy_ret = cross_entropy_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_np(
            input_np, label_np, weight=weight_np)[0]
        self.assertTrue(np.allclose(static_ret, dy_ret_value))
        self.assertTrue(np.allclose(static_ret, expected))
        self.assertTrue(np.allclose(dy_ret_value, expected))

    def test_cross_entropy_loss_with_weight_sum(self):
        input_np = np.random.random([100, 200]).astype(np.float64)
        label_np = np.random.randint(0, 100, size=(100, )).astype(np.int64)
        weight_np = np.random.random([200]).astype(np.float64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(name='input', shape=[100, 200], dtype='float64')
            label = fluid.data(name='label', shape=[100], dtype='int64')
            weight = fluid.data(name='weight', shape=[200], dtype='float64')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight, reduction='sum')
            ret = cross_entropy_loss(input, label)

            exe = fluid.Executor(place)
            static_ret = exe.run(prog,
                                 feed={
                                     'input': input_np,
                                     'label': label_np,
                                     "weight": weight_np
                                 },
                                 fetch_list=[ret])
            self.assertIsNotNone(static_ret)
        with fluid.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=fluid.dygraph.to_variable(weight_np), reduction='sum')
            dy_ret = cross_entropy_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_np(
            input_np, label_np, weight=weight_np, reduction='sum')[0]
        self.assertTrue(np.allclose(static_ret, dy_ret_value))
        self.assertTrue(np.allclose(static_ret, expected))
        self.assertTrue(np.allclose(dy_ret_value, expected))

    def test_cross_entropy_loss_with_weight_none(self):
        input_np = np.random.random([100, 200]).astype(np.float64)
        label_np = np.random.randint(0, 100, size=(100, )).astype(np.int64)
        weight_np = np.random.random([200]).astype(np.float64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(name='input', shape=[100, 200], dtype='float64')
            label = fluid.data(name='label', shape=[100], dtype='int64')
            weight = fluid.data(name='weight', shape=[200], dtype='float64')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight, reduction='none')
            ret = cross_entropy_loss(input, label)

            exe = fluid.Executor(place)
            static_ret = exe.run(prog,
                                 feed={
                                     'input': input_np,
                                     'label': label_np,
                                     "weight": weight_np
                                 },
                                 fetch_list=[ret])
            self.assertIsNotNone(static_ret)
        with fluid.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=fluid.dygraph.to_variable(weight_np), reduction='none')
            dy_ret = cross_entropy_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_np(
            input_np, label_np, weight=weight_np, reduction='none')
        self.assertTrue(np.allclose(static_ret, dy_ret_value))
        self.assertTrue(np.allclose(static_ret, expected))
        self.assertTrue(np.allclose(dy_ret_value, expected))

    def test_cross_entropy_loss_mean(self):
        input_np = np.random.random([100, 200]).astype(np.float64)
        label_np = np.random.randint(0, 100, size=(100, )).astype(np.int64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(name='input', shape=[100, 200], dtype='float64')
            label = fluid.data(name='label', shape=[100], dtype='int64')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss()
            ret = cross_entropy_loss(input, label)

            exe = fluid.Executor(place)
            static_ret = exe.run(prog,
                                 feed={'input': input_np,
                                       'label': label_np},
                                 fetch_list=[ret])
            self.assertIsNotNone(static_ret)
        with fluid.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss()
            dy_ret = cross_entropy_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_np(input_np, label_np)[0]
        self.assertTrue(np.allclose(static_ret, dy_ret_value))
        self.assertTrue(np.allclose(static_ret, expected))
        self.assertTrue(np.allclose(dy_ret_value, expected))

    def test_cross_entropy_loss_sum(self):
        input_np = np.random.random([100, 200]).astype(np.float64)
        label_np = np.random.randint(0, 100, size=(100, )).astype(np.int64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(name='input', shape=[100, 200], dtype='float64')
            label = fluid.data(name='label', shape=[100], dtype='int64')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='sum')
            ret = cross_entropy_loss(input, label)

            exe = fluid.Executor(place)
            static_ret = exe.run(prog,
                                 feed={'input': input_np,
                                       'label': label_np},
                                 fetch_list=[ret])
            self.assertIsNotNone(static_ret)
        with fluid.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='sum')
            dy_ret = cross_entropy_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_np(input_np, label_np, reduction='sum')[0]
        self.assertTrue(np.allclose(static_ret, dy_ret_value))
        self.assertTrue(np.allclose(static_ret, expected))
        self.assertTrue(np.allclose(dy_ret_value, expected))

    def test_cross_entropy_loss_none(self):
        input_np = np.random.random([100, 200]).astype(np.float64)
        label_np = np.random.randint(0, 100, size=(100, )).astype(np.int64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(name='input', shape=[100, 200], dtype='float64')
            label = fluid.data(name='label', shape=[100], dtype='int64')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='none')
            ret = cross_entropy_loss(input, label)

            exe = fluid.Executor(place)
            static_ret = exe.run(prog,
                                 feed={'input': input_np,
                                       'label': label_np},
                                 fetch_list=[ret])
            self.assertIsNotNone(static_ret)
        with fluid.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='none')
            dy_ret = cross_entropy_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_np(input_np, label_np, reduction='none')
        self.assertTrue(np.allclose(static_ret, dy_ret_value))
        self.assertTrue(np.allclose(static_ret, expected))
        self.assertTrue(np.allclose(dy_ret_value, expected))


if __name__ == "__main__":
    unittest.main()
