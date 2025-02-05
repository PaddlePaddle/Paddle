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

import os
import unittest

import numpy as np

import paddle
from paddle.distributed import fleet
from paddle.distributed.collective import _c_softmax_with_cross_entropy


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    # clip to shiftx, otherwise, when calc loss with
    # log(exp(shiftx)), may get log(0)=INF
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def cross_entropy(softmax, label, soft_label, axis, ignore_index=-1):
    if soft_label:
        return (-label * np.log(softmax)).sum(axis=axis, keepdims=True)
    shape = softmax.shape
    axis %= len(shape)
    n = int(np.prod(shape[:axis]))
    axis_dim = shape[axis]
    remain = int(np.prod(shape[axis + 1 :]))
    softmax_reshape = softmax.reshape((n, axis_dim, remain))
    label_reshape = label.reshape((n, 1, remain))
    result = np.zeros_like(label_reshape, dtype=softmax.dtype)
    for i in range(n):
        for j in range(remain):
            lbl = label_reshape[i, 0, j]
            if lbl != ignore_index:
                result[i, 0, j] -= np.log(softmax_reshape[i, lbl, j])
    return result.reshape(label.shape)


def softmax_with_cross_entropy_grad(softmax, label, loss_grad, axis):
    logit_grad = softmax.copy()
    shape = softmax.shape
    axis %= len(shape)
    n = int(np.prod(shape[:axis]))
    d = int(np.prod(shape[axis:]))
    for i in range(n * d):
        row = int(i / d)
        col = i % d
        if col == label[row]:
            logit_grad[row][col] = (logit_grad[row][col] - 1.0) * loss_grad[row]
        else:
            logit_grad[row][col] = logit_grad[row][col] * loss_grad[row]
    return logit_grad


class TestCSoftmaxWithCrossEntropy(unittest.TestCase):
    def setUp(self):
        self.num_class = 1000
        self.batch_size = 1024
        fleet.init(is_collective=True)
        strategy = fleet.DistributedStrategy()
        strategy.tensor_parallel = True
        strategy.tensor_parallel_configs = {'tensor_parallel_degree': 2}

    def test_model(self, data_type="float32"):
        rank = fleet.worker_index()

        # get data that is shared by both ranks
        np.random.seed(os.getuid())
        label = np.random.randint(
            0, self.num_class, size=(self.batch_size, 1), dtype='int32'
        )
        ignore_index = label[0][0]

        local_elements = int(self.num_class / 2)
        # get input data for rank 0
        np.random.seed(0)
        input0 = np.random.uniform(
            low=-40.0, high=40.0, size=(self.batch_size, local_elements)
        ).astype(data_type)

        # get input data for rank 1
        np.random.seed(1)
        input1 = np.random.uniform(
            low=-40.0, high=40.0, size=(self.batch_size, local_elements)
        ).astype(data_type)

        # get combined input data
        inputs = np.concatenate((input0, input1), axis=1)

        if rank == 0:
            loss, softmax = _c_softmax_with_cross_entropy(
                paddle.to_tensor(input0),
                paddle.to_tensor(label),
                ignore_index=ignore_index,
                return_softmax=True,
            )
        else:
            loss, softmax = _c_softmax_with_cross_entropy(
                paddle.to_tensor(input1),
                paddle.to_tensor(label),
                ignore_index=ignore_index,
                return_softmax=True,
            )
        paddle.device.cuda.synchronize()
        softmax_list = []
        paddle.distributed.all_gather(softmax_list, softmax)

        # calculate analytic result
        need_softmax = np.apply_along_axis(stable_softmax, 1, inputs)
        need_loss = cross_entropy(
            need_softmax, label, False, 1, ignore_index=ignore_index
        )

        softmax = np.concatenate(
            (softmax_list[0].numpy(), softmax_list[1].numpy()), axis=1
        )

        # compare results
        rtol = 1e-6
        np.testing.assert_allclose(loss.numpy(), need_loss, rtol=rtol)
        np.testing.assert_allclose(softmax, need_softmax, rtol=rtol)

    def test_model_soft_label(self, data_type="float32"):
        rank = fleet.worker_index()

        # get data that is shared by both ranks
        np.random.seed(os.getuid())

        C = 4

        label = []
        for _ in range(self.batch_size):
            tmp = np.random.choice(
                range(0, self.num_class), size=C, replace=False
            )
            label.append(tmp)
        label = np.array(label)

        ignore_index = -100

        local_elements = int(self.num_class / 2)
        # get input data for rank 0
        np.random.seed(0)
        input0 = np.random.uniform(
            low=-40.0, high=40.0, size=(self.batch_size, local_elements)
        ).astype(data_type)

        # get input data for rank 1
        np.random.seed(1)
        input1 = np.random.uniform(
            low=-40.0, high=40.0, size=(self.batch_size, local_elements)
        ).astype(data_type)

        # get combined input data
        inputs = np.concatenate((input0, input1), axis=1)

        if rank == 0:
            loss, softmax = _c_softmax_with_cross_entropy(
                paddle.to_tensor(input0),
                paddle.to_tensor(label),
                ignore_index=ignore_index,
                return_softmax=True,
            )
        else:
            loss, softmax = _c_softmax_with_cross_entropy(
                paddle.to_tensor(input1),
                paddle.to_tensor(label),
                ignore_index=ignore_index,
                return_softmax=True,
            )
        paddle.device.cuda.synchronize()
        softmax_list = []
        paddle.distributed.all_gather(softmax_list, softmax)

        # calculate analytic result
        need_softmax = np.apply_along_axis(stable_softmax, 1, inputs)
        prob = 1.0 / C
        need_label = np.zeros_like(inputs)

        for i in range(len(label)):
            for c in label[i]:
                need_label[i][c] = prob

        need_loss = cross_entropy(
            need_softmax, need_label, True, 1, ignore_index=ignore_index
        )

        softmax = np.concatenate(
            (softmax_list[0].numpy(), softmax_list[1].numpy()), axis=1
        )

        # compare results
        rtol = 1e-6
        np.testing.assert_allclose(loss.numpy(), need_loss, rtol=rtol)
        np.testing.assert_allclose(softmax, need_softmax, rtol=rtol)


if __name__ == '__main__':
    unittest.main()
