#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import paddle.fluid as fluid
import paddle.fluid.core as core
import numpy as np


def npairloss(anchor, positive, labels, l2_reg=0.002):
    def softmax_cross_entropy_with_logits(logits, labels):
        logits = np.exp(logits)
        logits = logits / np.sum(logits, axis=1).reshape(-1, 1)

        return np.mean(
            -np.sum(labels * np.log(logits), axis=1), dtype=np.float32)

    batch_size = labels.shape[0]

    labels = np.reshape(labels, (batch_size, 1))
    labels = np.equal(labels, labels.transpose()).astype(float)
    labels = labels / np.sum(labels, axis=1, keepdims=True)

    l2loss = np.mean(np.sum(np.power(anchor, 2), 1)) + np.mean(
        np.sum(np.power(positive, 2), 1))
    l2loss = (l2loss * 0.25 * l2_reg).astype(np.float32)

    similarity_matrix = np.matmul(anchor, positive.transpose())
    celoss = np.mean(
        softmax_cross_entropy_with_logits(similarity_matrix, labels))

    return l2loss + celoss


def create_or_get_tensor(scope, var_name, var, place):
    tensor = scope.var(var_name).get_tensor()
    if var is not None:
        assert isinstance(var, np.ndarray)
        tensor.set_recursive_sequence_lengths([])
        tensor.set(var, place)
    return tensor


class TestNpairLossOp(unittest.TestCase):
    def setUp(self):
        self.dtype = np.float32

    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        self.assertTrue(np.allclose(np.array(tensor), np_array, atol=atol), msg)

    def check_with_place(self, place, dtype, shape):
        reg_lambda = 0.002
        num_data, feat_dim, num_classes = shape[0], shape[1], shape[2]

        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        embeddings_anchor = np.random.rand(num_data,
                                           feat_dim).astype(np.float32)
        embeddings_positive = np.random.rand(num_data,
                                             feat_dim).astype(np.float32)
        labels = np.random.randint(
            0, num_classes, size=(num_data)).astype(np.float32)
        out_loss = npairloss(
            embeddings_anchor, embeddings_positive, labels, l2_reg=reg_lambda)

        anchor_tensor = fluid.layers.data(
            name='anchor',
            shape=[num_data, feat_dim],
            dtype=self.dtype,
            append_batch_size=False)
        positive_tensor = fluid.layers.data(
            name='positive',
            shape=[num_data, feat_dim],
            dtype=self.dtype,
            append_batch_size=False)
        labels_tensor = fluid.layers.data(
            name='labels',
            shape=[num_data],
            dtype=self.dtype,
            append_batch_size=False)

        npair_loss_op = fluid.layers.npair_loss(
            anchor=anchor_tensor,
            positive=positive_tensor,
            labels=labels_tensor,
            l2_reg=reg_lambda)
        out_tensor = exe.run(feed={
            'anchor': embeddings_anchor,
            'positive': embeddings_positive,
            'labels': labels
        },
                             fetch_list=[npair_loss_op.name])

        self.__assert_close(
            out_tensor,
            out_loss,
            "inference output are different at " + str(place) + ", " +
            str(np.dtype(dtype)) + str(np.array(out_tensor)) + str(out_loss),
            atol=1e-3)

    def test_check_output(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu("npair_loss"):
            places.append(core.CUDAPlace(0))

        for place in places:
            self.check_with_place(place, self.dtype, [18, 6, 3])


if __name__ == '__main__':
    unittest.main()
