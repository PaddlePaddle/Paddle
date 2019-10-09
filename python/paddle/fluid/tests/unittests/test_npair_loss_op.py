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


class TestNpairLossOp(unittest.TestCase):
    def setUp(self):
        self.dtype = np.float32

    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        self.assertTrue(np.allclose(np.array(tensor), np_array, atol=atol), msg)

    def test_npair_loss(self):
        reg_lambda = 0.002
        num_data, feat_dim, num_classes = 18, 6, 3

        place = core.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        embeddings_anchor = np.random.rand(num_data,
                                           feat_dim).astype(np.float32)
        embeddings_positive = np.random.rand(num_data,
                                             feat_dim).astype(np.float32)
        row_labels = np.random.randint(
            0, num_classes, size=(num_data)).astype(np.float32)
        out_loss = npairloss(
            embeddings_anchor,
            embeddings_positive,
            row_labels,
            l2_reg=reg_lambda)

        anc = fluid.layers.create_tensor(
            dtype='float32', persistable=True, name='anc')
        pos = fluid.layers.create_tensor(
            dtype='float32', persistable=True, name='pos')
        lab = fluid.layers.create_tensor(
            dtype='float32', persistable=True, name='lab')
        fluid.layers.assign(input=embeddings_anchor, output=anc)
        fluid.layers.assign(input=embeddings_positive, output=pos)
        fluid.layers.assign(input=row_labels, output=lab)

        npair_loss_op = fluid.layers.npair_loss(
            anchor=anc, positive=pos, labels=lab, l2_reg=reg_lambda)
        out_tensor = exe.run(feed={'anc': anc,
                                   'pos': pos,
                                   'lab': lab},
                             fetch_list=[npair_loss_op.name])

        self.__assert_close(
            out_tensor,
            out_loss,
            "inference output are different at " + str(place) + ", " +
            str(np.dtype('float32')) + str(np.array(out_tensor)) +
            str(out_loss),
            atol=1e-3)


if __name__ == '__main__':
    unittest.main()
