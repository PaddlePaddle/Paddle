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

import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base import core


def npairloss(anchor, positive, labels, l2_reg=0.002):
    def softmax_cross_entropy_with_logits(logits, labels):
        logits = np.exp(logits)
        logits = logits / np.sum(logits, axis=1).reshape(-1, 1)

        return np.mean(
            -np.sum(labels * np.log(logits), axis=1), dtype=np.float32
        )

    batch_size = labels.shape[0]

    labels = np.reshape(labels, (batch_size, 1))
    labels = np.equal(labels, labels.transpose()).astype(float)
    labels = labels / np.sum(labels, axis=1, keepdims=True)

    l2loss = np.mean(np.sum(np.power(anchor, 2), 1)) + np.mean(
        np.sum(np.power(positive, 2), 1)
    )
    l2loss = (l2loss * 0.25 * l2_reg).astype(np.float32)

    similarity_matrix = np.matmul(anchor, positive.transpose())
    celoss = np.mean(
        softmax_cross_entropy_with_logits(similarity_matrix, labels)
    )

    return l2loss + celoss


class TestNpairLossOp(unittest.TestCase):
    def setUp(self):
        self.dtype = np.float32

    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        np.testing.assert_allclose(
            np.array(tensor), np_array, rtol=1e-05, atol=atol, err_msg=msg
        )

    def test_npair_loss(self):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            reg_lambda = 0.002
            num_data, feat_dim, num_classes = 18, 6, 3

            place = core.CPUPlace()
            exe = base.Executor(place)
            exe.run(startup)

            embeddings_anchor = np.random.rand(num_data, feat_dim).astype(
                np.float32
            )
            embeddings_positive = np.random.rand(num_data, feat_dim).astype(
                np.float32
            )
            row_labels = np.random.randint(
                0, num_classes, size=(num_data)
            ).astype(np.float32)
            out_loss = npairloss(
                embeddings_anchor,
                embeddings_positive,
                row_labels,
                l2_reg=reg_lambda,
            )

            anc = paddle.static.data(
                dtype='float32',
                name='anc',
                shape=embeddings_anchor.shape,
            )
            pos = paddle.static.data(
                dtype='float32',
                name='pos',
                shape=embeddings_positive.shape,
            )
            lab = paddle.static.data(
                dtype='float32',
                name='lab',
                shape=row_labels.shape,
            )

            npair_loss_op = paddle.nn.functional.npair_loss(
                anchor=anc, positive=pos, labels=lab, l2_reg=reg_lambda
            )
            out_tensor = exe.run(
                feed={
                    'anc': embeddings_anchor,
                    'pos': embeddings_positive,
                    'lab': row_labels,
                },
                fetch_list=[npair_loss_op],
            )

            self.__assert_close(
                out_tensor,
                out_loss,
                "inference output are different at "
                + str(place)
                + ", "
                + str(np.dtype('float32'))
                + str(np.array(out_tensor))
                + str(out_loss),
                atol=1e-3,
            )


class TestNpairLossOpError(unittest.TestCase):

    def test_errors(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            anchor_np = np.random.random((2, 4)).astype("float32")
            positive_np = np.random.random((2, 4)).astype("float32")
            labels_np = np.random.random(2).astype("float32")
            anchor_data = paddle.static.data(
                name='anchor', shape=[2, 4], dtype='float32'
            )
            positive_data = paddle.static.data(
                name='positive', shape=[2, 4], dtype='float32'
            )
            labels_data = paddle.static.data(
                name='labels', shape=[2], dtype='float32'
            )

            def test_anchor_Variable():
                # the anchor type must be Variable
                paddle.nn.functional.npair_loss(
                    anchor=anchor_np, positive=positive_data, labels=labels_data
                )

            def test_positive_Variable():
                # the positive type must be Variable
                paddle.nn.functional.npair_loss(
                    anchor=anchor_data, positive=positive_np, labels=labels_data
                )

            def test_labels_Variable():
                # the labels type must be Variable
                paddle.nn.functional.npair_loss(
                    anchor=anchor_data, positive=positive_data, labels=labels_np
                )

            self.assertRaises(TypeError, test_anchor_Variable)
            self.assertRaises(TypeError, test_positive_Variable)
            self.assertRaises(TypeError, test_labels_Variable)

            def test_anchor_type():
                # dtype must be float32 or float64
                anchor_data1 = paddle.static.data(
                    name='anchor1', shape=[2, 4], dtype='int32'
                )
                paddle.nn.functional.npair_loss(
                    anchor=anchor_data, positive=positive_data, labels=labels_np
                )

            def test_positive_type():
                # dtype must be float32 or float64
                positive_data1 = paddle.static.data(
                    name='positive1', shape=[2, 4], dtype='int32'
                )
                paddle.nn.functional.npair_loss(
                    anchor=anchor_data,
                    positive=positive_data1,
                    labels=labels_np,
                )

            def test_labels_type():
                # dtype must be float32 or float64
                labels_data1 = paddle.static.data(
                    name='labels1', shape=[2], dtype='int32'
                )
                paddle.nn.functional.npair_loss(
                    anchor=anchor_data,
                    positive=positive_data,
                    labels=labels_data1,
                )

            self.assertRaises(TypeError, test_anchor_type)
            self.assertRaises(TypeError, test_positive_type)
            self.assertRaises(TypeError, test_labels_type)


class TestNpairLossZeroError(unittest.TestCase):
    def test_errors(self):
        with paddle.base.dygraph.guard():

            def test_anchor_0_size():
                array = np.array([], dtype=np.float32)
                anchor = paddle.to_tensor(
                    np.reshape(array, [0, 0, 0]), dtype='float32'
                )
                positive = paddle.to_tensor(
                    np.reshape(array, [0]), dtype='float32'
                )
                array = np.array([1, 2, 3, 4], dtype=np.float32)
                labels = paddle.to_tensor(
                    np.reshape(array, [4]), dtype='float32'
                )
                paddle.nn.functional.npair_loss(anchor, positive, labels)

            def test_positive_0_size():
                array = np.array([1], dtype=np.float32)
                array1 = np.array([], dtype=np.float32)
                anchor = paddle.to_tensor(
                    np.reshape(array, [1, 1, 1]), dtype='float32'
                )
                positive = paddle.to_tensor(
                    np.reshape(array1, [0]), dtype='float32'
                )
                array = np.array([1, 2, 3, 4], dtype=np.float32)
                labels = paddle.to_tensor(
                    np.reshape(array, [4]), dtype='float32'
                )
                paddle.nn.functional.npair_loss(anchor, positive, labels)

            self.assertRaises(ValueError, test_anchor_0_size)
            self.assertRaises(ValueError, test_positive_0_size)


if __name__ == '__main__':
    unittest.main()
