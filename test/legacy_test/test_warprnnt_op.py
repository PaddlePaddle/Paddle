#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest

import paddle
from paddle import _C_ops

paddle.enable_static()


def python_api(
    logits,
    label,
    logits_length,
    labels_length,
    blank=0,
    fastemit_lambda=0.0,
    num_threads=1,
):
    loss_out = _C_ops.warprnnt(
        logits,
        label,
        logits_length,
        labels_length,
        blank,
        fastemit_lambda,
        num_threads,
    )
    return loss_out


class TestWarpRNNTOp(OpTest):
    def set_act(self):
        # logsoftmax
        self.acts = np.array(
            [
                [
                    [
                        [-1.40493705, -0.68276381, -1.38870219],
                        [-1.25243963, -1.03148021, -1.02802034],
                        [-1.19624572, -0.93786934, -1.18347801],
                    ],
                    [
                        [-1.03417513, -0.84465814, -1.53815849],
                        [-0.96884241, -1.01432347, -1.35545407],
                        [-0.82076925, -1.10135010, -1.48067081],
                    ],
                    [
                        [-1.43828803, -1.16579869, -0.79630424],
                        [-1.38401855, -0.83654478, -1.15129927],
                        [-1.05188255, -1.29604414, -0.97522265],
                    ],
                    [
                        [-1.34330978, -0.86678589, -1.14344457],
                        [-0.72518815, -1.32106859, -1.39063758],
                        [-1.09984781, -1.00059987, -1.20590993],
                    ],
                ],
                [
                    [
                        [-1.02221057, -1.47617485, -0.88748174],
                        [-1.18362952, -0.78488945, -1.43689575],
                        [-1.00784739, -1.28566450, -1.02574476],
                    ],
                    [
                        [-1.02589709, -1.13153743, -1.14260096],
                        [-1.09942215, -1.12238913, -1.07459704],
                        [-1.09359647, -0.89829379, -1.35585602],
                    ],
                    [
                        [-1.07782876, -0.84361953, -1.47178440],
                        [-1.23424792, -1.00248783, -1.07299990],
                        [-0.96521771, -1.19895815, -1.14698912],
                    ],
                    [
                        [-1.50722446, -1.15380039, -0.76994115],
                        [-1.19125975, -0.89919308, -1.24041594],
                        [-0.91301359, -1.19665577, -1.21576258],
                    ],
                ],
                [
                    [
                        [-1.02221057, -1.47617485, -0.88748174],
                        [-1.18362952, -0.78488945, -1.43689575],
                        [-1.00784739, -1.28566450, -1.02574476],
                    ],
                    [
                        [-1.02589709, -1.13153743, -1.14260096],
                        [-1.09942215, -1.12238913, -1.07459704],
                        [-1.09359647, -0.89829379, -1.35585602],
                    ],
                    [
                        [-1.07782876, -0.84361953, -1.47178440],
                        [-1.23424792, -1.00248783, -1.07299990],
                        [-0.96521771, -1.19895815, -1.14698912],
                    ],
                    [
                        [-1.50722446, -1.15380039, -0.76994115],
                        [-1.19125975, -0.89919308, -1.24041594],
                        [-0.91301359, -1.19665577, -1.21576258],
                    ],
                ],
            ],
            dtype=np.float32,
        )

    def set_gradient(self):
        self.gradient = np.array(
            [
                [
                    [
                        [-0.43222645, -0.56777355, 0.0],
                        [-0.3656501, 0.0, -0.20212345],
                        [-0.20212345, 0.0, 0.0],
                    ],
                    [
                        [-0.16521672, -0.26700973, 0.0],
                        [-0.39436539, 0.0, -0.23829444],
                        [-0.44041789, 0.0, 0.0],
                    ],
                    [
                        [-0.05212979, -0.11308693, 0.0],
                        [-0.18313787, 0.0, -0.32431445],
                        [-0.76473234, 0.0, 0.0],
                    ],
                    [
                        [0.0, -0.05212979, 0.0],
                        [0.0, 0.0, -0.23526766],
                        [-1.0, 0.0, 0.0],
                    ],
                ],
                [
                    [
                        [-0.71614241, -0.28385759, 0.0],
                        [-0.18382932, -0.10002826, 0.0],
                        [-0.10002826, 0.0, 0.0],
                    ],
                    [
                        [-0.41121795, -0.30492447, 0.0],
                        [-0.32957594, -0.15917785, 0.0],
                        [-0.25920611, 0.0, 0.0],
                    ],
                    [
                        [-0.11607642, -0.29514153, 0.0],
                        [-0.28653336, -0.3381841, 0.0],
                        [-0.59739022, 0.0, 0.0],
                    ],
                    [
                        [0.0, -0.11607642, 0.0],
                        [0.0, -0.40260978, 0.0],
                        [-1.0, 0.0, 0.0],
                    ],
                ],
                [
                    [
                        [-0.71614241, -0.28385759, 0.0],
                        [-0.18382932, -0.10002826, 0.0],
                        [-0.10002826, 0.0, 0.0],
                    ],
                    [
                        [-0.41121795, -0.30492447, 0.0],
                        [-0.32957594, -0.15917785, 0.0],
                        [-0.25920611, 0.0, 0.0],
                    ],
                    [
                        [-0.11607642, -0.29514153, 0.0],
                        [-0.28653336, -0.3381841, 0.0],
                        [-0.59739022, 0.0, 0.0],
                    ],
                    [
                        [0.0, -0.11607642, 0.0],
                        [0.0, -0.40260978, 0.0],
                        [-1.0, 0.0, 0.0],
                    ],
                ],
            ],
            dtype=np.float32,
        )

    def config(self):
        self.blank = 0
        self.fastemit_lambda = 0.0
        self.set_act()
        self.labels = np.array([[1, 2], [1, 1], [1, 1]], dtype=np.int32)
        self.logit_lens = np.array([4, 4, 4], dtype=np.int32)
        self.label_lens = np.array([2, 2, 2], dtype=np.int32)

        self.loss = np.array(
            [4.2806528590890736, 3.9384369822503591, 3.9384369822503591],
            dtype=np.float64,
        )
        self.set_gradient()

    def setUp(self):
        self.op_type = "warprnnt"
        self.config()
        self.python_api = python_api
        self.python_out_sig = ["loss"]

        self.inputs = {
            "input": self.acts,
            "label": self.labels,
            "input_lengths": self.logit_lens,
            "label_lengths": self.label_lens,
        }
        self.outputs = {"loss": self.loss}
        self.attrs = {
            "blank": self.blank,
            "fastemit_lambda": self.fastemit_lambda,
            "num_threads": 1,
        }

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.outputs["warprnntgrad"] = self.gradient
        self.check_grad(
            ["input"], "loss", numeric_grad_delta=0.009, check_pir=True
        )


class TestWarpRNNTFP64Op(TestWarpRNNTOp):
    def test_check_output(self):
        self.acts.astype(np.float64)
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.acts.astype(np.float64)
        self.outputs["warprnntgrad"] = self.gradient
        self.check_grad(
            ["input"], "loss", numeric_grad_delta=0.009, check_pir=True
        )


class TestWarpRNNTOpError(unittest.TestCase):

    def test_errors1(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            logits_length = paddle.static.data(
                name='logit_lengths', shape=[None], dtype='int32'
            )
            label = paddle.static.data(
                name='labels', shape=[16, 3], dtype='int32'
            )
            label_length = paddle.static.data(
                name='label_lengths', shape=[None], dtype='int32'
            )

            def test_logits_Variable():
                logits_data = paddle.static.data(
                    name='logits_data', shape=[5, 16, 6], dtype='int32'
                )
                paddle.nn.functional.rnnt_loss(
                    input=logits_data,
                    label=label,
                    input_lengths=logits_length,
                    label_lengths=label_length,
                )

            self.assertRaises(TypeError, test_logits_Variable)

    def test_errors2(self):
        with paddle.pir_utils.OldIrGuard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                logits = paddle.static.data(
                    name='input', shape=[5, 16, 6], dtype='float32'
                )
                logits_length = paddle.static.data(
                    name='logit_lengths', shape=[None], dtype='int32'
                )
                label = paddle.static.data(
                    name='labels', shape=[16, 3], dtype='int32'
                )
                label_length = paddle.static.data(
                    name='label_lengths', shape=[None], dtype='int32'
                )

                def test_label_Variable():
                    label_data = paddle.static.data(
                        name='label_data', shape=[16, 3], dtype='int64'
                    )
                    paddle.nn.functional.rnnt_loss(
                        input=logits,
                        label=label_data,
                        input_lengths=logits_length,
                        label_lengths=label_length,
                    )

                self.assertRaises(TypeError, test_label_Variable)

                def test_logits_len_Variable():
                    logits_length_data = paddle.static.data(
                        name='logits_length_data', shape=[None], dtype='int64'
                    )
                    paddle.nn.functional.rnnt_loss(
                        input=logits,
                        label=label,
                        input_lengths=logits_length_data,
                        label_lengths=label_length,
                    )

                self.assertRaises(TypeError, test_logits_len_Variable)

                def test_label_len_Variable():
                    label_length_data = paddle.static.data(
                        name='label_length_data', shape=[None], dtype='int64'
                    )
                    paddle.nn.functional.rnnt_loss(
                        input=logits,
                        label=label,
                        input_lengths=logits_length,
                        label_lengths=label_length_data,
                    )

                self.assertRaises(TypeError, test_label_len_Variable)

    def test_dygraph_errors(self):
        def test_dygraph_with_lod():
            print("test_dygraph_with_lod")
            logits = np.random.uniform(0.1, 1.0, [20, 15]).astype("float32")
            # labels should not be blank
            labels = np.random.randint(0, 15 - 1, [15, 1], dtype="int32")
            labels_len = np.random.randint(0, 15 - 1, [15, 1], dtype="int64")
            logits_len = np.random.randint(0, 15 - 1, [15, 1], dtype="int32")

            softmax = paddle.to_tensor(logits)
            labels = paddle.to_tensor(labels)
            logits_len = paddle.to_tensor(logits_len)
            labels_len = paddle.to_tensor(labels_len)

            paddle.nn.functional.rnnt_loss(
                input=softmax,
                label=labels,
                input_lengths=logits_len,
                label_lengths=labels_len,
            )

        paddle.disable_static()
        self.assertRaises(ValueError, test_dygraph_with_lod)
        paddle.enable_static()


class TestRNNTLossAPICase(unittest.TestCase):
    def set_act(self):
        # logsoftmax
        self.acts = np.array(
            [
                [
                    [
                        [-1.40493705, -0.68276381, -1.38870219],
                        [-1.25243963, -1.03148021, -1.02802034],
                        [-1.19624572, -0.93786934, -1.18347801],
                    ],
                    [
                        [-1.03417513, -0.84465814, -1.53815849],
                        [-0.96884241, -1.01432347, -1.35545407],
                        [-0.82076925, -1.10135010, -1.48067081],
                    ],
                    [
                        [-1.43828803, -1.16579869, -0.79630424],
                        [-1.38401855, -0.83654478, -1.15129927],
                        [-1.05188255, -1.29604414, -0.97522265],
                    ],
                    [
                        [-1.34330978, -0.86678589, -1.14344457],
                        [-0.72518815, -1.32106859, -1.39063758],
                        [-1.09984781, -1.00059987, -1.20590993],
                    ],
                ],
                [
                    [
                        [-1.02221057, -1.47617485, -0.88748174],
                        [-1.18362952, -0.78488945, -1.43689575],
                        [-1.00784739, -1.28566450, -1.02574476],
                    ],
                    [
                        [-1.02589709, -1.13153743, -1.14260096],
                        [-1.09942215, -1.12238913, -1.07459704],
                        [-1.09359647, -0.89829379, -1.35585602],
                    ],
                    [
                        [-1.07782876, -0.84361953, -1.47178440],
                        [-1.23424792, -1.00248783, -1.07299990],
                        [-0.96521771, -1.19895815, -1.14698912],
                    ],
                    [
                        [-1.50722446, -1.15380039, -0.76994115],
                        [-1.19125975, -0.89919308, -1.24041594],
                        [-0.91301359, -1.19665577, -1.21576258],
                    ],
                ],
                [
                    [
                        [-1.02221057, -1.47617485, -0.88748174],
                        [-1.18362952, -0.78488945, -1.43689575],
                        [-1.00784739, -1.28566450, -1.02574476],
                    ],
                    [
                        [-1.02589709, -1.13153743, -1.14260096],
                        [-1.09942215, -1.12238913, -1.07459704],
                        [-1.09359647, -0.89829379, -1.35585602],
                    ],
                    [
                        [-1.07782876, -0.84361953, -1.47178440],
                        [-1.23424792, -1.00248783, -1.07299990],
                        [-0.96521771, -1.19895815, -1.14698912],
                    ],
                    [
                        [-1.50722446, -1.15380039, -0.76994115],
                        [-1.19125975, -0.89919308, -1.24041594],
                        [-0.91301359, -1.19665577, -1.21576258],
                    ],
                ],
            ],
            dtype=np.float32,
        )

    def config(self):
        self.blank = 0
        self.fastemit_lambda = 0.0
        self.set_act()
        self.labels = np.array([[1, 2], [1, 1], [1, 1]], dtype=np.int32)
        self.logit_lens = np.array([4, 4, 4], dtype=np.int32)
        self.label_lens = np.array([2, 2, 2], dtype=np.int32)

        self.loss = np.array(
            [4.2806528590890736, 3.9384369822503591, 3.9384369822503591],
            dtype=np.float64,
        )

    def test_functinal_api(self):
        self.config()

        paddle.disable_static()

        acts = paddle.to_tensor(self.acts)
        labels = paddle.to_tensor(self.labels)
        logit_lens = paddle.to_tensor(self.logit_lens)
        label_lens = paddle.to_tensor(self.label_lens)

        loss_pd_mean = paddle.nn.functional.rnnt_loss(
            acts,
            labels,
            logit_lens,
            label_lens,
            blank=self.blank,
            reduction='mean',
            fastemit_lambda=self.fastemit_lambda,
        )
        loss_pd_mean = loss_pd_mean.numpy()

        loss_pd_sum = paddle.nn.functional.rnnt_loss(
            acts,
            labels,
            logit_lens,
            label_lens,
            blank=self.blank,
            reduction='sum',
            fastemit_lambda=self.fastemit_lambda,
        )
        loss_pd_sum = loss_pd_sum.numpy()

        paddle.enable_static()
        B = self.loss.shape[0]
        loss_np_mean = self.loss.sum() / B
        loss_np_sum = self.loss.sum()

        np.testing.assert_allclose(
            loss_pd_mean, loss_np_mean, rtol=1e-05, atol=1
        )
        np.testing.assert_allclose(loss_pd_sum, loss_np_sum, rtol=1e-05, atol=1)

    def test_class_api(self):
        self.config()

        paddle.disable_static()

        acts = paddle.to_tensor(self.acts)
        labels = paddle.to_tensor(self.labels)
        logit_lens = paddle.to_tensor(self.logit_lens)
        label_lens = paddle.to_tensor(self.label_lens)

        loss_pd = paddle.nn.RNNTLoss(self.blank, self.fastemit_lambda, 'none')(
            acts, labels, logit_lens, label_lens
        )
        loss_pd = loss_pd.numpy()
        paddle.enable_static()
        np.testing.assert_allclose(loss_pd, self.loss, rtol=1e-05, atol=1)


if __name__ == "__main__":
    unittest.main()
