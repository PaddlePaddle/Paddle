#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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


class TestWarpRNNTCPUOp(OpTest):
    def config(self):
        self.blank = 0
        self.fastemit_lambda = 0.0
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
            dtype=np.float64,
        )
        self.labels = np.array([[1, 2], [1, 1], [1, 1]], dtype=np.int32)
        self.logit_lens = np.array([4, 4, 4], dtype=np.int32)
        self.label_lens = np.array([2, 2, 2], dtype=np.int32)

        self.loss = np.array(
            [[4.2806528590890736], [3.9384369822503591], [3.9384369822503591]],
            dtype=np.float64,
        )
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
            dtype=np.float64,
        )

    def setUp(self):
        self.op_type = "warprnnt"
        self.config()
        self.python_api = python_api
        self.python_out_sig = ["loss"]

        self.inputs = {
            "logits": self.acts,
            "label": self.labels,
            "logits_length": self.logit_lens,
            "labels_length": self.label_lens,
        }
        self.outputs = {"loss": self.loss, "warprnntgrad": self.gradient}
        self.attrs = {
            "blank": self.blank,
            "fastemit_lambda": self.fastemit_lambda,
            "num_threads": 1,
        }

    def test_check_output(self):
        self.check_output_with_place(
            paddle.CPUPlace(), check_eager=True, check_dygraph=False
        )

    def test_check_grad(self):
        self.check_grad_with_place(
            paddle.CPUPlace(),
            ["logits"],
            "loss",
            max_relative_error=1e-7,
            check_dygraph=False,
            check_eager=True,
        )


# class TestWarpRNNTGPUOp(TestWarpRNNTCPUOp):
#     def test_check_output(self):
#         self.check_output_with_place(paddle.CUDAPlace(0), check_eager=True, check_dygraph=False)

#     def test_check_grad(self):
#         if core.is_compiled_with_rocm():
#             self.check_grad_with_place(paddle.CUDAPlace(0),
#                 ["Logits"],
#                 "Loss",
#                 max_relative_error=0.009,
#                 check_dygraph=False,
#                 check_eager=True
#             )
#         else:
#             self.check_grad_with_place(paddle.CUDAPlace(0),
#                 ["Logits"],
#                 "Loss",
#                 max_relative_error=0.007,
#                 check_dygraph=False,
#                 check_eager=True
#             )


# class TestWarpCTCOpFp64(OpTest):
#     def config(self):
#         self.blank = 0
#         self.fastemit_lambda = 0.0
#         self.reduction = 'mean'
#         self.acts = np.array([[[[0.1, 0.6, 0.1, 0.1, 0.1],
#                   [0.1, 0.1, 0.6, 0.1, 0.1],
#                   [0.1, 0.1, 0.2, 0.8, 0.1]],
#                  [[0.1, 0.6, 0.1, 0.1, 0.1],
#                   [0.1, 0.1, 0.2, 0.1, 0.1],
#                   [0.7, 0.1, 0.2, 0.1, 0.1]]]], dtype=np.float64)
#         self.labels = [[1, 2]]
#         self.logit_lens = [2]
#         self.label_lens = [2]

#     def setUp(self):
#         self.op_type = "warprnnt"
#         self.python_api = python_api
#         self.python_out_sig = ["Loss"]
#         self.config()

#         self.loss = 4.49566677
#         self.gradient = np.array([[[[-0.13116688, -0.3999269 ,  0.17703125,  0.17703125,
#                             0.17703125],
#                             [-0.18572757,  0.12247056, -0.18168412,  0.12247056,
#                             0.12247056],
#                             [-0.32091254,  0.06269141,  0.06928472,  0.12624499,
#                             0.06269141]],

#                             [[ 0.05456069, -0.21824276,  0.05456069,  0.05456069,
#                             0.05456069],
#                             [ 0.12073959,  0.12073959, -0.48295835,  0.12073959,
#                             0.12073959],
#                             [-0.6925882 ,  0.16871116,  0.18645467,  0.16871116,
#                             0.16871116]]]], dtype=self.act.dtype)

#         self.inputs = {
#             "Logits": self.acts,
#             "Label": self.Labels,
#             "LogitsLength": self.logit_lens,
#             "LabelLength": self.label_lens,
#         }
#         self.outputs = {"Loss": loss}
#         self.attrs = {
#             "blank": self.blank,
#             "fastemit_lambda": self.fastemit_lambda,
#             'num_threads': 1,
#         }

#     def test_check_output(self):
#         self.check_output(check_eager=True)

#     def test_check_grad(self):
#         self.outputs['WarpCTCGrad'] = self.gradient
#         self.check_grad(["Logits"], "Loss", check_eager=True)


# class TestWarpCTCOpError(unittest.TestCase):
#     def test_errors(self):
#         with program_guard(Program(), Program()):
#             logits = fluid.data(
#                 name='logits', shape=[5, 16, 6], dtype='float32'
#             )
#             logits_length = fluid.data(
#                 name='logits_length', shape=[None], dtype='int64'
#             )
#             label = fluid.data(name='label', shape=[16, 3], dtype='int32')
#             label_length = fluid.data(
#                 name='labels_length', shape=[None], dtype='int64'
#             )

#             def test_logits_Variable():
#                 logits_data = np.random.rand(5, 16, 6).astype(logits.dtype)
#                 fluid.layers.warprnnt(
#                     input=logits_data,
#                     label=label,
#                     input_length=logits_length,
#                     label_length=label_length,
#                 )

#             self.assertRaises(TypeError, test_logits_Variable)

#             def test_label_Variable():
#                 label_data = np.random.randint(0, 5, [5, 1]).astype("int32")
#                 fluid.layers.warprnnt(
#                     input=logits,
#                     label=label_data,
#                     input_length=logits_length,
#                     label_length=label_length,
#                 )

#             self.assertRaises(TypeError, test_label_Variable)

#             def test_logits_len_Variable():
#                 logits_length_data = np.array([5] * 16).astype("int64")
#                 fluid.layers.warprnnt(
#                     input=logits,
#                     label=label,
#                     input_length=logits_length_data,
#                     label_length=label_length,
#                 )

#             self.assertRaises(TypeError, test_logits_len_Variable)

#             def test_label_len_Variable():
#                 label_length_data = np.array([3] * 16).astype("int64")
#                 fluid.layers.warprnnt(
#                     input=logits,
#                     label=label,
#                     input_length=logits_length,
#                     label_length=label_length_data,
#                 )

#             self.assertRaises(TypeError, test_label_len_Variable)

#     def test_dygraph_errors(self):
#         def test_dygraph_with_lod():

#             logits = np.random.uniform(0.1, 1.0, [20, 15]).astype("float32")
#             # labels should not be blank
#             labels = np.random.randint(0, 15 - 1, [15, 1], dtype="int32")
#             softmax = paddle.to_tensor(logits)
#             labels = paddle.to_tensor(labels)

#             fluid.layers.warprnnt(input=softmax, label=labels)

#         paddle.disable_static()
#         self.assertRaises(ValueError, test_dygraph_with_lod)
#         paddle.enable_static()


# class TestCTCLossAPICase(unittest.TestCase):
#     def test_functinal_api(self):
#         self.batch_size = 4
#         self.num_classes = CUDA_BLOCK_SIZE + 2
#         self.logits_length = np.array([4, 1, 3, 3], dtype=np.int64)
#         self.labels_length = np.array([3, 1, 4, 4], dtype=np.int64)
#         self.blank = self.num_classes - 1
#         self.norm_by_times = False

#         logits = np.random.uniform(
#             0.1,
#             1.0,
#             [max(self.logits_length), self.batch_size, self.num_classes],
#         ).astype("float32")
#         softmax = np.apply_along_axis(stable_softmax, -1, logits)
#         # labels should not be blank
#         labels = np.random.randint(
#             0,
#             self.num_classes - 1,
#             [self.batch_size, max(self.labels_length)],
#             dtype="int32",
#         )

#         ctc = CTCForward(
#             softmax,
#             self.logits_length,
#             labels,
#             self.labels_length,
#             self.num_classes,
#             self.batch_size,
#             self.blank,
#             self.norm_by_times,
#         )
#         loss_np = ctc.forward()

#         paddle.disable_static()
#         softmax = paddle.to_tensor(logits)
#         labels = paddle.to_tensor(labels)
#         logits_length = paddle.to_tensor(self.logits_length)
#         labels_length = paddle.to_tensor(self.labels_length)
#         loss_pd_mean = F.ctc_loss(
#             softmax,
#             labels,
#             logits_length,
#             labels_length,
#             blank=self.blank,
#             reduction='mean',
#         )
#         loss_pd_mean = loss_pd_mean.numpy()

#         loss_pd_sum = F.ctc_loss(
#             softmax,
#             labels,
#             logits_length,
#             labels_length,
#             blank=self.blank,
#             reduction='sum',
#         )
#         loss_pd_sum = loss_pd_sum.numpy()
#         paddle.enable_static()
#         loss_np = np.squeeze(loss_np, axis=-1)
#         loss_np_mean = (loss_np / labels_length.numpy()).mean()
#         loss_np_sum = loss_np.sum()

#         np.testing.assert_allclose(
#             loss_pd_mean, loss_np_mean, rtol=1e-05, atol=1
#         )
#         np.testing.assert_allclose(loss_pd_sum, loss_np_sum, rtol=1e-05, atol=1)

#     def test_class_api(self):
#         self.batch_size = 3
#         self.num_classes = 15
#         self.logits_length = np.array([3, 3, 3], dtype=np.int64)
#         self.labels_length = np.array([0, 1, 2], dtype=np.int64)
#         self.blank = 0
#         self.norm_by_times = False

#         logits = np.random.uniform(
#             0.1,
#             1.0,
#             [max(self.logits_length), self.batch_size, self.num_classes],
#         ).astype("float32")
#         softmax = np.apply_along_axis(stable_softmax, -1, logits)
#         # labels should not be blank
#         labels = np.random.randint(
#             1,
#             self.num_classes,
#             [self.batch_size, max(self.labels_length)],
#             dtype="int32",
#         )

#         ctc = CTCForward(
#             softmax,
#             self.logits_length,
#             labels,
#             self.labels_length,
#             self.num_classes,
#             self.batch_size,
#             self.blank,
#             self.norm_by_times,
#         )
#         loss_np = ctc.forward()

#         paddle.disable_static()
#         softmax = paddle.to_tensor(logits)
#         labels = paddle.to_tensor(labels)
#         logits_length = paddle.to_tensor(self.logits_length)
#         labels_length = paddle.to_tensor(self.labels_length)

#         loss_pd = paddle.nn.CTCLoss(self.blank, 'none')(
#             softmax, labels, logits_length, labels_length
#         )
#         loss_pd = loss_pd.numpy()
#         paddle.enable_static()
#         loss_np = np.squeeze(loss_np, axis=-1)

#         np.testing.assert_allclose(loss_pd, loss_np, rtol=1e-05, atol=1)


if __name__ == "__main__":
    unittest.main()
