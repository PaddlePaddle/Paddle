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
import paddle.fluid.core as core

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
            dtype=np.float64,
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
            "logits": self.acts,
            "label": self.labels,
            "logits_length": self.logit_lens,
            "labels_length": self.label_lens,
        }
        self.outputs = {"loss": self.loss}
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
        self.outputs["warprnntgrad"] = self.gradient
        self.check_grad_with_place(
            paddle.CPUPlace(),
            ["logits"],
            "loss",
            max_relative_error=0.005,
            check_dygraph=False,
            check_eager=True,
        )


class TestWarpRNNTCPUFP64Op(TestWarpRNNTCPUOp):
    def test_check_output(self):
        self.acts.astype(np.float64)
        self.check_output_with_place(
            paddle.CPUPlace(), check_eager=True, check_dygraph=False
        )

    def test_check_grad(self):
        self.acts.astype(np.float64)
        self.outputs["warprnntgrad"] = self.gradient
        self.check_grad_with_place(
            paddle.CPUPlace(),
            ["logits"],
            "loss",
            max_relative_error=0.005,
            check_dygraph=False,
            check_eager=True,
        )


class TestWarpRNNTGPUOp(TestWarpRNNTCPUOp):
    def set_act(self):
        # logits
        self.acts = np.array(
            [
                [
                    [
                        [
                            0.06535690384862791,
                            0.7875301411923206,
                            0.08159176605666074,
                        ],
                        [
                            0.5297155426466327,
                            0.7506749639230854,
                            0.7541348379087998,
                        ],
                        [
                            0.6097641124736383,
                            0.8681404965673826,
                            0.6225318186056529,
                        ],
                    ],
                    [
                        [
                            0.6685222872103057,
                            0.8580392805336061,
                            0.16453892311765583,
                        ],
                        [
                            0.989779515236694,
                            0.944298460961015,
                            0.6031678586829663,
                        ],
                        [
                            0.9467833543605416,
                            0.666202507295747,
                            0.28688179752461884,
                        ],
                    ],
                    [
                        [
                            0.09418426230195986,
                            0.3666735970751962,
                            0.736168049462793,
                        ],
                        [
                            0.1666804425271342,
                            0.7141542198635192,
                            0.3993997272216727,
                        ],
                        [
                            0.5359823524146038,
                            0.29182076440286386,
                            0.6126422611507932,
                        ],
                    ],
                    [
                        [
                            0.3242405528768486,
                            0.8007644367291621,
                            0.5241057606558068,
                        ],
                        [
                            0.779194617063042,
                            0.18331417220174862,
                            0.113745182072432,
                        ],
                        [
                            0.24022162381327106,
                            0.3394695622533106,
                            0.1341595066017014,
                        ],
                    ],
                ],
                [
                    [
                        [
                            0.5055615569388828,
                            0.051597282072282646,
                            0.6402903936686337,
                        ],
                        [
                            0.43073311517251,
                            0.8294731834714112,
                            0.1774668847323424,
                        ],
                        [
                            0.3207001991262245,
                            0.04288308912457006,
                            0.30280282975568984,
                        ],
                    ],
                    [
                        [
                            0.6751777088333762,
                            0.569537369330242,
                            0.5584738347504452,
                        ],
                        [
                            0.08313242153985256,
                            0.06016544344162322,
                            0.10795752845152584,
                        ],
                        [
                            0.7486153608562472,
                            0.943918041459349,
                            0.4863558118797222,
                        ],
                    ],
                    [
                        [
                            0.4181986264486809,
                            0.6524078485043804,
                            0.024242983423721887,
                        ],
                        [
                            0.13458171554507403,
                            0.3663418070512402,
                            0.2958297395361563,
                        ],
                        [
                            0.9236695822497084,
                            0.6899291482654177,
                            0.7418981733448822,
                        ],
                    ],
                    [
                        [
                            0.25000547599982104,
                            0.6034295486281007,
                            0.9872887878887768,
                        ],
                        [
                            0.5926057265215715,
                            0.8846724004467684,
                            0.5434495396894328,
                        ],
                        [
                            0.6607698886038497,
                            0.3771277082495921,
                            0.3580209022231813,
                        ],
                    ],
                ],
                [
                    [
                        [
                            0.5055615569388828,
                            0.051597282072282646,
                            0.6402903936686337,
                        ],
                        [
                            0.43073311517251,
                            0.8294731834714112,
                            0.1774668847323424,
                        ],
                        [
                            0.3207001991262245,
                            0.04288308912457006,
                            0.30280282975568984,
                        ],
                    ],
                    [
                        [
                            0.6751777088333762,
                            0.569537369330242,
                            0.5584738347504452,
                        ],
                        [
                            0.08313242153985256,
                            0.06016544344162322,
                            0.10795752845152584,
                        ],
                        [
                            0.7486153608562472,
                            0.943918041459349,
                            0.4863558118797222,
                        ],
                    ],
                    [
                        [
                            0.4181986264486809,
                            0.6524078485043804,
                            0.024242983423721887,
                        ],
                        [
                            0.13458171554507403,
                            0.3663418070512402,
                            0.2958297395361563,
                        ],
                        [
                            0.9236695822497084,
                            0.6899291482654177,
                            0.7418981733448822,
                        ],
                    ],
                    [
                        [
                            0.25000547599982104,
                            0.6034295486281007,
                            0.9872887878887768,
                        ],
                        [
                            0.5926057265215715,
                            0.8846724004467684,
                            0.5434495396894328,
                        ],
                        [
                            0.6607698886038497,
                            0.3771277082495921,
                            0.3580209022231813,
                        ],
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
                        [-1.86843902e-01, -6.25548810e-02, 2.49398798e-01],
                        [-2.03376666e-01, 2.02399328e-01, 9.77333169e-04],
                        [-1.41016081e-01, 7.91234672e-02, 6.18926100e-02],
                    ],
                    [
                        [-1.15517676e-02, -8.12802389e-02, 9.28319991e-02],
                        [-1.54257029e-01, 2.29432687e-01, -7.51756504e-02],
                        [-2.46593088e-01, 1.46404594e-01, 1.00188486e-01],
                    ],
                    [
                        [-1.29182907e-02, -6.15932420e-02, 7.45115355e-02],
                        [-5.59857301e-02, 2.19830811e-01, -1.63845062e-01],
                        [-4.97626871e-01, 2.09239945e-01, 2.88386941e-01],
                    ],
                    [
                        [1.36048580e-02, -3.02196294e-02, 1.66147724e-02],
                        [1.13924511e-01, 6.27811998e-02, -1.76705718e-01],
                        [-6.67078257e-01, 3.67658824e-01, 2.99419403e-01],
                    ],
                ],
                [
                    [
                        [-3.56343776e-01, -5.53474613e-02, 4.11691219e-01],
                        [-9.69219357e-02, 2.94591039e-02, 6.74628317e-02],
                        [-6.35175705e-02, 2.76544970e-02, 3.58630717e-02],
                    ],
                    [
                        [-1.54499024e-01, -7.39420280e-02, 2.28441030e-01],
                        [-1.66789949e-01, -8.78955179e-05, 1.66877866e-01],
                        [-1.72369644e-01, 1.05565332e-01, 6.68043196e-02],
                    ],
                    [
                        [2.38748826e-02, -1.18255816e-01, 9.43809375e-02],
                        [-1.04707085e-01, -1.08934477e-01, 2.13641584e-01],
                        [-3.69844258e-01, 1.80118099e-01, 1.89726159e-01],
                    ],
                    [
                        [2.57137045e-02, -7.94617534e-02, 5.37480488e-02],
                        [1.22328237e-01, -2.38788679e-01, 1.16460443e-01],
                        [-5.98686993e-01, 3.02203178e-01, 2.96483815e-01],
                    ],
                ],
                [
                    [
                        [-3.56343776e-01, -5.53474613e-02, 4.11691219e-01],
                        [-9.69219357e-02, 2.94591039e-02, 6.74628317e-02],
                        [-6.35175705e-02, 2.76544970e-02, 3.58630717e-02],
                    ],
                    [
                        [-1.54499024e-01, -7.39420280e-02, 2.28441030e-01],
                        [-1.66789949e-01, -8.78955179e-05, 1.66877866e-01],
                        [-1.72369644e-01, 1.05565332e-01, 6.68043196e-02],
                    ],
                    [
                        [2.38748826e-02, -1.18255816e-01, 9.43809375e-02],
                        [-1.04707085e-01, -1.08934477e-01, 2.13641584e-01],
                        [-3.69844258e-01, 1.80118099e-01, 1.89726159e-01],
                    ],
                    [
                        [2.57137045e-02, -7.94617534e-02, 5.37480488e-02],
                        [1.22328237e-01, -2.38788679e-01, 1.16460443e-01],
                        [-5.98686993e-01, 3.02203178e-01, 2.96483815e-01],
                    ],
                ],
            ],
            dtype=np.float64,
        )

    def test_check_output(self):
        self.check_output_with_place(
            paddle.CUDAPlace(0), check_eager=True, check_dygraph=False
        )

    def test_check_grad(self):
        self.outputs["warprnntgrad"] = self.gradient
        if core.is_compiled_with_rocm():
            self.check_grad_with_place(
                paddle.CUDAPlace(0),
                ["logits"],
                "loss",
                max_relative_error=0.008,
                check_dygraph=False,
                check_eager=True,
            )
        else:
            self.check_grad_with_place(
                paddle.CUDAPlace(0),
                ["logits"],
                "loss",
                max_relative_error=0.008,
                check_dygraph=False,
                check_eager=True,
            )


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
