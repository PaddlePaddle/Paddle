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

from __future__ import print_function

import sys

sys.path.append("..")
import sys
import unittest
import numpy as np
from op_test import OpTest
from test_softmax_op import stable_softmax
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard
import paddle
import paddle.nn.functional as F

paddle.enable_static()

MLU_CLUSTER_SIZE = 8


class CTCM(object):

    def __init__(self, L, S, T, mb, alphabet_size, blank_label, labels):

        self.s_counter = 0
        self.e_counter = 0
        self.repeats = 0
        self.L = L
        self.S = S
        self.T = T
        self.mb = mb
        self.alphabet_size = alphabet_size
        self.blank_label = blank_label
        self.labels = labels
        self.alphas = np.inf * np.ones((self.S * self.T), dtype=float) * -1
        self.betas = np.inf * np.ones((self.S), dtype=np.int32) * -1
        self.output = np.inf * np.ones((self.alphabet_size), dtype=float) * -1
        self.labels_w_blanks = np.zeros((self.S), dtype=np.int32)
        self.e_inc = np.zeros((self.S), dtype=np.int32)
        self.s_inc = np.zeros((self.S), dtype=np.int32)

    def setup_labels(self):

        self.s_inc[self.s_counter] = 1
        self.s_counter += 1
        for i in range(1, self.L):
            if (self.labels[i - 1] == self.labels[i]):
                self.s_inc[self.s_counter] = 1
                self.s_counter += 1
                self.s_inc[self.s_counter] = 1
                self.s_counter += 1
                self.e_inc[self.e_counter] = 1
                self.e_counter += 1
                self.e_inc[self.e_counter] = 1
                self.e_counter += 1
                self.repeats += 1
            else:

                self.s_inc[self.s_counter] = 2
                self.s_counter += 1
                self.e_inc[self.e_counter] = 2
                self.e_counter += 1

        self.e_inc[self.e_counter] = 1
        self.e_counter += 1

        for i in range(self.L):
            self.labels_w_blanks[2 * i] = self.blank_label
            self.labels_w_blanks[2 * i + 1] = self.labels[i]
        self.labels_w_blanks[self.S - 1] = self.blank_label

        return self.repeats, self.s_inc, self.e_inc, self.labels_w_blanks, self.alphas, self.betas, self.output


class CTCForward(object):

    def __init__(self, softmax, input_lengths, labels, labels_lengths,
                 num_classes, batch_size, blank):
        self.softmax = softmax
        self.input_lengths = input_lengths
        self.multi_label = True if (len(labels.shape) > 1) else False
        self.labels = labels.flatten()
        self.labels_lengths = labels_lengths
        self.blank = blank
        self.alphas = np
        self.level = 0
        self.offset = 0
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.loss = np.zeros([self.batch_size, 1], dtype=softmax.dtype)
        self.grads = np.zeros(self.softmax.shape, dtype=softmax.dtype)

        # float64
        self.EXP_MAX = sys.float_info.max
        self.EXP_MIN = sys.float_info.min
        self.LOG_ZERO = np.log(self.EXP_MIN)
        self.LOG_INFINITY = np.log(self.EXP_MAX)

    def safe_exp(self, x):
        if x <= self.LOG_ZERO:
            return 0.0
        if x >= self.LOG_INFINITY:
            return self.EXP_MAX
        return np.exp(x)

    def safe_log(self, x):
        if x <= self.EXP_MIN:
            return self.LOG_ZERO
        return np.log(x)

    # x = lna and y = lnb are in log scale,
    # ln(a + b) = lna + ln(1 + exp(lnb - lna)), where b > a
    def log_add(self, x, y):
        if (x == -np.inf):
            return y
        if (y == -np.inf):
            return x
        if x < y:
            t = y
            y = x
            x = t
        return x + self.safe_log(1 + self.safe_exp(y - x))

    def compute_alphas(self, probs, repeats, S, T, e_inc, s_inc, labels,
                       alphas):

        start = 0 if ((S / 2) + repeats - T) < 0 else 1
        end = 2 if S > 1 else 1

        for i in range(start, end):
            alphas[i] = self.safe_log(probs[self.offset + labels[i]])

        for t in range(1, T):
            remain = int(S / 2) + repeats - (T - t)
            if (remain >= 0):
                start += s_inc[remain]
            if (t <= (S / 2) + repeats):
                end += e_inc[t - 1]
            startloop = start
            idx1 = t * S
            idx2 = (t - 1) * S
            idx3 = t * (self.num_classes * self.batch_size)

            if (start == 0):
                alphas[idx1] = alphas[idx2] + self.safe_log(
                    probs[self.offset + self.blank + idx3])
                startloop += 1
            for i in range(startloop, end):
                prev_sum = self.log_add(alphas[i + idx2],
                                        alphas[(i - 1) + idx2])

                # Skip two if not on blank and not on repeat.
                if (labels[i] != self.blank and i != 1
                        and labels[i] != labels[i - 2]):
                    prev_sum = self.log_add(prev_sum, alphas[(i - 2) + idx2])

                alphas[i + idx1] = prev_sum + self.safe_log(
                    probs[self.offset + labels[i] + idx3])
        loglike = -np.inf
        for i in range(start, end):
            loglike = self.log_add(loglike, alphas[i + (T - 1) * S])

        return loglike

    def compute_betas_and_grad(self, grad, probs, log_partition, repeats, S, T,
                               e_inc, s_inc, labels, alphas, betas, output):
        start = (S - 2) if (S - 1) else 0
        end = S if (T > (S / 2) + repeats) else (S - 1)

        #set the starting values in the beta column at the very right edge
        for i in range(start, end):
            betas[i] = self.safe_log(
                probs[self.offset + labels[i] + (T - 1) *
                      (self.num_classes * self.batch_size)])

            #compute alpha * beta in log space at this position in (S, T) space
            alphas[i + (T - 1) * S] += betas[i]

            #update the gradient associated with this label
            #essentially performing a reduce-by-key in a sequential manner
            output[labels[i]] = self.log_add(alphas[i + (T - 1) * S],
                                             output[labels[i]])

        #update the gradient wrt to each unique label
        for i in range(self.num_classes):
            idx3 = (T - 1) * self.num_classes * self.batch_size + i

            if (output[i] == 0.0 or output[i] == -np.inf
                    or probs[self.offset + idx3] == 0.0):
                grad[self.offset + idx3] = probs[self.offset + idx3]
            else:
                grad[self.offset +
                     idx3] = probs[self.offset + idx3] - self.safe_exp(
                         output[i] - self.safe_log(probs[self.offset + idx3]) -
                         log_partition)

        #loop from the second to last column all the way to the left
        for t in range(T - 2, -1, -1):
            remain = int(S / 2) + repeats - (T - t)
            if (remain >= -1):
                start -= s_inc[remain + 1]
            if (t < (S / 2) + repeats):
                end -= e_inc[t]

            endloop = (end - 1) if (end == S) else end
            idx1 = t * S
            idx3 = t * (self.num_classes * self.batch_size)

            output = [-np.inf for i in output]

            for i in range(start, endloop):
                next_sum = self.log_add(betas[i], betas[(i + 1)])
                # Skip two if not on blank and not on repeat.
                if (labels[i] != self.blank and i != (S - 2)
                        and labels[i] != labels[i + 2]):
                    next_sum = self.log_add(next_sum, betas[(i + 2)])
                betas[i] = next_sum + self.safe_log(
                    probs[self.offset + labels[i] + idx3])

                #compute alpha * beta in log space
                alphas[i + idx1] += betas[i]

                #update the gradient associated with this label
                output[labels[i]] = self.log_add(alphas[i + idx1],
                                                 output[labels[i]])

            if (end == S):
                betas[(S - 1)] = betas[(S - 1)] + self.safe_log(
                    probs[self.offset + self.blank + idx3])
                alphas[(S - 1) + idx1] += betas[(S - 1)]

                output[labels[S - 1]] = self.log_add(alphas[S - 1 + idx1],
                                                     output[labels[S - 1]])

            #go over the unique labels and compute the final grad
            #wrt to each one at this time step
            for i in range(self.num_classes):
                if (output[i] == 0.0 or output[i] == -np.inf
                        or probs[self.offset + idx3] == 0.0):
                    grad[self.offset + idx3] = probs[self.offset + idx3]
                else:
                    grad[self.offset +
                         idx3] = probs[self.offset + idx3] - self.safe_exp(
                             output[i] -
                             self.safe_log(probs[self.offset + idx3]) -
                             log_partition)

                idx3 += 1

        loglike = -np.inf
        for i in range(start, end):
            loglike = self.log_add(loglike, betas[i])

        return -loglike, grad

    def cost_and_grad(self, grads, probs, labels, T, L, mb):
        S = 2 * L + 1
        ctcm = CTCM(L, S, T, mb, self.num_classes, self.blank, labels)
        repeats, s_inc, e_inc, labels_w_blanks, alphas, betas, output = ctcm.setup_labels(
        )
        over_threshold = False
        if (L + repeats > T):
            return float(0), over_threshold

        llForward = self.compute_alphas(probs, repeats, S, T, e_inc, s_inc,
                                        labels_w_blanks, alphas)
        loss, grads = self.compute_betas_and_grad(grads, probs, llForward,
                                                  repeats, S, T, e_inc, s_inc,
                                                  labels_w_blanks, alphas,
                                                  betas, output)
        return loss, grads

    def forward(self):
        for i in range(self.batch_size):
            T = self.input_lengths[i]
            L = self.labels_lengths[i]
            self.offset = i * self.num_classes
            if self.multi_label:
                self.loss[i], grads = self.cost_and_grad(
                    self.grads, self.softmax,
                    self.labels[max(self.labels_lengths) * i:], T, L, i)
            else:
                self.loss[i], grads = self.cost_and_grad(
                    self.grads, self.softmax,
                    self.labels[sum(self.labels_lengths[:i]):], T, L, i)

        return self.loss, grads


class TestWarpCTCOp(OpTest):

    def config(self):
        self.batch_size = 4
        self.num_classes = 12
        self.logits_lod = [[4, 1, 4, 4]]
        self.labels_lod = [[3, 1, 4, 4]]
        self.logits_length = np.array([4, 1, 4, 4], dtype=np.int32)
        self.labels_length = np.array([3, 1, 4, 4], dtype=np.int32)
        self.blank = self.num_classes - 1
        self.norm_by_times = False

    def setUp(self):
        self.op_type = "warpctc"
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.config()

        logits = np.random.uniform(
            0.1, 1.0,
            [max(self.logits_lod[0]), self.batch_size, self.num_classes
             ]).astype("float32")
        softmax = np.apply_along_axis(stable_softmax, -1, logits)
        # labels should not be blank
        labels = np.random.randint(0,
                                   self.num_classes - 1,
                                   [sum(self.labels_lod[0])],
                                   dtype="int32")

        ctc = CTCForward(softmax.flatten(), self.logits_length, labels,
                         self.labels_length, self.num_classes, self.batch_size,
                         self.blank)
        loss, self.gradient = ctc.forward()
        loss = np.squeeze(loss, axis=-1)

        self.inputs = {
            "Logits": logits,
            "Label": labels,
            "LogitsLength": self.logits_length,
            "LabelLength": self.labels_length
        }
        self.outputs = {"Loss": loss}
        self.attrs = {
            "blank": self.blank,
        }

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.outputs['WarpCTCGrad'] = self.gradient
        if core.is_compiled_with_rocm():
            self.check_grad_with_place(
                self.place, ["Logits"],
                "Loss",
                max_relative_error=0.009,
                user_defined_grads=[
                    self.gradient.reshape(max(self.logits_lod[0]),
                                          self.batch_size, self.num_classes)
                ],
                check_dygraph=False)
        else:
            self.check_grad_with_place(
                self.place, ["Logits"],
                "Loss",
                max_relative_error=0.007,
                user_defined_grads=[
                    self.gradient.reshape(max(self.logits_lod[0]),
                                          self.batch_size, self.num_classes)
                ],
                check_dygraph=False)


class TestWarpCTCOpCase1(TestWarpCTCOp):

    def config(self):
        self.batch_size = 4
        self.num_classes = MLU_CLUSTER_SIZE + 2
        self.logits_lod = [[4, 1, 4, 4]]
        self.labels_lod = [[3, 1, 4, 4]]
        self.logits_length = np.array([4, 1, 4, 4], dtype=np.int32)
        self.labels_length = np.array([3, 1, 4, 4], dtype=np.int32)
        self.blank = self.num_classes - 1
        self.norm_by_times = False


class TestWarpCTCOpWithPadding(OpTest):

    def config(self):
        self.batch_size = 4
        self.num_classes = 8
        self.logits_lod = [[4, 1, 4, 4]]
        self.labels_lod = [[3, 1, 4, 4]]
        self.logits_length = np.array([4, 1, 4, 4], dtype=np.int32)
        self.labels_length = np.array([3, 1, 4, 4], dtype=np.int32)
        self.blank = self.num_classes - 1
        self.norm_by_times = False

    def setUp(self):
        self.op_type = "warpctc"
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True

        self.config()

        logits = np.random.uniform(
            0.1, 1.0,
            [max(self.logits_length), self.batch_size, self.num_classes
             ]).astype("float32")
        softmax = np.apply_along_axis(stable_softmax, -1, logits)
        # labels should not be blank
        labels = np.random.randint(0,
                                   self.num_classes - 1,
                                   [sum(self.labels_length)],
                                   dtype="int32")

        ctc = CTCForward(softmax.flatten(), self.logits_length, labels,
                         self.labels_length, self.num_classes, self.batch_size,
                         self.blank)
        loss, self.gradient = ctc.forward()
        loss = np.squeeze(loss, axis=-1)
        max_sequence_length = 0
        for i in range(self.batch_size):
            max_sequence_length = max(max_sequence_length,
                                      self.logits_length[i])

        max_target_seq_length = 0
        for i in range(self.batch_size):
            max_target_seq_length = max(max_target_seq_length,
                                        self.labels_length[i])
        new_labels = np.zeros([self.batch_size, max_target_seq_length],
                              dtype="int32")

        cur = 0
        for batch_id in range(self.batch_size):
            for i in range(self.labels_length[batch_id]):
                new_labels[batch_id, i] = labels[cur + i]
            cur = cur + self.labels_length[batch_id]

        self.inputs = {
            "Logits": logits,
            "Label": new_labels,
            "LogitsLength": self.logits_length,
            "LabelLength": self.labels_length
        }
        self.outputs = {"Loss": loss}
        self.attrs = {
            "blank": self.blank,
        }

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.outputs['WarpCTCGrad'] = self.gradient
        if core.is_compiled_with_rocm():
            self.check_grad_with_place(
                self.place, ["Logits"],
                "Loss",
                max_relative_error=0.009,
                user_defined_grads=[
                    self.gradient.reshape(max(self.logits_lod[0]),
                                          self.batch_size, self.num_classes)
                ],
                check_dygraph=False)
        else:
            self.check_grad_with_place(
                self.place, ["Logits"],
                "Loss",
                max_relative_error=0.007,
                user_defined_grads=[
                    self.gradient.reshape(max(self.logits_lod[0]),
                                          self.batch_size, self.num_classes)
                ],
                check_dygraph=False)


class TestWarpCTCOpWithPaddingCase1(TestWarpCTCOpWithPadding):

    def config(self):
        self.batch_size = 4
        self.num_classes = MLU_CLUSTER_SIZE + 2
        self.logits_lod = [[4, 1, 4, 4]]
        self.labels_lod = [[3, 1, 4, 4]]
        self.logits_length = np.array([4, 1, 4, 4], dtype=np.int32)
        self.labels_length = np.array([3, 1, 4, 4], dtype=np.int32)
        self.blank = self.num_classes - 1
        self.norm_by_times = False


class TestWarpCTCOpWithPaddingCase2(TestWarpCTCOpWithPadding):

    def config(self):
        self.batch_size = 4
        self.num_classes = 8
        self.logits_lod = [[4, 1, 5, 5]]
        self.labels_lod = [[3, 1, 4, 2]]
        self.logits_length = np.array([4, 1, 5, 5], dtype=np.int32)
        self.labels_length = np.array([3, 1, 4, 2], dtype=np.int32)
        self.blank = self.num_classes - 1
        self.norm_by_times = False


class TestWarpCTCOpError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            logits = fluid.data(name='logits',
                                shape=[5, 16, 6],
                                dtype='float32')
            logits_length = fluid.data(name='logits_length',
                                       shape=[None],
                                       dtype='int64')
            label = fluid.data(name='label', shape=[16, 3], dtype='int32')
            label_length = fluid.data(name='labels_length',
                                      shape=[None],
                                      dtype='int64')

            def test_logits_Variable():
                logits_data = np.random.rand(5, 16, 6).astype(logits.dtype)
                fluid.layers.warpctc(input=logits_data,
                                     label=label,
                                     input_length=logits_length,
                                     label_length=label_length)

            self.assertRaises(TypeError, test_logits_Variable)

            def test_label_Variable():
                label_data = np.random.randint(0, 5, [5, 1]).astype("int32")
                fluid.layers.warpctc(input=logits,
                                     label=label_data,
                                     input_length=logits_length,
                                     label_length=label_length)

            self.assertRaises(TypeError, test_label_Variable)

            def test_logits_len_Variable():
                logits_length_data = np.array([5] * 16).astype("int64")
                fluid.layers.warpctc(input=logits,
                                     label=label,
                                     input_length=logits_length_data,
                                     label_length=label_length)

            self.assertRaises(TypeError, test_logits_len_Variable)

            def test_label_len_Variable():
                label_length_data = np.array([3] * 16).astype("int64")
                fluid.layers.warpctc(input=logits,
                                     label=label,
                                     input_length=logits_length,
                                     label_length=label_length_data)

            self.assertRaises(TypeError, test_label_len_Variable)

    def test_dygraph_errors(self):

        def test_dygraph_with_lod():

            logits = np.random.uniform(0.1, 1.0, [20, 15]).astype("float32")
            # labels should not be blank
            labels = np.random.randint(0, 15 - 1, [15, 1], dtype="int32")
            softmax = paddle.to_tensor(logits)
            labels = paddle.to_tensor(labels)

            fluid.layers.warpctc(input=softmax, label=labels)

        paddle.disable_static()
        self.assertRaises(ValueError, test_dygraph_with_lod)
        paddle.enable_static()


class TestCTCLossAPICase(unittest.TestCase):

    def test_functinal_api(self):
        self.batch_size = 4
        self.num_classes = MLU_CLUSTER_SIZE + 2
        self.logits_length = np.array([4, 1, 3, 3], dtype=np.int32)
        self.labels_length = np.array([3, 1, 4, 4], dtype=np.int32)
        self.blank = self.num_classes - 1
        self.norm_by_times = False

        logits = np.random.uniform(
            0.1, 1.0,
            [max(self.logits_length), self.batch_size, self.num_classes
             ]).astype("float32")
        softmax = np.apply_along_axis(stable_softmax, -1, logits)
        # labels should not be blank
        labels = np.random.randint(
            0,
            self.num_classes - 1,
            [self.batch_size, max(self.labels_length)],
            dtype="int32")

        ctc = CTCForward(softmax.flatten(), self.logits_length, labels,
                         self.labels_length, self.num_classes, self.batch_size,
                         self.blank)
        loss_np, grads = ctc.forward()

        paddle.disable_static()
        softmax = paddle.to_tensor(logits)
        labels = paddle.to_tensor(labels)
        logits_length = paddle.to_tensor(self.logits_length)
        labels_length = paddle.to_tensor(self.labels_length)
        loss_pd_mean = F.ctc_loss(softmax,
                                  labels,
                                  logits_length,
                                  labels_length,
                                  blank=self.blank,
                                  reduction='mean')
        loss_pd_mean = loss_pd_mean.numpy()

        loss_pd_sum = F.ctc_loss(softmax,
                                 labels,
                                 logits_length,
                                 labels_length,
                                 blank=self.blank,
                                 reduction='sum')
        loss_pd_sum = loss_pd_sum.numpy()
        paddle.enable_static()
        loss_np = np.squeeze(loss_np, axis=-1)
        loss_np_mean = (loss_np / labels_length.numpy()).mean()
        loss_np_sum = loss_np.sum()

        self.assertTrue(np.allclose(loss_pd_mean, loss_np_mean, atol=1))
        self.assertTrue(np.allclose(loss_pd_sum, loss_np_sum, atol=1))

    def test_class_api(self):
        self.batch_size = 3
        self.num_classes = 15
        self.logits_length = np.array([3, 3, 3], dtype=np.int32)
        self.labels_length = np.array([1, 1, 2], dtype=np.int32)
        self.blank = 0
        self.norm_by_times = False

        logits = np.random.uniform(
            0.1, 1.0,
            [max(self.logits_length), self.batch_size, self.num_classes
             ]).astype("float32")
        softmax = np.apply_along_axis(stable_softmax, -1, logits)
        # labels should not be blank
        labels = np.random.randint(
            1,
            self.num_classes,
            [self.batch_size, max(self.labels_length)],
            dtype="int32")

        ctc = CTCForward(softmax.flatten(), self.logits_length, labels,
                         self.labels_length, self.num_classes, self.batch_size,
                         self.blank)
        loss_np, grads = ctc.forward()

        paddle.disable_static()
        softmax = paddle.to_tensor(logits)
        labels = paddle.to_tensor(labels)
        logits_length = paddle.to_tensor(self.logits_length)
        labels_length = paddle.to_tensor(self.labels_length)

        loss_pd = F.ctc_loss(softmax,
                             labels,
                             logits_length,
                             labels_length,
                             blank=self.blank,
                             reduction='none')
        loss_pd = loss_pd.numpy()
        paddle.enable_static()
        loss_np = np.squeeze(loss_np, axis=-1)

        self.assertTrue(np.allclose(loss_pd, loss_np, atol=1))


if __name__ == "__main__":
    unittest.main()
