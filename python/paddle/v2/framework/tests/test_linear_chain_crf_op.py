import unittest
import random
import numpy as np

from op_test import OpTest


class LinearChainCrfForward(object):
    def __init__(self, seq_start_positions, emission_weights,
                 transition_weights, labels):
        self.tag_num = emission_weights.shape[1]
        self.seq_num = len(seq_start_positions) - 1

        self.seq_start_positions = seq_start_positions
        self.labels = labels
        self.x = emission_weights

        self.x_row_max = np.amax(self.x, axis=1, keepdims=True)
        self.x_exps = np.exp(self.x - self.x_row_max)

        # unnormalized logits of the transition weights for the start mark.
        self.a = transition_weights[0, :]
        self.a_exps = np.exp(self.a)
        # unnormalized logits of the transition weights for the end mark.
        self.b = transition_weights[1, :]
        self.b_exps = np.exp(self.b)
        # unnormalized logits of the transition weights for all the other tags.
        self.w = transition_weights[2:, :]
        self.w_exps = np.exp(self.w)

        # The output of linear chain crf operator.
        # alpha is a memo table in dynamic programming to caculate
        # nomalization factor.
        self.alpha = np.zeros(
            (seq_start_positions[-1], self.tag_num), dtype="float32")
        self.log_likelihood = np.zeros((self.tag_num, 1))

    def _l1_norm(self, x):
        s = np.sum(x)
        x /= s
        return s

    def _forward_a_sequence(self, x, x_row_max, x_exps, label, alpha):
        seq_len = x_row_max.shape[0]
        log_likelihood = 0.

        for i in range(self.tag_num):
            alpha[0, i] = self.a_exps[i] * x_exps[0, i]
        log_likelihood = -x_row_max[0] - np.log(self._l1_norm(alpha[0, :]))

        # calculate the unnormalized logits of the normalization factor.
        for k in range(1, seq_len):
            for i in range(self.tag_num):
                s = 0.
                for j in range(self.tag_num):
                    s += alpha[k - 1, j] * self.w_exps[j, i]
                alpha[k, i] = x_exps[k, i] * s
            log_likelihood -= x_row_max[k] + np.log(self._l1_norm(alpha[k, :]))
        s = 0.
        for i in range(self.tag_num):
            s += alpha[-1, i] * self.b_exps[i]
        log_likelihood -= np.log(s)

        # calculate the noninator part.
        log_likelihood += (
            self.a[label[0]] + self.x[0, label[0]] + self.b[label[-1]])
        for k in range(1, seq_len):
            log_likelihood += (
                self.x[k, label[k]] + self.w[label[k - 1], label[k]])
        return log_likelihood

    def crf_forward_compute(self):
        for i in range(self.seq_num):
            start = self.seq_start_positions[i]
            end = self.seq_start_positions[i + 1]

            self.log_likelihood[i] = self._forward_a_sequence(
                self.x[start:end], self.x_row_max[start:end, :],
                self.x_exps[start:end, :], self.labels[start:end, :],
                self.alpha[start:end, :])
        return self.alpha, self.log_likelihood


class TestLinearChainCrfOp(OpTest):
    def set_test_data(self):
        SEQ_NUM = 3
        TAG_NUM = 17
        MAX_SEQ_LEN = 13

        # the linear_chain_crf operator only supports sequence (LoD level = 1)
        lod = [[0]]
        for i in range(SEQ_NUM):
            lod[-1].append(lod[-1][-1] + random.randint(1, MAX_SEQ_LEN))

        emission = np.random.uniform(-1, 1,
                                     [lod[-1][-1], TAG_NUM]).astype("float32")
        transition = np.random.uniform(-0.5, 0.5,
                                       [TAG_NUM + 2, TAG_NUM]).astype("float32")
        labels = np.random.randint(
            low=0, high=TAG_NUM, size=(lod[-1][-1], 1), dtype="int32")

        self.inputs = {
            "Emission": (emission, lod),
            "Transition": transition,
            "label": (labels, lod)
        }

        crf = LinearChainCrfForward(lod[0], emission, transition, labels)
        alpha, log_likelihood = crf.crf_forward_compute()

        self.outputs = {"Alpha": alpha, "LogLikelihood": log_likelihood}

    def setUp(self):
        self.op_type = "linear_chain_crf"
        self.set_test_data()

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
