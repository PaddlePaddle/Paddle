import unittest
import random
import numpy as np

from op_test import OpTest


class CRFDecoding(object):
    def __init__(self, emission_weights, transition_weights,
                 seq_start_positions):
        assert (emission_weights.shape[0] == seq_start_positions[-1])
        self.tag_num = emission_weights.shape[1]
        self.seq_num = len(seq_start_positions) - 1

        self.seq_start_positions = seq_start_positions
        self.x = emission_weights

        self.a = transition_weights[0, :]
        self.b = transition_weights[1, :]
        self.w = transition_weights[2:, :]

        self.track = np.zeros(
            (seq_start_positions[-1], self.tag_num), dtype="int32")
        self.decoded_path = np.zeros(
            (seq_start_positions[-1], 1), dtype="int32")

    def _decode_one_sequence(self, decoded_path, x):
        seq_len, tag_num = x.shape
        alpha = np.zeros((seq_len, tag_num), dtype="float64")
        track = np.zeros((seq_len, tag_num), dtype="int32")

        for i in range(tag_num):
            alpha[0, i] = self.a[i] + x[0, i]

        for k in range(1, seq_len):
            for i in range(tag_num):
                max_score = -np.finfo("float64").max
                max_idx = 0
                for j in range(tag_num):
                    score = alpha[k - 1, j] + self.w[j, i]
                    if score > max_score:
                        max_score = score
                        max_idx = j
                alpha[k, i] = max_score + x[k, i]
                track[k, i] = max_idx

        max_score = -np.finfo("float64").max
        max_idx = 0
        for i in range(tag_num):
            score = alpha[seq_len - 1, i] + self.b[i]
            if score > max_score:
                max_score = score
                max_idx = i

        decoded_path[-1] = max_idx
        for i in range(seq_len - 1, 0, -1):
            decoded_path[i - 1] = max_idx = track[i, max_idx]

    def decode(self):
        for i in range(self.seq_num):
            start = self.seq_start_positions[i]
            end = self.seq_start_positions[i + 1]
            self._decode_one_sequence(self.decoded_path[start:end, :],
                                      self.x[start:end, :])
        return self.decoded_path


class TestCRFDecodingOp1(OpTest):
    """
    Compare the dynamic program with random generated parameters and inputs
    with grouth truth not being given.
    """

    def set_test_data(self):
        SEQ_NUM = 3
        TAG_NUM = 17
        MAX_SEQ_LEN = 10

        lod = [[0]]
        for i in range(SEQ_NUM):
            lod[-1].append(lod[-1][-1] + random.randint(1, MAX_SEQ_LEN))
        emission = np.random.uniform(-1, 1,
                                     [lod[-1][-1], TAG_NUM]).astype("float64")
        transition = np.random.uniform(-0.5, 0.5,
                                       [TAG_NUM + 2, TAG_NUM]).astype("float64")

        self.inputs = {
            "Emission": (emission, lod),
            "Transition": transition,
        }

        decoder = CRFDecoding(emission, transition, lod[0])
        decoded_path = decoder.decode()

        self.outputs = {"ViterbiPath": decoded_path}

    def setUp(self):
        self.op_type = "crf_decoding"
        self.set_test_data()

    def test_check_output(self):
        self.check_output()


class TestCRFDecodingOp2(OpTest):
    """
    Compare the dynamic program with brute force computation with
    ground truth being given.
    """

    def setUp(self):
        self.op_type = "crf_decoding"
        TAG_NUM = 5

        lod = [[0, 1, 3, 6, 10]]
        transition = np.repeat(
            np.arange(
                TAG_NUM, dtype="float64").reshape(1, TAG_NUM),
            TAG_NUM + 2,
            axis=0)
        emission = np.repeat(
            np.arange(
                TAG_NUM, dtype="float64").reshape(1, TAG_NUM),
            lod[-1][-1],
            axis=0)

        labels = np.random.randint(
            low=0, high=TAG_NUM, size=(lod[-1][-1], 1), dtype="int32")
        predicted_labels = np.ones(
            (lod[-1][-1], 1), dtype="int32") * (TAG_NUM - 1)
        expected_output = (labels == predicted_labels).astype("int32")

        self.inputs = {
            "Emission": (emission, lod),
            "Transition": transition,
            "Label": (labels, lod)
        }

        self.outputs = {"ViterbiPath": expected_output}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
