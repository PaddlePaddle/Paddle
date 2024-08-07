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

import random
import unittest

import numpy as np
from op_test import OpTest

import paddle


def api_wrapper(emission, transition, label=None, length=None):
    return paddle._C_ops.crf_decoding(emission, transition, label, length)


class CRFDecoding:
    def __init__(
        self, emission_weights, transition_weights, seq_start_positions
    ):
        assert emission_weights.shape[0] == sum(seq_start_positions)
        self.tag_num = emission_weights.shape[1]
        self.seq_num = len(seq_start_positions)

        self.seq_start_positions = seq_start_positions
        self.x = emission_weights

        self.a = transition_weights[0, :]
        self.b = transition_weights[1, :]
        self.w = transition_weights[2:, :]

        self.track = np.zeros(
            (sum(seq_start_positions), self.tag_num), dtype="int64"
        )
        self.decoded_path = np.zeros(
            (sum(seq_start_positions), 1), dtype="int64"
        )

    def _decode_one_sequence(self, decoded_path, x):
        seq_len, tag_num = x.shape
        alpha = np.zeros((seq_len, tag_num), dtype="float64")
        track = np.zeros((seq_len, tag_num), dtype="int64")

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
        cur_pos = 0
        for i in range(self.seq_num):
            start = cur_pos
            cur_pos += self.seq_start_positions[i]
            end = cur_pos
            self._decode_one_sequence(
                self.decoded_path[start:end, :], self.x[start:end, :]
            )
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

        lod = [[]]
        total_len = 0
        for i in range(SEQ_NUM):
            lod[-1].append(random.randint(1, MAX_SEQ_LEN))
            total_len += lod[-1][-1]
        emission = np.random.uniform(-1, 1, [total_len, TAG_NUM]).astype(
            "float64"
        )
        transition = np.random.uniform(
            -0.5, 0.5, [TAG_NUM + 2, TAG_NUM]
        ).astype("float64")

        self.inputs = {
            "Emission": (emission, lod),
            "Transition": transition,
        }

        decoder = CRFDecoding(emission, transition, lod[0])
        decoded_path = decoder.decode()

        self.outputs = {"ViterbiPath": decoded_path}

    def setUp(self):
        self.op_type = "crf_decoding"
        self.python_api = api_wrapper
        self.set_test_data()

    def test_check_output(self):
        self.check_output()


class TestCRFDecodingOp2(OpTest):
    """
    Compare the dynamic program with brute force computation with
    ground truth being given.
    """

    def init_lod(self):
        self.lod = [[1, 2, 3, 4]]

    def setUp(self):
        self.op_type = "crf_decoding"
        self.python_api = api_wrapper
        TAG_NUM = 5

        self.init_lod()
        total_len = sum(self.lod[-1])
        transition = np.repeat(
            np.arange(TAG_NUM, dtype="float64").reshape(1, TAG_NUM),
            TAG_NUM + 2,
            axis=0,
        )
        emission = np.repeat(
            np.arange(TAG_NUM, dtype="float64").reshape(1, TAG_NUM),
            total_len,
            axis=0,
        )

        labels = np.random.randint(
            low=0, high=TAG_NUM, size=(total_len, 1), dtype="int64"
        )
        predicted_labels = np.ones((total_len, 1), dtype="int64") * (
            TAG_NUM - 1
        )
        expected_output = (labels == predicted_labels).astype("int64")

        self.inputs = {
            "Emission": (emission, self.lod),
            "Transition": transition,
            "Label": (labels, self.lod),
        }

        self.outputs = {"ViterbiPath": expected_output}

    def test_check_output(self):
        self.check_output()


class TestCRFDecodingOp3(TestCRFDecodingOp2):
    def init_lod(self):
        self.lod = [[1, 0, 0, 4]]


class TestCRFDecodingOp4(TestCRFDecodingOp2):
    def init_lod(self):
        self.lod = [[0, 2, 3, 0]]


def seq_pad(data, length):
    max_len = np.max(length)
    shape = [len(length), max_len, *data.shape[1:]]
    padded = np.zeros(shape).astype(data.dtype)
    offset = 0
    for i, l in enumerate(length):
        padded[i, 0:l] = data[offset : offset + l]
        offset += l
    return np.squeeze(padded)


class TestCRFDecodingOp5(OpTest):
    """
    Compare the dynamic program with random generated parameters and inputs
    with grouth truth not being given.
    """

    def set_test_data(self):
        SEQ_NUM = 3
        TAG_NUM = 17
        MAX_SEQ_LEN = 10

        lod = [[]]
        total_len = 0
        for i in range(SEQ_NUM):
            lod[-1].append(random.randint(1, MAX_SEQ_LEN))
            total_len += lod[-1][-1]
        emission = np.random.uniform(-1, 1, [total_len, TAG_NUM]).astype(
            "float64"
        )
        transition = np.random.uniform(
            -0.5, 0.5, [TAG_NUM + 2, TAG_NUM]
        ).astype("float64")

        self.inputs = {
            "Emission": seq_pad(emission, lod[0]),
            "Transition": transition,
            "Length": np.array(lod).astype('int64'),
        }

        decoder = CRFDecoding(emission, transition, lod[0])
        decoded_path = decoder.decode()

        self.outputs = {"ViterbiPath": seq_pad(decoded_path, lod[0])}

    def setUp(self):
        self.op_type = "crf_decoding"
        self.python_api = api_wrapper
        self.set_test_data()

    def test_check_output(self):
        self.check_output()


class TestCRFDecodingOp6(OpTest):
    def init_lod(self):
        self.lod = [[1, 2, 3, 4]]

    def setUp(self):
        self.op_type = "crf_decoding"
        self.python_api = api_wrapper
        TAG_NUM = 5

        self.init_lod()
        total_len = sum(self.lod[-1])
        transition = np.repeat(
            np.arange(TAG_NUM, dtype="float64").reshape(1, TAG_NUM),
            TAG_NUM + 2,
            axis=0,
        )
        emission = np.repeat(
            np.arange(TAG_NUM, dtype="float64").reshape(1, TAG_NUM),
            total_len,
            axis=0,
        )

        labels = np.random.randint(
            low=0, high=TAG_NUM, size=(total_len, 1), dtype="int64"
        )
        predicted_labels = np.ones((total_len, 1), dtype="int64") * (
            TAG_NUM - 1
        )
        expected_output = (labels == predicted_labels).astype("int64")

        self.inputs = {
            "Emission": seq_pad(emission, self.lod[0]),
            "Transition": transition,
            "Label": seq_pad(labels, self.lod[0]),
            "Length": np.array(self.lod).astype('int64'),
        }

        self.outputs = {"ViterbiPath": seq_pad(expected_output, self.lod[0])}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
