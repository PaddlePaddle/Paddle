import sys
import unittest
import numpy as np
from op_test import OpTest
from test_softmax_op import stable_softmax


def CTCGreedyDecode(softmax, blank, merge_repeated):
    prev_token = -1
    result = []
    for token in np.argmax(softmax, axis=1):
        if (token != blank) and not (merge_repeated and token == prev_token):
            result.append(token)
    return np.array(result).reshape([len(result), 1])


class TestCTCGreedyDecodeOp(OpTest):
    def config(self):
        self.op_type = "ctc_greedy_decode"
        self.batch_size = 4
        self.num_classes = 8
        self.input_lod = [[0, 4, 5, 8, 11]]
        self.blank = 7
        self.merge_repeated = True

    def setUp(self):
        self.config()
        input = np.random.uniform(
            0.1, 1.0,
            [self.input_lod[0][-1], self.num_classes]).astype("float32")
        softmax = np.apply_along_axis(stable_softmax, 1, input)
        output = CTCGreedyDecode(softmax, self.blank, self.merge_repeated)

        self.inputs = {"Input": (softmax, self.input_lod), }
        self.outputs = {"Output": output}
        self.attrs = {
            "blank": self.blank,
            "merge_repeated": self.merge_repeated
        }

    def test_check_output(self):
        self.check_output()


class TestCTCGreedyDecodeOpCase1(TestCTCGreedyDecodeOp):
    def config(self):
        self.op_type = "ctc_greedy_decode"
        self.batch_size = 4
        self.num_classes = 1025
        self.input_lod = [[0, 4, 5, 8, 11]]
        self.blank = 0
        self.merge_repeated = True


if __name__ == "__main__":
    unittest.main()
