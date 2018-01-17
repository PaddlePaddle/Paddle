import sys
import unittest
import numpy as np
from op_test import OpTest
from test_softmax_op import stable_softmax


def CTCAlign(input, lod, blank, merge_repeated):
    lod0 = lod[0]
    result = []
    for i in range(len(lod0) - 1):
        prev_token = -1
        for j in range(lod0[i], lod0[i + 1]):
            token = input[j][0]
            if (token != blank) and not (merge_repeated and
                                         token == prev_token):
                result.append(token)
            prev_token = token
    result = np.array(result).reshape([len(result), 1]).astype("int32")
    return result


class TestCTCAlignOp(OpTest):
    def config(self):
        self.op_type = "ctc_align"
        self.input_lod = [[0, 11, 18]]
        self.blank = 0
        self.merge_repeated = False
        self.input = np.array(
            [0, 1, 2, 2, 0, 4, 0, 4, 5, 0, 6, 6, 0, 0, 7, 7, 7, 0]).reshape(
                [18, 1]).astype("int32")

    def setUp(self):
        self.config()
        output = CTCAlign(self.input, self.input_lod, self.blank,
                          self.merge_repeated)

        self.inputs = {"Input": (self.input, self.input_lod), }
        self.outputs = {"Output": output}
        self.attrs = {
            "blank": self.blank,
            "merge_repeated": self.merge_repeated
        }

    def test_check_output(self):
        self.check_output()
        pass


class TestCTCAlignOpCase1(TestCTCAlignOp):
    def config(self):
        self.op_type = "ctc_align"
        self.input_lod = [[0, 11, 19]]
        self.blank = 0
        self.merge_repeated = True
        self.input = np.array(
            [0, 1, 2, 2, 0, 4, 0, 4, 5, 0, 6, 6, 0, 0, 7, 7, 7, 0, 0]).reshape(
                [19, 1]).astype("int32")


if __name__ == "__main__":
    unittest.main()
