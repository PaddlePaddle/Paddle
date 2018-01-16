import unittest
import numpy as np
import sys
import math
from op_test import OpTest


class TestIOUSimilarityOp(OpTest):
    def set_data(self):
        self.init_test_data()
        self.inputs = {'X': self.boxes1, 'Y': self.boxes2}

        self.outputs = {'Out': self.output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        return

    def setUp(self):
        self.op_type = "iou_similarity"
        self.set_data()

    def init_test_data(self):
        self.boxes1 = np.array(
            [[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]]).astype('float32')
        self.boxes2 = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                                [0.0, 0.0, 20.0, 20.0]]).astype('float32')
        self.output = np.array(
            [[2.0 / 16.0, 0, 6.0 / 400.0],
             [1.0 / 16.0, 0.0, 5.0 / 400.0]]).astype('float32')


if __name__ == '__main__':
    unittest.main()
