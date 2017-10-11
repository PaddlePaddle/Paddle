import unittest
import numpy as np
from op_test import OpTest


class TestSeqExpand(OpTest):
    #class TestSeqExpand():
    def set_data(self):
        self.op_type = 'seq_expand'
        x = np.random.uniform(0.1, 1, [3, 2, 2]).astype('float32')
        y = np.zeros((6, 2, 2)).astype('float32')
        lod = [[0, 2, 3, 6]]
        print "x = %s" % x
        self.inputs = {'X': x, 'Y': (y, lod)}
        self.repeat = None

    def compute(self):
        x = self.inputs['X']
        cpy_map = {}
        lod = []
        out_shape = []
        if self.repeat:
            level0 = []
            for i in range(x.shape[0] + 1):
                level0.append(i * self.repeat)
            lod.append(level0)

            for i in x.shape:
                out_shape.append(i)
            out_shape[0] = out_shape[0] * self.repeat
        else:
            y, lod = self.inputs['Y']
            out_shape = y.shape
        out = np.zeros(out_shape).astype('float32')

        start = 0

        for i in range(len(lod[0]) - 1):
            for j in range(lod[0][i], lod[0][i + 1]):
                cpy_map[j] = i
        print "cpy_map = %s" % cpy_map
        for i in range(len(out)):
            out[i] = x[cpy_map[i]]

        print "out = %s" % out
        self.outputs = {'Out': (out, lod)}

    def setUp(self):
        self.set_data()
        self.compute()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


if __name__ == '__main__':
    unittest.main()
#    TestSeqExpand().setUp()
