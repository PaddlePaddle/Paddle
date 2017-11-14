import unittest
import numpy as np
import sys
from op_test import OpTest

class TestSubSequenceOp(OpTest):
    def set_data(self):
        # only supprot one level LoD
        x = np.random.random((100, 3, 2)).astype('float32')
        lod = [[0, 20, 40, 60, 80, 100]]
        offsets = np.array([1, 2, 3, 4, 5]).flatten()
        sizes = np.array([10, 8, 6, 4, 2]).flatten()

        self.inputs = {'X': (x, lod)}
        self.attrs = {'offset': offsets, 'size': sizes}
        outs = []
        out_lod = [[0]]
        out_lod_offset = 0
        for i in range(len(offsets)):
            sub_x = x[lod[0][i] + offsets[i]: lod[0]
                      [i] + offsets[i] + sizes[i], :]
            outs.append(sub_x)
            out_lod_offset = out_lod_offset + len(sub_x)
            out_lod[0].append(out_lod_offset)

        outs = np.concatenate(outs, axis=0)
        self.outputs = {'Out': outs}

    def setUp(self):
        self.op_type = "sub_sequence"
        self.set_data()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

if __name__ == '__main__':
    unittest.main()
