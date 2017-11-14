import unittest
import numpy as np
import sys
from op_test import OpTest

class TestSequenceSliceOp(OpTest):
    def set_data(self):
        # only supprot one level LoD
        x = np.random.random((100, 3, 2)).astype('float32')
        lod = [[0, 20, 40, 60, 80, 100]]
        offset = np.array([1, 2, 3, 4, 5]).flatten().astype("int64")
        length = np.array([10, 8, 6, 4, 2]).flatten().astype("int64")

        self.inputs = {'X': (x, lod), 'Offset': offset, 'Length': length}
        outs = np.zeros((100, 3, 2)).astype('float32')
        out_lod = [[0]]
        out_lod_offset = 0
        for i in range(len(offset)):
            sub_x = x[lod[0][i] + offset[i]: lod[0]
                      [i] + offset[i] + length[i], :]
            out_lod_offset = out_lod_offset + len(sub_x)
            outs[out_lod[0][i]: out_lod_offset, :] = sub_x
            out_lod[0].append(out_lod_offset)

        self.outputs = {'Out': (outs, out_lod)}

    def setUp(self):
        self.op_type = "sequence_slice"
        self.set_data()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

if __name__ == '__main__':
    unittest.main()
