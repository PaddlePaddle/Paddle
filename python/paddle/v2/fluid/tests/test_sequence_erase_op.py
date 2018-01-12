import unittest
import numpy as np
from op_test import OpTest


def sequence_erase(in_seq, lod0, tokens):
    new_lod0 = [0]
    out_seq = []
    for i in range(0, len(lod0) - 1):
        num_out = 0
        for dat in in_seq[lod0[i]:lod0[i + 1]]:
            if dat not in tokens:
                out_seq.append(dat)
                num_out += 1
        new_lod0.append(new_lod0[-1] + num_out)
    return np.array(out_seq).astype("int32"), new_lod0


class TestSequenceEraseOp(OpTest):
    def setUp(self):
        self.op_type = "sequence_erase"
        in_seq = np.random.randint(0, 10, (30, 1)).astype("int32")
        lod = [[0, 9, 13, 24, 30]]
        tokens = [2, 3, 5]
        out_seq, new_lod0 = sequence_erase(in_seq, lod[0], tokens)
        self.attrs = {'tokens': tokens}
        self.inputs = {'X': (in_seq, lod)}
        self.outputs = {'Out': (out_seq, [new_lod0])}

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
