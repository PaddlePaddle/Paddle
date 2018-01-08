import unittest
import numpy as np
from op_test import OpTest


def sequence_erase(in_seq, lod0, tokens):
    # num_erased[i]: the number of elments to be removed before #i elements
    num_erased = [0] * (len(in_seq) + 1)
    for i in range(1, len(in_seq) + 1):
        num_erased[i] = num_erased[i - 1]
        if in_seq[i - 1] in tokens:
            num_erased[i] += 1

    # recalculate lod information
    new_lod0 = [0] * len(lod0)
    for i in range(1, len(lod0)):
        new_lod0[i] = lod0[i] - num_erased[lod0[i]]

    out_seq = np.zeros(
        (len(in_seq) - num_erased[len(in_seq)], 1)).astype("int32")
    for i in range(0, len(in_seq)):
        if num_erased[i] == num_erased[i + 1]:
            out_seq[i - num_erased[i]] = in_seq[i]
        # else in_seq[i] needs to be removed 
    return out_seq, new_lod0


class TestSequenceEraseOp(OpTest):
    def setUp(self):
        self.op_type = "sequence_erase"
        in_seq = np.random.randint(0, 10, (10, 1)).astype("int32")
        lod = [[0, 3, 6, 10]]
        tokens = [2, 3, 5]
        out_seq, new_lod0 = sequence_erase(in_seq, lod[0], tokens)
        self.attrs = {'tokens': tokens}
        self.inputs = {'X': (in_seq, lod)}
        self.outputs = {'Out': (out_seq, [new_lod0])}

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    """
    in_seq = np.random.randint(0, 10, (30, 1)).astype("int32")
    lod0 = [0, 5, 15, 30]
    tokens = [2, 5]
    out_seq, new_lod = sequence_erase(in_seq, lod0, tokens)
    
    print lod0, new_lod
    print("compare")
    for i in range(0,  len(lod0)-1):
        print(np.transpose(in_seq[lod0[i] : lod0[i+1]]))
        print(np.transpose(out_seq[new_lod[i] : new_lod[i+1]]))
        print("\n")
    """
    unittest.main()
