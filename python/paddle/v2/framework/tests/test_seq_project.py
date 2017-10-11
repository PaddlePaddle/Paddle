import unittest
import numpy as np
from op_test import OpTest


class TestSeqProject(OpTest):
    def setUp(self):
        self.init_test_case()
        self.op_type = 'sequence_project'
        # one level, batch size
        x = np.random.uniform(
            0.1, 1, [self.input_size[0], self.input_size[1]]).astype('float32')
        lod = [[0, 4, 5, 8, self.input_size[0]]]

        self.begin_pad = np.max([0, -self.context_start])
        self.end_pad = np.max([0, self.context_start + self.context_length - 1])
        self.total_pad = self.begin_pad + self.end_pad
        w = np.ones((self.total_pad, self.input_size[1])) * 100

        self.inputs = {'X': (x, lod), 'PaddingData': w}
        self.attrs = {
            'context_start': self.context_start,
            'context_length': self.context_length,
            'padding_trainable': self.padding_trainable
        }
        out = np.zeros((self.input_size[0], self.input_size[1] *
                        self.context_length)).astype('float32')
        self.outputs = {'Out': out}
        self.compute()

    def compute(self):
        x, lod = self.inputs['X']
        w = self.inputs['PaddingData']
        out = self.outputs['Out']
        lod = lod[0]

        for i in range(len(lod) - 1):
            for j in range(self.context_length):
                in_begin = lod[i] + self.context_start + j
                in_end = lod[i + 1] + self.context_start + j
                out_begin = lod[i]
                out_end = lod[i + 1]
                if in_begin < lod[i]:
                    pad_size = np.min([lod[i] - in_begin, lod[i + 1] - lod[i]])
                    if self.padding_trainable:
                        sub_w = w[j:pad_size, :]
                        out[lod[i]:lod[i] + pad_size, j * self.input_size[1]:(
                            j + 1) * self.input_size[1]] = sub_w
                        # pass
                    out_begin = lod[i] + pad_size
                    in_begin = lod[i]

                if in_end > lod[i + 1]:
                    pad_size = np.min(
                        [in_end - lod[i + 1], lod[i + 1] - lod[i]])
                    out_sub = out[lod[i + 1] - pad_size:lod[i + 1], :]
                    if self.padding_trainable:
                        sub_w = w[j - pad_size:j, :]
                        out[lod[i + 1] - pad_size:lod[i + 1], j * self.
                            input_size[1]:(j + 1) * self.input_size[1]] = sub_w
                        # pass
                    in_end = lod[i + 1]
                    out_end = lod[i + 1] - pad_size
                if in_end <= in_begin:
                    continue

                in_sub = x[in_begin:in_end, :]
                out[out_begin:out_end, j * self.input_size[1]:(j + 1) *
                    self.input_size[1]] += in_sub

    def init_test_case(self):
        self.input_size = [11, 23]
        self.op_type = "sequence_project"

        self.context_start = -1
        self.context_length = 3
        self.padding_trainable = False

    def test_check_output(self):
        self.check_output()

    # def test_check_grad(self):
    #     self.check_grad(["X"], "Out")

    # class TestSeqAvgPool2D(TestSeqProject):
    #     def init_test_case(self):
    #         self.input_size = [11, 23]
    #         self.op_type = "sequence_project"
    #
    #         self.context_start = -1
    #         self.context_length = 3
    #         self.padding_trainable = True


if __name__ == '__main__':
    unittest.main()
