import unittest
import numpy as np
import random
from op_test import OpTest


class TestSeqProject(OpTest):
    def setUp(self):
        self.init_test_case()
        self.op_type = 'sequence_project'
        # one level, batch size
        x = np.random.uniform(
            0.1, 1, [self.input_size[0], self.input_size[1]]).astype('float32')

        self.begin_pad = np.max([0, -self.context_start])
        self.end_pad = np.max([0, self.context_start + self.context_length - 1])
        self.total_pad = self.begin_pad + self.end_pad
        w = np.random.uniform(
            0.1, 1, [self.total_pad, self.input_size[1]]).astype('float32')
        self.inputs = {
            'X': (x, self.lod),
            'PaddingData': (w, [[0, self.total_pad]])
        }
        self.attrs = {
            'context_start': self.context_start,
            'context_length': self.context_length,
            'padding_trainable': self.padding_trainable,
            'context_stride': self.context_stride
        }
        out = np.zeros((self.input_size[0], self.input_size[1] *
                        self.context_length)).astype('float32')
        self.outputs = {'Out': out}
        self.compute()

    def compute(self):
        x, lod = self.inputs['X']
        w, _ = self.inputs['PaddingData']
        out = self.outputs['Out']
        lod = lod[0]
        begin_pad = np.max([0, -self.context_start])

        for i in range(len(lod) - 1):
            for j in range(self.context_length):
                in_begin = lod[i] + self.context_start + j
                in_end = lod[i + 1] + self.context_start + j
                out_begin = lod[i]
                out_end = lod[i + 1]
                if in_begin < lod[i]:
                    pad_size = np.min([lod[i] - in_begin, lod[i + 1] - lod[i]])
                    if self.padding_trainable:
                        sub_w = w[j:j + pad_size, :]
                        out[lod[i]:lod[i] + pad_size, j * self.input_size[1]:(
                            j + 1) * self.input_size[1]] = sub_w
                    out_begin = lod[i] + pad_size
                    in_begin = lod[i]

                if in_end > lod[i + 1]:
                    pad_size = np.min(
                        [in_end - lod[i + 1], lod[i + 1] - lod[i]])
                    if self.padding_trainable:
                        sub_w = w[begin_pad + self.context_start + j - pad_size:
                                  begin_pad + self.context_start + j, :]
                        out[lod[i + 1] - pad_size:lod[i + 1], j * self.
                            input_size[1]:(j + 1) * self.input_size[1]] = sub_w
                    in_end = lod[i + 1]
                    out_end = lod[i + 1] - pad_size
                if in_end <= in_begin:
                    continue

                in_sub = x[in_begin:in_end, :]
                out[out_begin:out_end, j * self.input_size[1]:(j + 1) *
                    self.input_size[1]] += in_sub

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            set(['X', 'PaddingData']), 'Out', max_relative_error=0.05)

    def test_check_grad_no_filter(self):
        self.check_grad(
            ['X'],
            'Out',
            max_relative_error=0.05,
            no_grad_set=set(['PaddingData']))

    def test_check_grad_no_input(self):
        self.check_grad(
            ['PaddingData'],
            'Out',
            max_relative_error=0.05,
            no_grad_set=set(['X']))

    def init_test_case(self):
        self.op_type = "sequence_project"
        self.input_row = 11
        self.context_start = -1
        self.context_length = 3
        self.padding_trainable = True
        self.context_stride = 1

        self.input_size = [self.input_row, 23]
        self.lod = [[0, 4, 5, 8, self.input_row]]


class TestSeqProjectCase1(TestSeqProject):
    def init_test_case(self):
        self.op_type = "sequence_project"
        self.input_row = 25
        self.context_start = 2
        self.context_length = 3
        self.padding_trainable = True
        self.context_stride = 1

        self.input_size = [self.input_row, 23]
        idx = range(self.input_size[0])
        del idx[0]
        self.lod = [[0] + np.sort(random.sample(idx, 8)).tolist() +
                    [self.input_size[0]]]


'''
class TestSeqProjectCases(TestSeqProject):
    def setUp(self):
        self.init_test_case()
        self.op_type = 'sequence_project'

        num = 0
        for context_start in [-5, -3, -1, 0, 3]:
            for context_length in [1, 2, 5, 7]:
                for batch_size in [1, 2, 5, 7]:
                    for padding_trainable in [False, True]:

                        if context_length == 1 and context_start == 0 and padding_trainable:
                            continue

                        self.context_start = context_start
                        self.context_length = context_length
                        self.padding_trainable = padding_trainable
                        self.input_size = [batch_size, 23]
                        x = np.random.uniform(0.1, 1,
                                              self.input_size).astype('float32')
                        self.lod = [[0, self.input_size[0]]]
                        if self.input_size[0] > 2:
                            idx = range(self.input_size[0])
                            del idx[0]
                            self.lod = [
                                [0] + np.sort(random.sample(idx, 2)).tolist() +
                                [self.input_size[0]]
                            ]

                        self.begin_pad = np.max([0, -self.context_start])
                        self.end_pad = np.max(
                            [0, self.context_start + self.context_length - 1])
                        self.total_pad = self.begin_pad + self.end_pad
                        # w =  np.ones((self.total_pad, self.input_size[1])) * 100
                        w = np.array(range(self.total_pad * self.input_size[1]))
                        w.shape = self.total_pad, self.input_size[1]
                        if self.total_pad * self.input_size[1] == 0:
                            w = np.random.uniform(
                                0.1, 1,
                                (1, self.input_size[1])).astype('float32')
                            self.total_pad = 1

                        self.inputs = {
                            'X': (x, self.lod),
                            'PaddingData': (w, [[0, self.total_pad]])
                        }
                        self.attrs = {
                            'context_start': self.context_start,
                            'context_length': self.context_length,
                            'padding_trainable': self.padding_trainable,
                            'context_stride': self.context_stride
                        }
                        out = np.zeros((self.input_size[0], self.input_size[1] *
                                        self.context_length)).astype('float32')
                        self.outputs = {'Out': out}
                        print num
                        print self.attrs
                        print batch_size
                        print padding_trainable
                        print "$$$$$$$$$$$$$"

                        self.compute()
                        self.test_check_output()

                        num += 1
'''

if __name__ == '__main__':
    unittest.main()
