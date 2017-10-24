import unittest
import numpy as np
import random
from op_test import OpTest


class TestSeqProject(OpTest):
    def setUp(self):
        self.init_test_case()
        self.op_type = 'sequence_conv'

        if self.context_length == 1 \
                and self.context_start == 0 \
                and self.padding_trainable:
            print "If context_start is 0 " \
                  "and context_length is 1," \
                  " padding_trainable should be false."
            return

        # one level, batch size
        x = np.random.uniform(0.1, 1, [self.input_size[0],
                                       self.input_size[1]]).astype('float32')

        self.begin_pad = np.max([0, -self.context_start])
        self.end_pad = np.max([0, self.context_start + self.context_length - 1])
        self.total_pad = self.begin_pad + self.end_pad
        if self.total_pad == 0:
            self.total_pad = 1

        # PaddingData mast be not empty.
        # Otherwise(EnforceNotMet: enforce numel() > 0 failed, 0 <= 0)
        padding_data = np.random.uniform(
            0.1, 1, [self.total_pad, self.input_size[1]]).astype('float32')
        w = np.random.uniform(
            0.1, 1, [self.context_length, self.input_size[1]]).astype('float32')
        self.inputs = {
            'X': (x, self.lod),
            'PaddingData': (padding_data, [[0, self.total_pad]]),
            'Filter': (w, [[0, self.context_length]])
        }
        self.attrs = {
            'context_start': self.context_start,
            'context_length': self.context_length,
            'padding_trainable': self.padding_trainable,
            'context_stride': self.context_stride
        }
        out = np.zeros((self.input_size[0], 1)).astype('float32')
        self.outputs = {'Out': out}
        self.compute()

    def compute(self):
        x, lod = self.inputs['X']
        filter = self.inputs['Filter']
        pading_data, _ = self.inputs['PaddingData']
        out = np.zeros((self.input_size[0], self.context_length *
                        self.input_size[1])).astype('float32')
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
                        sub_w = pading_data[j:j + pad_size, :]
                        out[lod[i]:lod[i] + pad_size, j * self.input_size[1]:(
                            j + 1) * self.input_size[1]] = sub_w
                    out_begin = lod[i] + pad_size
                    in_begin = lod[i]

                if in_end > lod[i + 1]:
                    pad_size = np.min(
                        [in_end - lod[i + 1], lod[i + 1] - lod[i]])
                    if self.padding_trainable:
                        sub_w = pading_data[begin_pad + self.context_start + j -
                                            pad_size:begin_pad +
                                            self.context_start + j, :]
                        out[lod[i + 1] - pad_size:lod[i + 1], j * self.
                            input_size[1]:(j + 1) * self.input_size[1]] = sub_w
                    in_end = lod[i + 1]
                    out_end = lod[i + 1] - pad_size
                if in_end <= in_begin:
                    continue

                in_sub = x[in_begin:in_end, :]
                out[out_begin:out_end, j * self.input_size[1]:(j + 1) *
                    self.input_size[1]] += in_sub

        filter_dim = filter[0].shape
        output_dim = self.outputs['Out'].shape
        filter[0].shape = filter_dim[0] * filter_dim[1]
        self.outputs['Out'].shape = (output_dim[0], )
        np.dot(out, filter[0], out=self.outputs['Out'])
        filter[0].shape = filter_dim
        self.outputs['Out'].shape = output_dim

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.padding_trainable:
            self.check_grad(
                set(['X', 'PaddingData', 'Filter']),
                'Out',
                max_relative_error=0.05)

    def test_check_grad_input(self):
        self.check_grad(
            ['X'],
            'Out',
            max_relative_error=0.05,
            no_grad_set=set(['PaddingData', 'Filter']))

    def test_check_grad_padding_data(self):
        if self.padding_trainable:
            self.check_grad(
                ['PaddingData'],
                'Out',
                max_relative_error=0.05,
                no_grad_set=set(['X', 'Filter']))

    def test_check_grad_Filter(self):
        self.check_grad(
            ['Filter'],
            'Out',
            max_relative_error=0.05,
            no_grad_set=set(['X', 'PaddingData']))

    def test_check_grad_input_filter(self):
        self.check_grad(
            ['X', 'Filter'],
            'Out',
            max_relative_error=0.05,
            no_grad_set=set(['PaddingData']))

    def test_check_grad_padding_input(self):
        if self.padding_trainable:
            self.check_grad(
                ['X', 'PaddingData'],
                'Out',
                max_relative_error=0.05,
                no_grad_set=set(['Filter']))

    def test_check_grad_padding_filter(self):
        if self.padding_trainable:
            self.check_grad(
                ['PaddingData', 'Filter'],
                'Out',
                max_relative_error=0.05,
                no_grad_set=set(['X']))

    def init_test_case(self):
        self.input_row = 11
        self.context_start = 0
        self.context_length = 1
        self.padding_trainable = False
        self.context_stride = 1

        self.input_size = [self.input_row, 23]
        self.lod = [[0, 4, 5, 8, self.input_row]]


class TestSeqProjectCase1(TestSeqProject):
    def init_test_case(self):
        self.input_row = 11
        self.context_start = -1
        self.context_length = 3
        self.padding_trainable = True
        self.context_stride = 1

        self.input_size = [self.input_row, 23]
        self.lod = [[0, 4, 5, 8, self.input_row]]


class TestSeqProjectCase2(TestSeqProject):
    def init_test_case(self):
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
                        self.end_pad = np.max([0, self.context_start + self.context_length - 1])
                        self.total_pad = self.begin_pad + self.end_pad
                        if self.total_pad == 0:
                            self.total_pad = 1
                        # PaddingData mast be not empty. Otherwise(EnforceNotMet: enforce numel() > 0 failed, 0 <= 0)
                        padding_data = np.random.uniform(
                            0.1, 1, [self.total_pad, self.input_size[1]]).astype('float32')

                        self.inputs = {
                            'X': (x, self.lod),
                            'PaddingData': (padding_data, [[0, self.total_pad]])
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
