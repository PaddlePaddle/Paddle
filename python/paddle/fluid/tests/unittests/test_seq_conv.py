#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        w = np.random.uniform(0.1, 1, [
            self.context_length * self.input_size[1], self.output_represention
        ]).astype('float32')

        begin_pad = np.max([0, -self.context_start])
        end_pad = np.max([0, self.context_start + self.context_length - 1])
        total_pad = begin_pad + end_pad
        padding_data = np.random.uniform(
            0.1, 1, [total_pad, self.input_size[1]]).astype('float32')
        self.pad_data = padding_data
        self.inputs = {
            'X': (x, self.lod),
            'Filter': w,
        }
        self.inputs_val = ['X', 'Filter']
        self.inputs_val_no_x = ['Filter']
        self.inputs_val_no_f = ['X']

        if total_pad != 0:
            self.inputs['PaddingData'] = padding_data
            self.inputs_val = ['X', 'PaddingData', 'Filter']
            self.inputs_val_no_x = ['PaddingData', 'Filter']
            self.inputs_val_no_f = ['PaddingData', 'X']

        self.attrs = {
            'contextStart': self.context_start,
            'contextLength': self.context_length,
            'paddingTrainable': self.padding_trainable,
            'contextStride': self.context_stride
        }
        out = np.zeros(
            (self.input_size[0], self.output_represention)).astype('float32')
        self.outputs = {'Out': out}
        self.compute()

    def compute(self):
        x, lod = self.inputs['X']
        filter = self.inputs['Filter']
        pading_data = self.pad_data
        out = np.zeros((self.input_size[0], self.context_length *
                        self.input_size[1])).astype('float32')
        offset = [0]
        for seq_len in lod[0]:
            offset.append(offset[-1] + seq_len)
        begin_pad = np.max([0, -self.context_start])

        for i in range(len(offset) - 1):
            for j in range(self.context_length):
                in_begin = offset[i] + self.context_start + j
                in_end = offset[i + 1] + self.context_start + j
                out_begin = offset[i]
                out_end = offset[i + 1]
                if in_begin < offset[i]:
                    pad_size = np.min(
                        [offset[i] - in_begin, offset[i + 1] - offset[i]])
                    if self.padding_trainable:
                        sub_w = pading_data[j:j + pad_size, :]
                        out[offset[i]:offset[i] + pad_size, j * self.input_size[
                            1]:(j + 1) * self.input_size[1]] = sub_w
                    out_begin = offset[i] + pad_size
                    in_begin = offset[i]

                if in_end > offset[i + 1]:
                    pad_size = np.min(
                        [in_end - offset[i + 1], offset[i + 1] - offset[i]])
                    if self.padding_trainable:
                        sub_w = pading_data[begin_pad + self.context_start + j -
                                            pad_size:begin_pad +
                                            self.context_start + j, :]
                        out[offset[i + 1] - pad_size:offset[i + 1], j * self.
                            input_size[1]:(j + 1) * self.input_size[1]] = sub_w
                    in_end = offset[i + 1]
                    out_end = offset[i + 1] - pad_size
                if in_end <= in_begin:
                    continue

                in_sub = x[in_begin:in_end, :]
                out[out_begin:out_end, j * self.input_size[1]:(j + 1) *
                    self.input_size[1]] += in_sub

        np.dot(out, filter, out=self.outputs['Out'])

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.padding_trainable:
            self.check_grad(
                set(self.inputs_val), 'Out', max_relative_error=0.05)

    def test_check_grad_input(self):
        self.check_grad(
            ['X'],
            'Out',
            max_relative_error=0.05,
            no_grad_set=set(self.inputs_val_no_x))

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
            no_grad_set=set(self.inputs_val_no_f))

    def test_check_grad_input_filter(self):
        if self.padding_trainable:
            self.check_grad(
                ['X', 'Filter'],
                'Out',
                max_relative_error=0.05,
                no_grad_set=set(['PaddingData']))

    def test_check_grad_padding_input(self):
        if self.padding_trainable:
            self.check_grad(
                self.inputs_val_no_f,
                'Out',
                max_relative_error=0.05,
                no_grad_set=set(['Filter']))

    def test_check_grad_padding_filter(self):
        if self.padding_trainable:
            self.check_grad(
                self.inputs_val_no_x,
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
        offset_lod = [[0, 4, 5, 8, self.input_row]]
        self.lod = [[]]
        # convert from offset-based lod to length-based lod
        for i in range(len(offset_lod[0]) - 1):
            self.lod[0].append(offset_lod[0][i + 1] - offset_lod[0][i])
        self.output_represention = 8  # output feature size


class TestSeqProjectCase1(TestSeqProject):
    def init_test_case(self):
        self.input_row = 11
        self.context_start = -1
        self.context_length = 3
        self.padding_trainable = True
        self.context_stride = 1

        self.input_size = [self.input_row, 23]
        offset_lod = [[0, 4, 5, 8, self.input_row]]
        self.lod = [[]]
        # convert from offset-based lod to length-based lod
        for i in range(len(offset_lod[0]) - 1):
            self.lod[0].append(offset_lod[0][i + 1] - offset_lod[0][i])
        self.output_represention = 8  # output feature size


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
        offset_lod = [[0] + np.sort(random.sample(idx, 8)).tolist() +
                      [self.input_size[0]]]
        self.lod = [[]]
        # convert from offset-based lod to length-based lod
        for i in range(len(offset_lod[0]) - 1):
            self.lod[0].append(offset_lod[0][i + 1] - offset_lod[0][i])
        self.output_represention = 8  # output feature size


if __name__ == '__main__':
    unittest.main()
