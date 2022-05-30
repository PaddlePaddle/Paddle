#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid

paddle.enable_static()
SEED = 2021


class TestLookupTableV2(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "lookup_table_v2"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        self.init_dims()
        self.init_padding_idx()
        np.random.seed(SEED)
        w = np.random.random([self.vocab, self.dim]).astype(self.dtype)
        x = np.random.randint(
            0, self.vocab, size=(self.bsz, self.seqlen)).astype(self.ids_dtype)
        out = w[x]
        if self.padding_idx != -1:
            out[np.squeeze(x == self.padding_idx)] = np.zeros(self.dim)

        self.inputs = {
            'W': OpTest.np_dtype_to_fluid_dtype(w),
            'Ids': OpTest.np_dtype_to_fluid_dtype(x)
        }
        self.attrs = {
            'is_sparse': False,
            'is_distributed': False,
            'remote_prefetch': False,
            'padding_idx': self.padding_idx
        }
        self.outputs = {'Out': out}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32
        self.ids_dtype = np.int32

    def init_dims(self):
        self.bsz = 6
        self.seqlen = 8
        self.vocab = 10
        # embedding_dim is not multiple of 32
        self.dim = 20

    def init_padding_idx(self):
        self.padding_idx = -1

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place, ['W'], 'Out', max_relative_error=0.01)
        else:
            self.check_grad_with_place(self.place, ['W'], 'Out')


class TestLookupTableV2FP16(TestLookupTableV2):
    no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float16
        self.ids_dtype = np.int32

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True


class TestLookupTableV2Dim32(TestLookupTableV2):
    def init_dims(self):
        self.bsz = 6
        self.seqlen = 8
        self.vocab = 10
        # embedding_dim is multiple of 32
        self.dim = 64


class TestLookupTableV2Dim32FP16(TestLookupTableV2):
    no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float16
        self.ids_dtype = np.int64

    def init_dims(self):
        self.bsz = 6
        self.seqlen = 8
        self.vocab = 10
        self.dim = 64

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True


class TestLookupTableV2WithPadding(TestLookupTableV2):
    def init_padding_idx(self):
        self.padding_idx = np.random.randint(0, self.vocab)


class TestLookupTableV2WithPadding1(TestLookupTableV2):
    def init_padding_idx(self):
        self.padding_idx = np.random.randint(0, self.vocab)

    def init_dtype(self):
        self.dtype = np.float32
        self.ids_dtype = np.int64


if __name__ == '__main__':
    unittest.main()
