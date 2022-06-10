#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
import sys
sys.path.append("..")
import paddle
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()
SEED = 2022

class XPUTestLookupTableV2Op(XPUOpTestWrapper):
    def __init__(self):
      self.op_name = "lookup_table_v2" 


    class TestLookupTableV2Op(XPUOpTest):
        def setUp(self):
            self.op_type = "lookup_table_v2"
            self.set_shape()
            self.init_dtype()
            self.init_data()
    
        def set_shape(self):
            self.table_shape = (17,31)
            self.ids_shape = (0,17,4)
   
        def init_data(self):
            table = np.random.random(self.table_shape).astype(self.dtype)
            ids = np.random.randint(0, 17, 4).astype(np.int32)
            self.inputs = {'W': table, 'Ids': ids}
            self.outputs = {'Out': table[ids]}

        def init_dtype(self):
            self.dtype = self.in_type  

        def test_check_output_with_place(self):
            self.check_output_with_place(place=paddle.XPUPlace(0))

        def test_check_grad(self):
            if self.dtype == np.float16:
              self.check_grad_with_place(self.place, ['W'], 'Out', max_relative_error=0.01)
            else:
              self.check_grad_with_place(self.place, ['W'], 'Out')

    class TestLookupTableV2FP16(TestLookupTableV2Op):
        no_need_check_grad = True

        def init_dtype(self):
          self.dtype = np.float16
          self.ids_dtype = np.int32

        def set_xpu(self):
          self.__class__.use_xpu = True
          self.__class__.no_need_check_grad = True


    class TestLookupTableV2Dim32(TestLookupTableV2Op):
        def init_dims(self):
            self.bsz = 6
            self.seqlen = 8
            self.vocab = 10
            # embedding_dim is multiple of 32
            self.dim = 64


    class TestLookupTableV2Dim32FP16(TestLookupTableV2Op):
        no_need_check_grad = True

        def init_dtype(self):
            self.dtype = np.float16
            self.ids_dtype = np.int64

        def init_dims(self):
            self.bsz = 6
            self.seqlen = 8
            self.vocab = 10
            self.dim = 64

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True


    class TestLookupTableV2WithPadding(TestLookupTableV2Op):
        def init_padding_idx(self):
            self.padding_idx = np.random.randint(0, self.vocab)


    class TestLookupTableV2WithPadding1(TestLookupTableV2Op):
        def init_padding_idx(self):
            self.padding_idx = np.random.randint(0, self.vocab)

        def init_dtype(self):
            self.dtype = np.float32
            self.ids_dtype = np.int64

support_types = get_xpu_op_support_types("lookup_table_v2")
for stype in support_types:
    create_test_class(globals(), XPUTestLookupTableV2Op, stype)

if __name__ == "__main__":
    unittest.main()
