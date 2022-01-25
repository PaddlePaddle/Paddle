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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
from test_reorder_lod_tensor import convert_to_offset
from sequence.test_sequence_pool import compute_seqpool_sum, compute_seqpool_avg, compute_seqpool_sqrt
from test_cvm_op import cvm_compute
import paddle


class TestFusedSeqPoolCVMOp(OpTest):
    def setUp(self):
        self.w = 11
        self.use_cvm = True
        self.lods = [[[23]], [[14]]]
        self.set_conf()
        self.set_pooltype()
        self.op_type = 'fused_seqpool_cvm'
        bs = len(self.lods[0][0])
        inputs = []
        outs = []
        # The cvm variable is not actually used.
        cvm = np.array([[1, 0]]).astype("float32")
        i = 0
        for lod in self.lods:
            assert bs == len(lod[0]), 'All lod size should be equal'
            x = np.random.uniform(0.1, 1,
                                  [sum(lod[0]), self.w]).astype('float32')
            offset = convert_to_offset(lod)
            # lod: [[23]] ====offset: [[0, 23]] ===x: (23, 11)
            out = np.zeros((bs, self.w)).astype('float32')
            if self.pooltype == "SUM":
                compute_seqpool_sum(x, offset, out)
                out = cvm_compute(out, self.w, self.use_cvm)
            else:
                raise Exception("Unsupported pool type!")
            inputs.append(('x_{0}'.format(i), (x, lod)))
            outs.append(('out_{0}'.format(i), out))
            i = i + 1

        self.inputs = {'X': inputs, "CVM": cvm}
        self.outputs = {'Out': outs}
        self.attrs = {
            'pooltype': self.pooltype,
            'use_cvm': self.use_cvm,
            'pad_value': 0.0,
            'cvm_offset': 2,
        }

    def set_pooltype(self):
        self.pooltype = "SUM"

    def set_conf(self):
        pass

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestFusedSeqPoolCVMOpCase1(TestFusedSeqPoolCVMOp):
    def set_conf(self):
        self.use_cvm = False


class TestFusedSeqPoolCVMOpAPI(unittest.TestCase):
    def test_api(self):
        paddle.enable_static()
        show = paddle.fluid.layers.data(name="show", shape=[-1, 1], \
            dtype="int64", lod_level=1, append_batch_size=False)
        ones = paddle.fluid.layers.fill_constant_batch_size_like(input=show,\
            shape=[-1, 1], dtype="float32", value=1)
        show_clk = paddle.fluid.layers.cast(paddle.fluid.layers.concat([ones, ones],\
            axis=1), dtype='float32')
        emb = paddle.fluid.layers.embedding(input=show, size=[1, 1], \
            is_sparse=True, is_distributed=True, \
            param_attr=paddle.fluid.ParamAttr(name="embedding"))
        out1 = paddle.fluid.contrib.layers.fused_seqpool_cvm(
            [emb], 'sum', show_clk, use_cvm=True)

    def test_invalid_type(self):
        paddle.enable_static()
        show = paddle.fluid.layers.data(name="show", shape=[-1, 1], \
            dtype="int64", lod_level=1, append_batch_size=False)
        ones = paddle.fluid.layers.fill_constant_batch_size_like(input=show,\
            shape=[-1, 1], dtype="float32", value=1)
        show_clk = paddle.fluid.layers.cast(paddle.fluid.layers.concat([ones, ones],\
            axis=1), dtype='float32')
        emb = paddle.fluid.layers.embedding(input=show, size=[1, 1], \
            is_sparse=True, is_distributed=True, \
            param_attr=paddle.fluid.ParamAttr(name="embedding"))
        with self.assertRaises(TypeError):
            out1 = paddle.fluid.contrib.layers.fused_seqpool_cvm(
                [emb], 'average', show_clk, use_cvm=True)

    def test_invalid_input(self):
        paddle.enable_static()
        show = paddle.fluid.layers.data(name="show", shape=[-1, 1], \
            dtype="int64", lod_level=1, append_batch_size=False)
        ones = paddle.fluid.layers.fill_constant_batch_size_like(input=show,\
            shape=[-1, 1], dtype="float32", value=1)
        show_clk = paddle.fluid.layers.cast(paddle.fluid.layers.concat([ones, ones],\
            axis=1), dtype='float32')
        emb = paddle.fluid.layers.embedding(input=show, size=[1, 1], \
            is_sparse=True, is_distributed=True, \
            param_attr=paddle.fluid.ParamAttr(name="embedding"))
        with self.assertRaises(TypeError):
            out1 = paddle.fluid.contrib.layers.fused_seqpool_cvm(
                emb, 'sum', show_clk, use_cvm=True)

    def test_invalid_input_null(self):
        paddle.enable_static()
        show = paddle.fluid.layers.data(name="show", shape=[-1, 1], \
            dtype="int64", lod_level=1, append_batch_size=False)
        ones = paddle.fluid.layers.fill_constant_batch_size_like(input=show,\
            shape=[-1, 1], dtype="float32", value=1)
        show_clk = paddle.fluid.layers.cast(paddle.fluid.layers.concat([ones, ones],\
            axis=1), dtype='float32')
        emb = paddle.fluid.layers.embedding(input=show, size=[1, 1], \
            is_sparse=True, is_distributed=True, \
            param_attr=paddle.fluid.ParamAttr(name="embedding"))
        with self.assertRaises(TypeError):
            out1 = paddle.fluid.contrib.layers.fused_seqpool_cvm(
                [], 'sum', show_clk, use_cvm=True)


if __name__ == '__main__':
    unittest.main()
