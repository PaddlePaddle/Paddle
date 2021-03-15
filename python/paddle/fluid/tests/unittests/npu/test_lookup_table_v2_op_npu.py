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


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestLookupTableV2(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "lookup_table_v2"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        bsz=2
        seqlen=2
        vocab=3
        dim=2
        w = np.ones([vocab, dim]).astype(self.dtype)
        x = np.random.randint(0, vocab, size=(bsz, seqlen)).astype(np.int64)
        out = np.ones([bsz, seqlen, dim]).astype(self.dtype)

        self.inputs = {'W': OpTest.np_dtype_to_fluid_dtype(w), 'Ids': OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {
            'is_sparse': False,
            'is_distributed': False,
            'remote_prefetch':False,
            'padding_idx': -1
        }
        self.outputs = {'Out': out}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

    # TODO(ascendrc): Add grad test
    # def test_check_grad(self):
    #     if self.dtype == np.float16:
    #         return
    #     self.check_grad(['X'], 'Out')

@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestLookupTableV2FP16(TestLookupTableV2):
    no_need_check_grad = True
    def init_dtype(self):
        self.dtype = np.float16
    
#@unittest.skipIf(not paddle.is_compiled_with_npu(),
#                 "core is not compiled with NPU")
#class TestLookupTableV2Int8(TestLookupTableV2):
#    def init_dtype(self):
#        self.dtype = np.int8
#
#@unittest.skipIf(not paddle.is_compiled_with_npu(),
#                 "core is not compiled with NPU")
#class TestLookupTableV2UInt8(TestLookupTableV2):
#    def init_dtype(self):
#        self.dtype = np.uint8


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestLookupTableV2Net(unittest.TestCase):
    def _test(self, run_npu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        bsz=3
        seqlen=2
        vocab=3
        dim=2

        ids_np = np.random.randint(0, vocab, size=(bsz, seqlen)).astype('int64')

        with paddle.static.program_guard(main_prog, startup_prog):
            emb = paddle.nn.Embedding(vocab, dim)
            ids = paddle.static.data(name="ids", shape=[bsz, seqlen], dtype='int64')
            res = emb(ids)
            loss = res.sum()

        if run_npu:
            place = paddle.NPUPlace(0)
        else:
            place = paddle.CPUPlace()

        exe = paddle.static.Executor(place)
        exe.run(startup_prog)

        for epoch in range(1):
            loss_res, w = exe.run(
                main_prog,
                feed={"ids": ids_np},
                fetch_list=[loss, emb.weight])
            if epoch % 10 == 0:
                print(w)
                print("Epoch {} | Loss: {}".format(epoch, loss))

        return loss_res

    def test_npu(self):
        cpu_loss = self._test(False)
        npu_loss = self._test(True)
        self.assertTrue(np.allclose(npu_loss, cpu_loss))



if __name__ == '__main__':
    unittest.main()

