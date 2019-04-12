# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import unittest
from timeit import default_timer as timer
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.dataset.wmt16 as wmt16

os.environ['FLAGS_eager_delete_tensor_gb'] = "0.0"
os.environ[
    'RECORDIO_FILENAME'] = '/tmp/ir_memory_optimize_transformer.wmt16.recordio'

from test_parallel_executor_transformer import transformer, ModelHyperParams, transformer_model, transformer, prepare_batch_input
from parallel_executor_test_base import TestParallelExecutorBase


# NOTE(dzhwinter): test diferent strategy colisions.
# open the eager delete tensor strategy by default.
class TestTransformerWithIR(TestParallelExecutorBase):
    @classmethod
    def setUpClass(cls):
        os.environ['CPU_NUM'] = str(4)
        reader = paddle.batch(
            wmt16.train(ModelHyperParams.src_vocab_size,
                        ModelHyperParams.trg_vocab_size),
            batch_size=transformer_model.batch_size)

        with fluid.recordio_writer.create_recordio_writer(
                os.environ.get("RECORDIO_FILENAME")) as writer:
            for batch in reader():
                for tensor in prepare_batch_input(
                        batch, ModelHyperParams.src_pad_idx,
                        ModelHyperParams.trg_pad_idx, ModelHyperParams.n_head):
                    t = fluid.LoDTensor()
                    t.set(tensor, fluid.CPUPlace())
                    writer.append_tensor(t)
                writer.complete_append_tensor()

    def test_main(self):
        if core.is_compiled_with_cuda():
            # check python transpiler
            self.check_network_convergence(
                transformer,
                use_cuda=True,
                memory_opt=True,
                use_ir_memory_optimize=False,
                iter=2)
            # check IR memory optimize
            self.check_network_convergence(
                transformer,
                use_cuda=True,
                memory_opt=False,
                use_ir_memory_optimize=True,
                iter=2)


if __name__ == '__main__':
    unittest.main()
