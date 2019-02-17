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
import unittest
import paddle.fluid as fluid
import paddle.fluid.core as core

os.environ['FLAGS_eager_delete_tensor_gb'] = "0.0"
os.environ[
    'RECORDIO_FILENAME'] = '/tmp/ir_memory_optimize_transformer.wmt16.recordio'

from test_parallel_executor_transformer import TestTransformer
from test_parallel_executor_transformer import transformer


# NOTE(dzhwinter): test diferent strategy colisions.
# open the eager delete tensor strategy by default.
class TestTransformerWithIR(TestTransformer):
    def test_main(self):
        if core.is_compiled_with_cuda():
            # check python transpiler
            self.check_network_convergence(
                transformer,
                use_cuda=True,
                memory_opt=True,
                use_ir_memory_optimize=False)
            # check IR memory optimize
            self.check_network_convergence(
                transformer,
                use_cuda=True,
                memory_opt=False,
                use_ir_memory_optimize=True)


if __name__ == '__main__':
    unittest.main()
