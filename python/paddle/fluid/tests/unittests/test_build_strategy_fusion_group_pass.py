#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
import paddle.fluid as fluid
import paddle.fluid.core as core
from test_eager_deletion_padding_rnn import RNNConfig, PaddingRNNTestBase


class FusionGroupPaddingRNNTest(PaddingRNNTestBase):

    def set_customed_config(self):
        self.build_strategy.enable_auto_fusion = True

        # Use CUDA executor
        if core.is_compiled_with_cuda():
            self.exe = fluid.Executor(fluid.CUDAPlace(0))

    def test_train_enable_fusion_group(self):
        rnn_model = "static"
        config = RNNConfig("test", rnn_model)
        with fluid.scope_guard(fluid.Scope()):
            self.train(config, parallel=True, use_program_cache=False)


if __name__ == '__main__':
    unittest.main()
