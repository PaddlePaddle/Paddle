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

import time
import unittest
import numpy as np
from paddle.fluid.framework import Program

import paddle.fluid.core as core
import paddle.fluid as fluid


class TestFetchHandler(unittest.TestCase):

    @unittest.skip(reason="Skip unstable ci")
    def test_fetch_handler(self):
        place = core.CPUPlace()
        scope = core.Scope()

        table = np.random.random((3, 10)).astype("float32")

        prog = Program()
        block = prog.current_block()
        var_emb = block.create_var(name='emb', type=core.VarDesc.VarType.FP32)
        var_emb3 = block.create_var(name='emb3', type=core.VarDesc.VarType.FP32)

        class FH(fluid.executor.FetchHandler):

            def handler(self, fetch_dict):
                assert len(fetch_dict) == 1

        table_var = scope.var('emb').get_tensor()
        table_var.set(table, place)
        fh = FH(var_dict={'emb': var_emb}, period_secs=2)
        fm = fluid.trainer_factory.FetchHandlerMonitor(scope, fh)

        fm.start()
        time.sleep(3)
        fm.stop()

        default_fh = fluid.executor.FetchHandler(var_dict={
            'emb': var_emb,
            'emb2': None,
            'emb3': var_emb3
        },
                                                 period_secs=1)
        default_fm = fluid.trainer_factory.FetchHandlerMonitor(
            scope, default_fh)
        default_fm.start()
        time.sleep(5)
        default_fm.stop()


if __name__ == "__main__":
    unittest.main()
