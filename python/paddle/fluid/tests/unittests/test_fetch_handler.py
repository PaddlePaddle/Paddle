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

import time
import unittest
import numpy as np

import paddle.fluid.core as core
import paddle.fluid as fluid


class TestFetchHandler(unittest.TestCase):
    def test_fetch_handler(self):
        place = core.CPUPlace()
        scope = core.Scope()

        table = np.random.random((3, 10)).astype("float32")

        class FH(fluid.executor.FetchHandler):
            def handler(self, fetch_target_vars):
                assert len(fetch_target_vars) == 1

        table_var = scope.var('emb').get_tensor()
        table_var.set(table, place)

        fh = FH(['emb'], period_secs=2, return_np=True)
        fm = fluid.trainer_factory.FetchHandlerMonitor(scope, fh)

        fm.start()
        time.sleep(10)
        fm.stop()


if __name__ == "__main__":
    unittest.main()
