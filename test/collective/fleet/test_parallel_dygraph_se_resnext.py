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

from parallel_dygraph_se_resnext import TestSeResNeXt
from spawn_runner_base import TestDistSpawnRunner
from test_dist_base import TestDistBase

from paddle import base

flag_name = os.path.splitext(__file__)[0]


class TestParallelDygraphSeResNeXt(TestDistBase):
    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True

    def test_se_resnext(self):
        if base.core.is_compiled_with_cuda():
            self.check_with_place(
                "parallel_dygraph_se_resnext.py",
                delta=0.01,
                check_error_log=True,
                log_name=flag_name,
            )


class TestParallelDygraphSeResNeXtSpawn(TestDistSpawnRunner):
    def test_se_resnext_with_spawn(self):
        if base.core.is_compiled_with_cuda():
            self.check_dist_result_with_spawn(
                test_class=TestSeResNeXt, delta=0.01
            )


if __name__ == "__main__":
    unittest.main()
