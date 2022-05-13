# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import unittest

import paddle.fluid as fluid
from test_dist_base import TestDistBase
from spawn_runner_base import TestDistSpawnRunner
from parallel_dygraph_no_sync import TestNoSync
from parallel_dygraph_no_sync_unused_params import TestNoSyncUnusedParam
from parallel_dygraph_no_sync_control_flow import TestNoSyncControlFlow

flag_name = os.path.splitext(__file__)[0]


class TestParallelDygraphNoSync(TestDistBase):
    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._find_unused_parameters = False

    def test_no_sync(self):
        if fluid.core.is_compiled_with_cuda():
            self.check_with_place(
                "parallel_dygraph_no_sync.py",
                delta=1e-5,
                check_error_log=True,
                log_name=flag_name)


class TestParallelDygraphNoSyncUnusedParam(TestDistBase):
    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._find_unused_parameters = True

    def test_no_sync_ununsed_param(self):
        if fluid.core.is_compiled_with_cuda():
            self.check_with_place(
                "parallel_dygraph_no_sync_unused_params.py",
                delta=1e-5,
                check_error_log=True,
                log_name=flag_name)


class TestParallelDygraphNoSyncControlFlow(TestDistBase):
    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._find_unused_parameters = True

    def test_no_sync_control_flow(self):
        if fluid.core.is_compiled_with_cuda():
            self.check_with_place(
                "parallel_dygraph_no_sync_control_flow.py",
                delta=1e-5,
                check_error_log=True,
                log_name=flag_name)


class TestParallelDygraphNoSyncSpawn(TestDistSpawnRunner):
    def test_no_sync_with_spawn(self):
        if fluid.core.is_compiled_with_cuda() and sys.version_info >= (3, 4):
            self.check_dist_result_with_spawn(test_class=TestNoSync, delta=1e-5)


class TestParallelDygraphNoSyncUnusedParamSpawn(TestDistSpawnRunner):
    def _args_config(self, args):
        args.find_unused_parameters = True

    def test_no_sync_with_spawn(self):
        if fluid.core.is_compiled_with_cuda() and sys.version_info >= (3, 4):
            self.check_dist_result_with_spawn(
                test_class=TestNoSyncUnusedParam, delta=1e-5)


class TestParallelDygraphNoSyncControlFlowSpawn(TestDistSpawnRunner):
    def _args_config(self, args):
        args.find_unused_parameters = True

    def test_no_sync_with_spawn(self):
        if fluid.core.is_compiled_with_cuda() and sys.version_info >= (3, 4):
            self.check_dist_result_with_spawn(
                test_class=TestNoSyncControlFlow, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
