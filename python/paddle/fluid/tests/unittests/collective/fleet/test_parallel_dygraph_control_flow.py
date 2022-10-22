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

import os
import unittest

import paddle.fluid as fluid
from test_dist_base import TestDistBase

flag_name = os.path.splitext(__file__)[0]


class TestDygraphControlFlowSame(TestDistBase):

    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._find_unused_parameters = True

    def test_net(self):
        if fluid.core.is_compiled_with_cuda():
            self.check_with_place("parallel_dygraph_control_flow_same.py",
                                  delta=1e-5,
                                  check_error_log=True,
                                  log_name=flag_name)


class TestFleetDygraphControlFlowSame(TestDygraphControlFlowSame):

    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._use_fleet_api = True
        self._find_unused_parameters = True


class TestFleetDygraphControlFlowSameAccGrad(TestDygraphControlFlowSame):

    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._accumulate_gradient = True
        self._find_unused_parameters = True


class TestDygraphControlFlowDiff(TestDistBase):

    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._find_unused_parameters = True

    def test_net(self):
        if fluid.core.is_compiled_with_cuda():
            self.check_with_place("parallel_dygraph_control_flow_different.py",
                                  delta=1e-5,
                                  check_error_log=True,
                                  log_name=flag_name)


class TestFleetDygraphControlFlowDiff(TestDygraphControlFlowDiff):

    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._use_fleet_api = True
        self._find_unused_parameters = True


class TestFleetDygraphControlFlowDiffAccGrad(TestDygraphControlFlowDiff):

    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._accumulate_gradient = True
        self._find_unused_parameters = True


if __name__ == "__main__":
    unittest.main()
