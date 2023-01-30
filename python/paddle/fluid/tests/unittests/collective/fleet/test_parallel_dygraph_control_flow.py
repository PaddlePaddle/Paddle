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

<<<<<<< HEAD
import os
import unittest

from test_dist_base import TestDistBase

import paddle.fluid as fluid
=======
from __future__ import print_function

import os
import sys
import unittest

import paddle.fluid as fluid
from test_dist_base import TestDistBase
from spawn_runner_base import TestDistSpawnRunner
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

flag_name = os.path.splitext(__file__)[0]


class TestDygraphControlFlowSame(TestDistBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._find_unused_parameters = True

    def test_net(self):
        if fluid.core.is_compiled_with_cuda():
<<<<<<< HEAD
            self.check_with_place(
                "parallel_dygraph_control_flow_same.py",
                delta=1e-5,
                check_error_log=True,
                log_name=flag_name,
            )


class TestFleetDygraphControlFlowSame(TestDygraphControlFlowSame):
=======
            self.check_with_place("parallel_dygraph_control_flow_same.py",
                                  delta=1e-5,
                                  check_error_log=True,
                                  log_name=flag_name)


class TestFleetDygraphControlFlowSame(TestDygraphControlFlowSame):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._use_fleet_api = True
        self._find_unused_parameters = True


class TestFleetDygraphControlFlowSameAccGrad(TestDygraphControlFlowSame):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._accumulate_gradient = True
        self._find_unused_parameters = True


class TestDygraphControlFlowDiff(TestDistBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._find_unused_parameters = True

    def test_net(self):
        if fluid.core.is_compiled_with_cuda():
<<<<<<< HEAD
            self.check_with_place(
                "parallel_dygraph_control_flow_different.py",
                delta=1e-5,
                check_error_log=True,
                log_name=flag_name,
            )


class TestFleetDygraphControlFlowDiff(TestDygraphControlFlowDiff):
=======
            self.check_with_place("parallel_dygraph_control_flow_different.py",
                                  delta=1e-5,
                                  check_error_log=True,
                                  log_name=flag_name)


class TestFleetDygraphControlFlowDiff(TestDygraphControlFlowDiff):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._use_fleet_api = True
        self._find_unused_parameters = True


class TestFleetDygraphControlFlowDiffAccGrad(TestDygraphControlFlowDiff):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._accumulate_gradient = True
        self._find_unused_parameters = True


if __name__ == "__main__":
    unittest.main()
