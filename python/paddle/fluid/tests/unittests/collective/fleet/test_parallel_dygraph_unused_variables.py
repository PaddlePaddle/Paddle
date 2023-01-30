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

<<<<<<< HEAD
import os
import unittest

from parallel_dygraph_unused_variables import TestSparseEmbeddingUnusedVars
from spawn_runner_base import TestDistSpawnRunner
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
from parallel_dygraph_unused_variables import TestSparseEmbeddingUnusedVars
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

flag_name = os.path.splitext(__file__)[0]


class TestParallelDygraphUnusedVar(TestDistBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True

    def test_net(self):
        if fluid.core.is_compiled_with_cuda():
            self.check_with_place(
                os.path.abspath("../../parallel_dygraph_unused_variables.py"),
                delta=1e-5,
                check_error_log=True,
<<<<<<< HEAD
                log_name=flag_name,
            )


class TestFleetDygraphUnusedVar(TestParallelDygraphUnusedVar):
=======
                log_name=flag_name)


class TestFleetDygraphUnusedVar(TestParallelDygraphUnusedVar):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._use_fleet_api = True


class TestSparseEmbeddingUnusedVarsSpawn(TestDistSpawnRunner):
<<<<<<< HEAD
    def test_mnist_with_spawn(self):
        if fluid.core.is_compiled_with_cuda():
            self.check_dist_result_with_spawn(
                test_class=TestSparseEmbeddingUnusedVars, delta=1e-5
            )


class TestParallelDygraphNoVar(TestDistBase):
=======

    def test_mnist_with_spawn(self):
        if fluid.core.is_compiled_with_cuda() and sys.version_info >= (3, 4):
            self.check_dist_result_with_spawn(
                test_class=TestSparseEmbeddingUnusedVars, delta=1e-5)


class TestParallelDygraphNoVar(TestDistBase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True

    def test_net(self):
        if fluid.core.is_compiled_with_cuda():
            self.check_with_place(
                os.path.abspath("../../parallel_dygraph_none_var.py"),
                delta=1e-5,
                check_error_log=True,
<<<<<<< HEAD
                log_name=flag_name,
            )


class TestParallelDygraphSharedUnusedVariables(TestDistBase):
=======
                log_name=flag_name)


class TestParallelDygraphSharedUnusedVariables(TestDistBase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True

    def test_mnist(self):
        if fluid.core.is_compiled_with_cuda():
            self.check_with_place(
                os.path.abspath("../../parallel_dygraph_shared_unused_var.py"),
                delta=1e-5,
                check_error_log=True,
<<<<<<< HEAD
                log_name=flag_name,
            )
=======
                log_name=flag_name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == "__main__":
    unittest.main()
