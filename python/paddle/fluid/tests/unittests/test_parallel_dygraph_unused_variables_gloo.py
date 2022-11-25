# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from test_dist_base import TestDistBase

flag_name = os.path.splitext(__file__)[0]


class TestParallelDygraphUnusedVar_GLOO(TestDistBase):

    def _setup_config(self):
        self._sync_mode = False
        self._gloo_mode = True
        self._dygraph = True

    def test_net(self):
<<<<<<< HEAD
        self.check_with_place(
            "parallel_dygraph_unused_variables.py",
            delta=1e-5,
            check_error_log=True,
            log_name=flag_name,
        )
=======
        self.check_with_place("parallel_dygraph_unused_variables.py",
                              delta=1e-5,
                              check_error_log=True,
                              log_name=flag_name)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf


class TestParallelDygraphNoVar_GLOO(TestDistBase):

    def _setup_config(self):
        self._sync_mode = False
        self._gloo_mode = True
        self._dygraph = True

    def test_net(self):
<<<<<<< HEAD
        self.check_with_place(
            "parallel_dygraph_none_var.py",
            delta=1e-5,
            check_error_log=True,
            log_name=flag_name,
        )
=======
        self.check_with_place("parallel_dygraph_none_var.py",
                              delta=1e-5,
                              check_error_log=True,
                              log_name=flag_name)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf


class TestParallelDygraphSharedUnusedVariables_GLOO(TestDistBase):

    def _setup_config(self):
        self._sync_mode = False
        self._gloo_mode = True
        self._dygraph = True

    def test_mnist(self):
<<<<<<< HEAD
        self.check_with_place(
            "parallel_dygraph_shared_unused_var.py",
            delta=1e-5,
            check_error_log=True,
            log_name=flag_name,
        )
=======
        self.check_with_place("parallel_dygraph_shared_unused_var.py",
                              delta=1e-5,
                              check_error_log=True,
                              log_name=flag_name)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf


if __name__ == "__main__":
    unittest.main()
