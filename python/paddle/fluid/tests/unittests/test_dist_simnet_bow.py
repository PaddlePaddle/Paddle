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

import os
from __future__ import print_function
import unittest

from test_dist_base import TestDistBase


class TestDistSimnetBowDense2x2(TestDistBase):
    def _setup_config(self):
        self._sync_mode = True
        self._use_cuda = False

    def test_simnet_bow(self):
        os.environ['CPU_NUM'] = '1'
        os.environ["IS_DISTRIBUTED"] = '0'
        os.environ["IS_SPARSE"] = '0'
        self.check_with_place(
            "dist_simnet_bow.py", delta=1e-7, check_error_log=False)


class TestDistSimnetBow2x2DenseAsync(TestDistBase):
    def _setup_config(self):
        self._sync_mode = False
        self._use_cuda = False

    def test_simnet_bow(self):
        os.environ['CPU_NUM'] = '1'
        os.environ["IS_DISTRIBUTED"] = '0'
        os.environ["IS_SPARSE"] = '0'
        self.check_with_place(
            "dist_simnet_bow.py", delta=100, check_error_log=False)


class TestDistSimnetBowSparse2x2(TestDistBase):
    def _setup_config(self):
        self._sync_mode = True
        self._use_cuda = False

    def test_simnet_bow(self):
        os.environ['CPU_NUM'] = '1'
        os.environ["IS_DISTRIBUTED"] = '0'
        os.environ["IS_SPARSE"] = '1'
        self.check_with_place(
            "dist_simnet_bow.py", delta=1e-5, check_error_log=False)


class TestDistSimnetBow2x2SparseAsync(TestDistBase):
    def _setup_config(self):
        self._sync_mode = False
        self._use_cuda = False

    def test_simnet_bow(self):
        os.environ['CPU_NUM'] = '1'
        os.environ["IS_DISTRIBUTED"] = '0'
        os.environ["IS_SPARSE"] = '1'
        self.check_with_place(
            "dist_simnet_bow.py", delta=100, check_error_log=False)


if __name__ == "__main__":
    unittest.main()
