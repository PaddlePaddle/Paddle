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

import os
import unittest

from test_dist_base import TestDistBase


class TestDistSimnetBowDense2x2(TestDistBase):
    def _setup_config(self):
        self._sync_mode = True
        self._enforce_place = "CPU"

    def test_simnet_bow(self):
        need_envs = {
            "IS_DISTRIBUTED": '0',
            "IS_SPARSE": '0',
            'IS_SELF_CONTAINED_LR': '1'
        }
        self.check_with_place(
            "dist_simnet_bow.py",
            delta=1e-5,
            check_error_log=False,
            need_envs=need_envs)


class TestDistSimnetBow2x2DenseAsync(TestDistBase):
    def _setup_config(self):
        self._sync_mode = False
        self._enforce_place = "CPU"

    #FIXME(typhoonzero): fix async tests later
    def notest_simnet_bow(self):
        need_envs = {
            "IS_DISTRIBUTED": '0',
            "IS_SPARSE": '0',
            'IS_SELF_CONTAINED_LR': '1',
        }
        self.check_with_place(
            "dist_simnet_bow.py",
            delta=100,
            check_error_log=False,
            need_envs=need_envs)


class TestDistSimnetBowSparse2x2(TestDistBase):
    def _setup_config(self):
        self._sync_mode = True
        self._enforce_place = "CPU"

    def test_simnet_bow(self):
        need_envs = {
            "IS_DISTRIBUTED": '0',
            "IS_SPARSE": '1',
            'IS_SELF_CONTAINED_LR': '1'
        }
        self.check_with_place(
            "dist_simnet_bow.py",
            delta=1e-5,
            check_error_log=False,
            need_envs=need_envs)


class TestDistSimnetBow2x2SparseAsync(TestDistBase):
    def _setup_config(self):
        self._sync_mode = False
        self._enforce_place = "CPU"

    def test_simnet_bow(self):
        need_envs = {
            "IS_DISTRIBUTED": '0',
            "IS_SPARSE": '1',
            'IS_SELF_CONTAINED_LR': '1'
        }
        self.check_with_place(
            "dist_simnet_bow.py",
            delta=100,
            check_error_log=False,
            need_envs=need_envs)


# FIXME(tangwei): Learningrate variable is not created on pserver.
class TestDistSimnetBow2x2LookupTableSync(TestDistBase):
    def _setup_config(self):
        self._sync_mode = True
        self._enforce_place = "CPU"

    def test_simnet_bow(self):
        need_envs = {
            "IS_DISTRIBUTED": '1',
            "IS_SPARSE": '1',
            'IS_SELF_CONTAINED_LR': '1'
        }
        self.check_with_place(
            "dist_simnet_bow.py",
            delta=1e-5,
            check_error_log=True,
            need_envs=need_envs)


class TestDistSimnetBow2x2LookupTableAsync(TestDistBase):
    def _setup_config(self):
        self._sync_mode = False
        self._enforce_place = "CPU"

    def test_simnet_bow(self):
        need_envs = {
            "IS_DISTRIBUTED": '1',
            "IS_SPARSE": '1',
            'IS_SELF_CONTAINED_LR': '1'
        }
        self.check_with_place(
            "dist_simnet_bow.py",
            delta=100,
            check_error_log=False,
            need_envs=need_envs)


class TestDistSimnetBow2x2LookupTableNotContainLRSync(TestDistBase):
    def _setup_config(self):
        self._sync_mode = True
        self._enforce_place = "CPU"

    def test_simnet_bow(self):
        need_envs = {
            "IS_DISTRIBUTED": '1',
            "IS_SPARSE": '1',
            'IS_SELF_CONTAINED_LR': '0'
        }
        self.check_with_place(
            "dist_simnet_bow.py",
            delta=1e-5,
            check_error_log=False,
            need_envs=need_envs)


if __name__ == "__main__":
    unittest.main()
