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

import os
flag_name = os.path.splitext(__file__)[0]


class TestDistCTR2x2(TestDistBase):
    def _setup_config(self):
        self._sync_mode = True
        self._enforce_place = "CPU"

    def test_dist_ctr(self):
        self.check_with_place(
            "dist_ctr.py", delta=1e-2, check_error_log=True, log_name=flag_name)


class TestDistCTRWithL2Decay2x2(TestDistBase):
    def _setup_config(self):
        self._sync_mode = True
        self._enforce_place = "CPU"

    def test_dist_ctr(self):
        need_envs = {"USE_L2_DECAY": "1"}
        self.check_with_place(
            "dist_ctr.py",
            delta=1e-7,
            check_error_log=True,
            need_envs=need_envs,
            log_name=flag_name)


@unittest.skip(reason="Skip unstable ci")
class TestDistCTR2x2_ASYNC(TestDistBase):
    def _setup_config(self):
        self._sync_mode = False
        self._hogwild_mode = True
        self._enforce_place = "CPU"

    def test_dist_ctr(self):
        need_envs = {
            "FLAGS_communicator_send_queue_size": "2",
            "FLAGS_communicator_max_merge_var_num": "2",
            "FLAGS_communicator_max_send_grad_num_before_recv": "2",
        }

        self.check_with_place(
            "dist_ctr.py",
            delta=100,
            check_error_log=True,
            need_envs=need_envs,
            log_name=flag_name)


@unittest.skip(reason="Skip unstable ci")
class TestDistCTR2x2_ASYNCWithLRDecay2x2(TestDistBase):
    def _setup_config(self):
        self._sync_mode = False
        self._hogwild_mode = True
        self._enforce_place = "CPU"

    def test_dist_ctr(self):
        need_envs = {
            "FLAGS_communicator_send_queue_size": "2",
            "FLAGS_communicator_max_merge_var_num": "2",
            "FLAGS_communicator_max_send_grad_num_before_recv": "2",
            "LR_DECAY": "1"
        }

        self.check_with_place(
            "dist_ctr.py",
            delta=100,
            check_error_log=True,
            need_envs=need_envs,
            log_name=flag_name)


@unittest.skip(reason="Skip unstable ci")
class TestDistCTR2x2_ASYNC2(TestDistBase):
    def _setup_config(self):
        self._sync_mode = False
        self._hogwild_mode = True
        self._enforce_place = "CPU"

    def test_dist_ctr(self):
        need_envs = {
            "FLAGS_communicator_send_queue_size": "2",
            "FLAGS_communicator_max_merge_var_num": "2",
            "FLAGS_communicator_max_send_grad_num_before_recv": "2",
            "FLAGS_communicator_independent_recv_thread": "0",
            "FLAGS_communicator_is_sgd_optimizer": "0"
        }

        self.check_with_place(
            "dist_ctr.py",
            delta=100,
            check_error_log=True,
            need_envs=need_envs,
            log_name=flag_name)


if __name__ == "__main__":
    unittest.main()
