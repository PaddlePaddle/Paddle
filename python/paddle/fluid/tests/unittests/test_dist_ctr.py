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


def skip_ci(func):
    on_ci = bool(int(os.environ.get("SKIP_UNSTABLE_CI", '0')))

    def __func__(*args, **kwargs):
        if on_ci:
            return
        return func(*args, **kwargs)

    return __func__


@skip_ci
class TestDistCTR2x2(TestDistBase):
    def _setup_config(self):
        self._sync_mode = True
        self._enforce_place = "CPU"

    def test_dist_ctr(self):
        self.check_with_place("dist_ctr.py", delta=1e-7, check_error_log=False)


@skip_ci
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
            need_envs=need_envs)


if __name__ == "__main__":
    unittest.main()
