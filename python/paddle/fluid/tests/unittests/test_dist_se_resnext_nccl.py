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

import unittest
from test_dist_base import TestDistBase
import os

import os
import paddle

paddle.enable_static()
flag_name = os.path.splitext(__file__)[0]


class TestDistSeResneXtNCCL(TestDistBase):

    def _setup_config(self):
        self._sync_mode = True
        self._use_reader_alloc = False
        self._nccl2_mode = True

    def test_dist_train(self):
        import paddle.fluid as fluid
        if fluid.core.is_compiled_with_cuda():
            self.check_with_place("dist_se_resnext.py",
                                  delta=1e-5,
                                  check_error_log=True,
                                  log_name=flag_name)


class TestDistSeResneXtNCCLMP(TestDistBase):

    def _setup_config(self):
        self._sync_mode = True
        self._use_reader_alloc = False
        self._nccl2_mode = True
        self._mp_mode = True

    def test_dist_train(self):
        import paddle.fluid as fluid
        if fluid.core.is_compiled_with_cuda():
            self.check_with_place("dist_se_resnext.py",
                                  delta=1e-5,
                                  check_error_log=True,
                                  need_envs={"NCCL_P2P_DISABLE": "1"},
                                  log_name=flag_name)


if __name__ == "__main__":
    unittest.main()
