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

from __future__ import print_function
import unittest
from test_dist_base import TestDistBase

import os
import subprocess
flag_name = os.path.splitext(__file__)[0]


class TestDistMnistNCCL2DGC(TestDistBase):
    def _setup_config(self):
        self._sync_mode = True
        self._use_reduce = False
        self._use_reader_alloc = False
        self._nccl2_mode = True
        self._use_dgc = True

    def test_dist_train(self):
        import paddle.fluid as fluid
        if fluid.core.is_compiled_with_cuda():
            self.check_with_place(
                "dist_mnist.py",
                delta=1e-5,
                check_error_log=True,
                log_name=flag_name)

    def tearDown(self):
        # Used to determine if sparse communication is used
        cmd = 'grep sparse_all_reduce test_dist_mnist_dgc_nccl_tr0_err.log' \
              ' | grep in_numel | wc -l'
        child = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        result = child.communicate()[0]
        # only 1 layer use dgc now, run_step=5, rampup_begin_step=2, so 1 * (5 - 2) = 3
        self.assertEqual(result, '3\n')


class TestDistMnistNCCL2DGCMultiCards(TestDistBase):
    def _setup_config(self):
        self._sync_mode = True
        self._use_reduce = False
        self._use_reader_alloc = False
        self._nccl2_mode = True
        self._use_dgc = True

    def test_dist_train(self):
        import paddle.fluid as fluid
        if fluid.core.is_compiled_with_cuda():
            self.check_with_place_multi_cards(
                "dist_mnist.py",
                delta=1e-5,
                check_error_log=True,
                log_name=flag_name)

    def tearDown(self):
        # Used to determine if sparse communication is used
        cmd = 'grep sparse_all_reduce test_dist_mnist_dgc_nccl_dgc_2cards_local.log' \
              ' | grep in_numel | wc -l'
        child = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        result = child.communicate()[0]
        # same as above, but use two cards
        self.assertEqual(result, '6\n')


if __name__ == "__main__":
    unittest.main()
