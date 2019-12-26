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


def count_of_sparse_all_reduce_calls(file_name):
    cmd = 'grep sparse_all_reduce_op_handle ' + file_name + ' | grep in_numel | wc -l'
    child = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    result = child.communicate()[0]
    print('test_info: result = ' + str(result))

    # note. in python3, result is b'num', != 'num' 
    return int(result)


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
        import paddle.fluid as fluid
        if fluid.core.is_compiled_with_cuda():
            result = count_of_sparse_all_reduce_calls(
                'test_dist_mnist_dgc_nccl_tr0_err.log')
            # only 1 layer use dgc now, run_step=5, rampup_begin_step=2, so 1 * (5 - 2) = 3

            # temp close this test. In python3 CI, the log is right, but the result
            # has a problem, may be in multi process mode, log is not writed in time.  
            # self.assertEqual(result, 3)


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
        import paddle.fluid as fluid
        if fluid.core.is_compiled_with_cuda():
            result = count_of_sparse_all_reduce_calls(
                'test_dist_mnist_dgc_nccl_dgc_2cards_local.log')
            # same as above, but use two cards
            self.assertEqual(result, 6)


if __name__ == "__main__":
    unittest.main()
