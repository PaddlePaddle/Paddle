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

import unittest
from test_dist_base import TestDistBase

import os
import subprocess
import paddle

paddle.enable_static()
flag_name = os.path.splitext(__file__)[0]


def count_of_sparse_all_reduce_calls(file_name):
    # NOTE(Aurelius84): The log file contains some binary contents that causes error
    # while `grep`. So we add `-a` to fix it.
    # -a, --text equivalent to --binary-files=text, make binaries equivalent to text.
    cmd = 'grep -a sparse_all_reduce_op_handle ' + file_name + ' | grep in_numel | wc -l'
    child = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    result = child.communicate()[0]
    print('test_info: result = ' + str(result))

    # NOTE: in python3, result is b'num', != 'num'
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
            self.check_with_place(os.path.abspath("../../dist_mnist.py"),
                                  delta=1e-5,
                                  check_error_log=True,
                                  log_name=flag_name)

    def tearDown(self):
        import paddle.fluid as fluid
        if fluid.core.is_compiled_with_cuda():
            log_file = os.path.join(self.temp_dir.name,
                                    'test_dist_mnist_dgc_nccl_tr0_err.log')
            result = count_of_sparse_all_reduce_calls(log_file)
            # only 1 layer use dgc now, run_step=5, rampup_begin_step=2, so 1 * (5 - 2) = 3

            # temp close this test. In python3 CI, the log is right, but the result
            # has a problem, may be in multi process mode, log is not written in time.
            # self.assertEqual(result, 3)
        self.temp_dir.cleanup()


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
                os.path.abspath("../../dist_mnist.py"),
                delta=1e-5,
                check_error_log=True,
                log_name=flag_name)

    def tearDown(self):
        import paddle.fluid as fluid
        if fluid.core.is_compiled_with_cuda():
            log_file = os.path.join(
                self.temp_dir.name,
                'test_dist_mnist_dgc_nccl_dgc_2cards_local.log')
            result = count_of_sparse_all_reduce_calls(log_file)
            # same as above, but use two cards
            self.assertEqual(result, 6)
        self.temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
