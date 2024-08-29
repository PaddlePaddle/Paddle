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

from legacy_test.test_dist_base import TestDistBase

from paddle import base

flag_name = os.path.splitext(__file__)[0]


class TestDistMnistGradMerge(TestDistBase):
    def _setup_config(self):
        self._sync_mode = True
        self._use_reduce = False
        self._nccl2_mode = True
        self._nccl2_reduce_layer = True

    def test_dist_train(self):
        if base.core.is_compiled_with_cuda():
            self.check_with_place(
                "dist_mnist_gradient_merge.py",
                delta=1e-5,
                check_error_log=True,
                log_name=flag_name,
            )


class TestDistMnistGradMergeNoFuse(TestDistBase):
    def _setup_config(self):
        self._sync_mode = True
        self._use_reduce = False
        self._nccl2_mode = True
        self._fuse_all_reduce = False
        self._nccl2_reduce_layer = True

    def test_dist_train(self):
        if base.core.is_compiled_with_cuda():
            self.check_with_place(
                "dist_mnist_gradient_merge.py",
                delta=1e-5,
                check_error_log=True,
                log_name=flag_name + "_no_fuse",
            )


class TestDistMnistGradMergeRawOptimizerBase(TestDistBase):
    def _setup_config(self):
        self._use_reader_alloc = False
        self._nccl2_mode = True
        self._use_fleet_api = True
        self._use_fleet_api_20 = True
        self._nccl2_reduce_layer = True

    def enable_avg(self):
        return False

    def test_dist_train(self):
        if base.core.is_compiled_with_cuda():
            avg = str(self.enable_avg())
            log_name = flag_name + "_raw_optimizer_gm_avg_" + avg
            self.check_with_place(
                "dist_mnist_gradient_merge_raw_optimizer.py",
                delta=1e-5,
                check_error_log=True,
                log_name=log_name,
                need_envs={
                    'FLAGS_apply_pass_to_program': '1',
                    'enable_gm_avg': avg,
                },
            )


class TestDistMnistGradMergeRawOptimizerAvg(
    TestDistMnistGradMergeRawOptimizerBase
):
    def enable_avg(self):
        return True


if __name__ == "__main__":
    unittest.main()
