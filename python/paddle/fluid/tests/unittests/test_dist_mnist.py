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


class TestDistMnist2x2(TestDistBase):
    def _setup_config(self):
        self._sync_mode = True
        self._use_reduce = False

    def test_dist_train(self):
        self.check_with_place("dist_mnist.py", delta=1e-5)


class TestDistMnistNCCL2(TestDistBase):
    def _setup_config(self):
        self._sync_mode = True
        self._use_reduce = False
        self._use_reader_alloc = False
        self._nccl2_mode = True

    def test_dist_train(self):
        import paddle.fluid as fluid
        if fluid.core.is_compiled_with_cuda():
            self.check_with_place("dist_mnist.py", delta=1e-5)


class TestDistMnist2x2Lars(TestDistBase):
    def _setup_config(self):
        self._sync_mode = True
        self._use_reduce = False

    def test_se_resnext(self):
        self.check_with_place("dist_mnist_lars.py", delta=1e-5)


class TestDistMnist2x2WithMemopt(TestDistBase):
    def _setup_config(self):
        self._sync_mode = True
        self._mem_opt = True

    def test_dist_train(self):
        self.check_with_place("dist_mnist.py", delta=1e-5)


class TestDistMnistAsync(TestDistBase):
    def _setup_config(self):
        self._sync_mode = False
        self._use_reduce = False

    def test_dist_train(self):
        self.check_with_place("dist_mnist.py", delta=200)


class TestDistMnistDcAsgd(TestDistBase):
    def _setup_config(self):
        self._sync_mode = False
        self._dc_asgd = True

    def test_se_resnext(self):
        self.check_with_place("dist_mnist.py", delta=200)


# FIXME(typhoonzero): enable these tests once we have 4
# 4 GPUs on CI machine, and the base class should be updated.
#
# class TestDistMnist2x2ReduceMode(TestDistBase):
#     def _setup_config(self):
#         self._sync_mode = True
#         self._use_reduce = True

#     def test_se_resnext(self):
#         self.check_with_place("dist_mnist.py", delta=1e-7)

# class TestDistMnistAsyncReduceMode(TestDistBase):
#     def _setup_config(self):
#         self._sync_mode = False
#         self._use_reduce = True

#     def test_se_resnext(self):
#         self.check_with_place("dist_mnist.py", delta=200)

if __name__ == "__main__":
    unittest.main()
