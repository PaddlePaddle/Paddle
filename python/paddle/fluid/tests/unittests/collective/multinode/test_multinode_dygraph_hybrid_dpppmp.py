#!/usr/bin/python3

#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from test_collective_multi_nodes import TestDistBase

import os


class TestDYgraphHybrid(TestDistBase):

    def setUp(self):
        self._trainers = 16
        self._init_env()

    def test_hybrid_dpppmp(self):
        self.check_with_place("dygraph_hybrid_dpppmp.py",
                              backend="nccl",
                              need_envs=os.environ)

    def test_hybrid_recompute(self):
        self.check_with_place("dygraph_hybrid_recompute.py",
                              backend="nccl",
                              need_envs=os.environ)

    def test_hybrid_fp16(self):
        self.check_with_place("dygraph_hybrid_fp16.py",
                              backend="nccl",
                              need_envs=os.environ)


if __name__ == '__main__':
    unittest.main()
