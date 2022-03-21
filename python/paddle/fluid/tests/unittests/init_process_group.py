# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import random
import numpy as np
import os
import shutil

import paddle
from paddle.fluid import core
import datetime
from datetime import timedelta
import paddle.fluid.core as core
from paddle.fluid.framework import _test_eager_guard
from paddle.fluid.dygraph.parallel import ParallelEnv


class TestProcessGroupFp32(unittest.TestCase):
    def setUp(self):
        self.config()

    def config(self):
        pass

    def test_init_process_group(self):
        paddle.distributed.collective._init_parallel_env()
        paddle.distributed.collective._new_group()
        with self.assertRaises(ValueError):
            paddle.distributed.collective._new_group(
                backend="gloo", group_name="_default_pg")
        print("test ok\n")


if __name__ == "__main__":
    unittest.main()
