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
        with _test_eager_guard():
            paddle.distributed.init_parallel_env()
            paddle.distributed.new_group()
            group = paddle.distributed.new_group([-1, -2])
            assert group.process_group == None

            group = paddle.distributed.collective.Group(-1, 2, 0, [-1, -2])
            ret = paddle.distributed.barrier(group)
            assert ret == None
        paddle.enable_static()
        in_tensor = paddle.empty((1, 2))
        in_tensor2 = paddle.empty((1, 2))
        paddle.distributed.broadcast(in_tensor, src=0)
        paddle.distributed.all_gather([in_tensor, in_tensor2], in_tensor)
        print("test ok\n")


if __name__ == "__main__":
    unittest.main()
