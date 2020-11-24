#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import unittest

import paddle
import paddle.distributed as dist
from paddle.distributed.spawn import _get_subprocess_env_list

from paddle.fluid import core
from paddle.fluid.dygraph import parallel_helper

# NOTE(chenweihang): Coverage CI is currently not able to count python3
# unittest, so the unittests here covers some cases that will only be 
# executed in the python3 sub-process. 


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestInitParallelEnv(unittest.TestCase):
    def test_check_env_failed(self):
        os.environ['FLAGS_selected_gpus'] = '0'
        os.environ['PADDLE_TRAINER_ID'] = '0'
        os.environ['PADDLE_CURRENT_ENDPOINT'] = '127.0.0.1:6170'
        os.environ['PADDLE_TRAINERS_NUM'] = '2'
        with self.assertRaises(ValueError):
            dist.init_parallel_env()

    def test_init_parallel_env_break(self):
        os.environ['FLAGS_selected_gpus'] = '0'
        os.environ['PADDLE_TRAINER_ID'] = '0'
        os.environ['PADDLE_CURRENT_ENDPOINT'] = '127.0.0.1:6170'
        os.environ['PADDLE_TRAINERS_NUM'] = '1'
        os.environ['PADDLE_TRAINER_ENDPOINTS'] = '127.0.0.1:6170'
        # coverage success branch
        dist.init_parallel_env()
        self.assertFalse(parallel_helper._is_parallel_ctx_initialized())


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSpawnAssistMethod(unittest.TestCase):
    def test_only_cluster_node_ips_error(self):
        with self.assertRaises(ValueError):
            options = dict()
            options['cluster_node_ips'] = "127.0.0.1,127.0.0.2"
            _get_subprocess_env_list(nprocs=1, options=options)

    def test_nprocs_greater_than_device_num_error(self):
        with self.assertRaises(RuntimeError):
            _get_subprocess_env_list(nprocs=100, options=dict())

    def test_selected_gpus_error(self):
        with self.assertRaises(ValueError):
            options = dict()
            options['selected_gpus'] = "100,101"
            _get_subprocess_env_list(nprocs=2, options=options)

    def test_get_correct_env(self):
        env_dict = _get_subprocess_env_list(nprocs=1, options=dict())[0]
        self.assertEqual(env_dict['PADDLE_TRAINER_ID'], '0')
        self.assertEqual(env_dict['PADDLE_TRAINERS_NUM'], '1')


if __name__ == "__main__":
    unittest.main()
