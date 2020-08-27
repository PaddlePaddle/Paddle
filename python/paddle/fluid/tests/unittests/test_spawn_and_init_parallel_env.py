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

# NOTE(chenweihang): Coverage CI is currently not able to count python3
# unittest, so the unittests here covers some cases that will only be 
# executed in the python3 sub-process. 


class TestInitParallelEnv(unittest.TestCase):
    def test_beckend_type_error(self):
        with self.assertRaises(TypeError):
            dist.init_parallel_env(backend=1)

    def test_backend_value_error(self):
        with self.assertRaises(ValueError):
            dist.init_parallel_env(backend="mpi")


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
