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

# NOTE(chenweihang): Coverage CI is currently not able to count python3
# unittest, so the unittests here covers some cases that will only be 
# executed in the python3 sub-process. 
# If the coverage CI can check python3 and sub-process, 
# we can remove all unittests here


class TestInitParallelEnv(unittest.TestCase):
    def test_beckend_type_error(self):
        with self.assertRaises(TypeError):
            dist.init_parallel_env(backend=1)

    def test_backend_value_error(self):
        with self.assertRaises(ValueError):
            dist.init_parallel_env(backend="mpi")

    def test_rank_type_error(self):
        with self.assertRaises(TypeError):
            dist.init_parallel_env(rank="1")

    def test_rank_value_error(self):
        with self.assertRaises(ValueError):
            dist.init_parallel_env(rank=-2)

    def test_only_cluster_node_ips_error(self):
        with self.assertRaises(ValueError):
            dist.init_parallel_env(
                rank=0, cluster_node_ips="127.0.0.1,127.0.0.2")

    def test_no_started_port_error(self):
        with self.assertRaises(ValueError):
            dist.init_parallel_env(rank=0)

    def test_no_selected_gpus_error(self):
        with self.assertRaises(ValueError):
            dist.init_parallel_env(rank=0, started_port=6170)

    def test_check_env_failed(self):
        os.environ['FLAGS_selected_gpus'] = '0'
        os.environ['PADDLE_TRAINER_ID'] = '0'
        os.environ['PADDLE_CURRENT_ENDPOINT'] = '127.0.0.1:6170'
        os.environ['PADDLE_TRAINERS_NUM'] = '1'
        with self.assertRaises(ValueError):
            dist.init_parallel_env()

    def test_update_env(self):
        device = os.getenv("CUDA_VISIBLE_DEVICES", None)
        if device is None:
            device = '0'
        dist.init_parallel_env(rank=0, started_port=6170, selected_gpus=device)
        self.assertIsNotNone(os.environ.get('PADDLE_TRAINER_ID', None))
        self.assertIsNotNone(os.environ.get('PADDLE_CURRENT_ENDPOINT', None))
        self.assertIsNotNone(os.environ.get('PADDLE_TRAINERS_NUM', None))
        self.assertIsNotNone(os.environ.get('PADDLE_TRAINER_ENDPOINTS', None))


if __name__ == "__main__":
    unittest.main()
