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

import multiprocessing
import os
import unittest

import paddle
import paddle.distributed as dist
from paddle.base import core
from paddle.distributed.spawn import (
    _get_default_nprocs,
    _get_subprocess_env_list,
    _options_valid_check,
)

# NOTE(chenweihang): Coverage CI is currently not able to count python3
# unittest, so the unittests here covers some cases that will only be
# executed in the python3 sub-process.


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestInitParallelEnv(unittest.TestCase):
    def test_check_env_failed(self):
        os.environ['FLAGS_selected_gpus'] = '0'
        os.environ['PADDLE_TRAINER_ID'] = '0'
        # os.environ['PADDLE_CURRENT_ENDPOINT'] = '127.0.0.1:6170'
        os.environ['PADDLE_TRAINERS_NUM'] = '2'
        with self.assertRaises(ValueError):
            dist.init_parallel_env()

    def test_init_parallel_env_break(self):
        from paddle.distributed import parallel_helper

        os.environ['FLAGS_selected_gpus'] = '0'
        os.environ['PADDLE_TRAINER_ID'] = '0'
        os.environ['PADDLE_CURRENT_ENDPOINT'] = '127.0.0.1:6170'
        os.environ['PADDLE_TRAINERS_NUM'] = '1'
        os.environ['PADDLE_TRAINER_ENDPOINTS'] = '127.0.0.1:6170'
        # coverage success branch
        dist.init_parallel_env()
        self.assertFalse(parallel_helper._is_parallel_ctx_initialized())


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestSpawnAssistMethod(unittest.TestCase):
    def test_nprocs_greater_than_device_num_error(self):
        with self.assertRaises(RuntimeError):
            _get_subprocess_env_list(nprocs=100, options={})

    def test_selected_devices_error(self):
        with self.assertRaises(ValueError):
            options = {}
            options['selected_devices'] = "100,101"
            _get_subprocess_env_list(nprocs=2, options=options)

    def test_get_correct_env(self):
        options = {}
        options['print_config'] = True
        env_dict = _get_subprocess_env_list(nprocs=1, options=options)[0]
        self.assertEqual(env_dict['PADDLE_TRAINER_ID'], '0')
        self.assertEqual(env_dict['PADDLE_TRAINERS_NUM'], '1')

    def test_nprocs_not_equal_to_selected_devices(self):
        with self.assertRaises(ValueError):
            options = {}
            options['selected_devices'] = "100,101,102"
            _get_subprocess_env_list(nprocs=2, options=options)

    def test_options_valid_check(self):
        options = {}
        options['selected_devices'] = "100,101,102"
        _options_valid_check(options)

        with self.assertRaises(ValueError):
            options['error'] = "error"
            _options_valid_check(options)

    def test_get_default_nprocs(self):
        paddle.set_device('cpu')
        nprocs = _get_default_nprocs()
        self.assertEqual(nprocs, multiprocessing.cpu_count())

        paddle.set_device('gpu')
        nprocs = _get_default_nprocs()
        self.assertEqual(nprocs, core.get_cuda_device_count())


if __name__ == "__main__":
    unittest.main()
