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

from __future__ import division
from __future__ import print_function

import os
import unittest
import numpy as np

import paddle
from ps_pass_test_base import *
from paddle.distributed.ps.utils.public import logger, ps_log_root_dir
from paddle.fluid.tests.unittests.ps.ps_dnn_trainer import DnnTrainer


class TestPsTrainerPass(PsPassTestBase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def check(self, file1, file2):
        with open(file1, 'r', encoding='utf-8') as f:
            text1 = f.read()
        with open(file2, 'r', encoding='utf-8') as f:
            text2 = f.read()
        if text1 == text2:
            return True
        else:
            return False

    def test_ps_optimizer_minimize_cpu_async(self):
        self.init()
        self.config['ps_mode_config'] = "../ps/cpu_async_ps_config.yaml"
        self.config['run_minimize'] = '1'

        self.config['debug_new_minimize'] = '0'
        self.config['log_dir'] = ps_log_root_dir + "async_cpu_log_old_minimize"
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch()

        self.config['debug_new_minimize'] = '1'
        self.config['log_dir'] = ps_log_root_dir + "async_cpu_log_new_minimize"
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch()

        file1 = './ps_log/async_run_minimize_debug:_0_worker_main.prototxt'
        file2 = './ps_log/async_run_minimize_debug:_1_worker_main.prototxt'
        if self.check(file1, file2):
            logger.info('test_ps_optimizer_minimize_cpu_async passed!')
        else:
            logger.error('test_ps_optimizer_minimize_cpu_async failed!')

    def test_ps_optimizer_minimize_cpu_sync(self):
        self.init()
        self.config['ps_mode_config'] = "../ps/cpu_sync_ps_config.yaml"
        self.config['run_minimize'] = '1'

        self.config['debug_new_minimize'] = '0'
        self.config['log_dir'] = ps_log_root_dir + "sync_cpu_log_old_minimize"
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch()

        self.config['debug_new_minimize'] = '1'
        self.config['log_dir'] = ps_log_root_dir + "sync_cpu_log_new_minimize"
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch()
        '''
        file1 = './ps_log/sync_run_minimize_debug:_0_worker_main.prototxt'
        file2 = './ps_log/sync_run_minimize_debug:_1_worker_main.prototxt'
        if self.check(file1, file2):
            logger.info('test_ps_optimizer_minimize_cpu_sync passed!')
        else:
            logger.error('test_ps_optimizer_minimize_cpu_sync failed!')
        '''

    def test_ps_optimizer_minimize_cpu_geo(self):
        self.init()
        self.config['ps_mode_config'] = "../ps/cpu_geo_ps_config.yaml"
        self.config['run_minimize'] = '1'

        self.config['debug_new_minimize'] = '0'
        self.config['log_dir'] = ps_log_root_dir + "geo_cpu_log_old_minimize"
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch()

        self.config['debug_new_minimize'] = '1'
        self.config['log_dir'] = ps_log_root_dir + "geo_cpu_log_new_minimize"
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch()

        file1 = './ps_log/geo_run_minimize_debug:_0_worker_main.prototxt'
        file2 = './ps_log/geo_run_minimize_debug:_1_worker_main.prototxt'
        if self.check(file1, file2):
            logger.info('test_ps_optimizer_minimize_cpu_geo passed!')
        else:
            logger.error('test_ps_optimizer_minimize_cpu_geo failed!')

    # heter ps 二阶段
    def test_ps_optimizer_minimize_heter(self):
        self.init()
        self.config['worker_num'] = "2"
        self.config['server_num'] = "2"
        self.config['heter_worker_num'] = '2'
        self.config['heter_devices'] = 'gpu'

        self.config['run_minimize'] = '1'
        self.config['ps_mode_config'] = "../ps/heter_ps_config.yaml"

        self.config['debug_new_minimize'] = '0'
        self.config['log_dir'] = ps_log_root_dir + "heter_log_old_minimize"
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch('heter-ps')

        self.config['debug_new_minimize'] = '1'
        self.config['log_dir'] = ps_log_root_dir + "heter_log_new_minimize"
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch('heter-ps')
        '''
        file1 = './ps_log/heter_run_minimize_debug:_0_worker_main.prototxt'
        file2 = './ps_log/heter_run_minimize_debug:_1_worker_main.prototxt'
        file3 = './ps_log/heter_run_minimize_debug:_0_heter_worker_main.prototxt'
        file4 = './ps_log/heter_run_minimize_debug:_1_heter_worker_main.prototxt'
        if self.check(file1, file2) and self.check(file3, file4):
            logger.info('test_ps_optimizer_minimize_heter passed!')
        else:
            logger.error('test_ps_optimizer_minimize_heter failed!')
        '''

    def test_ps_optimizer_minimize_gpu(self):
        self.init()
        self.config['run_minimize'] = '1'
        self.config['ps_mode_config'] = "../ps/gpu_ps_config.yaml"

        self.config['debug_new_minimize'] = '0'
        self.config['log_dir'] = ps_log_root_dir + "gpubox_log_old_minimize"
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch("gpu-ps")

        self.config['debug_new_minimize'] = '1'
        self.config['log_dir'] = ps_log_root_dir + "gpubox_log_new_minimize"
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch("gpu-ps")

        file1 = './ps_log/gpubox_run_minimize_debug:_0_worker_main.prototxt'
        file2 = './ps_log/gpubox_run_minimize_debug:_1_worker_main.prototxt'
        if self.check(file1, file2):
            logger.info('test_ps_optimizer_minimize_gpu passed!')
        else:
            logger.error('test_ps_optimizer_minimize_gpu failed!')

    def test_append_send_ops_pass(self):
        self.init()
        self.config['run_single_pass'] = '1'
        self.config['ps_mode_config'] = "../ps/cpu_async_ps_config.yaml"
        self.config['applied_pass_name'] = "append_send_ops_pass"

        self.config['debug_new_pass'] = '0'
        self.config['log_dir'] = ps_log_root_dir + "log_old_" + self.config[
            'applied_pass_name']
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch("cpu-ps")

        self.config['debug_new_pass'] = '1'
        self.config['log_dir'] = ps_log_root_dir + "log_new_" + self.config[
            'applied_pass_name']
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch("cpu-ps")

        file1 = './ps_log/async_append_send_ops_pass_debug:_0_worker_main.prototxt'
        file2 = './ps_log/async_append_send_ops_pass_debug:_1_worker_main.prototxt'
        if self.check(file1, file2):
            logger.info('test_append_send_ops_pass passed!')
        else:
            logger.info('test_append_send_ops_pass failed!')

    def test_distributed_ops_pass(self):
        pass


if __name__ == '__main__':
    remove_path_if_exists('./ps_log')
    unittest.main()
