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
import paddle.fluid as fluid

import paddle
from paddle.fluid.tests.unittests.distributed_passes.ps_pass_test_base import *
from paddle.distributed.ps.utils.public import logger, ps_log_root_dir
from ps_dnn_trainer import DnnTrainer
import paddle.distributed.fleet.proto.the_one_ps_pb2 as ps_pb2
from google.protobuf import text_format


class TestTheOnePs(PsPassTestBase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def check(self, file1, file2):
        pass
        '''
        f = open(file1, "rb")
        ps_desc_1 = ps_pb2.PSParameter()
        text_format.Parse(f.read(), ps_desc_1)
        f.close()

        f = open(file2, "rb")
        ps_desc_2 = ps_pb2.PSParameter()
        text_format.Parse(f.read(), ps_desc_2)
        f.close()
        str1 = text_format.MessageToString(ps_desc_1)
        str2 = text_format.MessageToString(ps_desc_2)
        #logger.info('### msg10: {}'.format(str1))
        #logger.info('### msg20: {}'.format(str2))
        if str1 == str2:
            return True
        else:
            return False
        '''

    def test_ps_cpu_async(self):
        self.init()
        self.config['ps_mode_config'] = "../ps/cpu_async_ps_config.yaml"
        self.config['run_the_one_ps'] = '1'

        self.config['debug_the_one_ps'] = '0'
        self.config[
            'log_dir'] = ps_log_root_dir + "async_cpu_log_old_the_one_ps"
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch()

        self.config['debug_the_one_ps'] = '1'
        self.config[
            'log_dir'] = ps_log_root_dir + "async_cpu_log_new_the_one_ps"
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch()

        desc1 = '/ps_desc_baseline/async_worker_ps_desc'
        desc2 = '/ps_log/async_new_worker_ps_desc'
        desc3 = '/ps_desc_baseline/async_server_ps_desc'
        desc4 = '/ps_log/async_new_server_ps_desc'
        if self.check(desc1, desc2):
            logger.info('test_ps_cpu_async ps_desc: worker passed!')
        else:
            logger.info('test_ps_cpu_async ps_desc: worker failed!')
        if self.check(desc3, desc4):
            logger.info('test_ps_cpu_async ps_desc: server passed!')
        else:
            logger.info('test_ps_cpu_async ps_desc: server failed!')

    def test_ps_cpu_geo(self):
        self.init()
        self.config['ps_mode_config'] = "../ps/cpu_geo_ps_config.yaml"
        self.config['run_the_one_ps'] = '1'

        self.config['debug_the_one_ps'] = '0'
        self.config['log_dir'] = ps_log_root_dir + "geo_cpu_log_old_the_one_ps"
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch()

        self.config['debug_the_one_ps'] = '1'
        self.config['log_dir'] = ps_log_root_dir + "geo_cpu_log_new_the_one_ps"
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch()

        desc1 = '/ps_desc_baseline/geo_worker_ps_desc'
        desc2 = '/ps_log/geo_new_worker_ps_desc'
        desc3 = '/ps_desc_baseline/geo_server_ps_desc'
        desc4 = '/ps_log/geo_new_server_ps_desc'
        if self.check(desc1, desc2):
            logger.info('test_ps_cpu_geo ps_desc: worker passed!')
        else:
            logger.info('test_ps_cpu_geo ps_desc: worker failed!')
        if self.check(desc3, desc4):
            logger.info('test_ps_cpu_geo ps_desc: server passed!')
        else:
            logger.info('test_ps_cpu_geo ps_desc: server failed!')


if __name__ == '__main__':
    unittest.main()
