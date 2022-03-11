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
from paddle.distributed.ps.utils.public import logger, ps_log_root_dir, debug_program
import paddle.distributed.fleet as fleet


class TestFlPs(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def check(self, file1, file2):
        pass

    def test_fl_cpu_async(self):
        os.environ['WITH_DISTRIBUTE'] = 'ON'
        # 0. get role
        import paddle.distributed.fleet.base.role_maker as role_maker
        role_maker = role_maker.PaddleCloudRoleMaker()
        role_maker._generate_role()

        from paddle.fluid.tests.unittests.ps.ps_dnn_trainer import YamlHelper, StaticModel, get_user_defined_strategy
        # 1. load config
        yaml_helper = YamlHelper()
        config_yaml_path = '../ps/fl_async_ps_config.yaml'
        config = yaml_helper.load_yaml(config_yaml_path)
        yaml_helper.print_yaml(config)
        # 2. get static model
        paddle.enable_static()
        model = StaticModel(config)
        feeds_list = model.create_feeds()
        metrics = model.fl_net(feeds_list)
        loss = model._cost
        # 3. add optimizer
        user_defined_strategy = get_user_defined_strategy(config)
        learning_rate = config.get("hyper_parameters.optimizer.learning_rate")
        inner_optimizer = paddle.optimizer.Adam(learning_rate, lazy_mode=True)
        from paddle.distributed.fleet.meta_optimizers.ps_optimizer import ParameterServerOptimizer
        ps_optimizer = ParameterServerOptimizer(inner_optimizer)
        ps_optimizer._set_basic_info(loss, role_maker, inner_optimizer,
                                     user_defined_strategy)
        ps_optimizer.minimize_impl(loss)


if __name__ == '__main__':
    remove_path_if_exists('/ps_log')
    unittest.main()
