# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import random
import numpy as np

import unittest
import paddle
import paddle.distributed.fleet as fleet
from auto_parallel_pass_test_base import AutoPallelPassTestBase
from test_auto_parallel_amp_pass import TestAMPPass


class TestPF16Pass(TestAMPPass):
    def apply_passes(self):
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.amp = True
        dist_strategy.amp_configs = {
            "custom_white_list": [
                'softmax',
                'layer_norm',
                'gelu',
            ],
            "custom_black_list": ['c_softmax_with_cross_entropy'],
            "init_loss_scaling": 32768,
            "use_dynamic_loss_scaling": True,
            "use_pure_fp16": True
        }
        dist_strategy.semi_auto = True
        fleet.init(is_collective=True, strategy=dist_strategy)


if __name__ == "__main__":
    unittest.main()
