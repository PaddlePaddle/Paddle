#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import unittest
import sys
import paddle
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler

paddle.enable_static()
SEED = 2021


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestConvNet(unittest.TestCase):
    def test_training(self):
        ipu_strategy = compiler.get_ipu_strategy()

        assert ipu_strategy.num_ipus == 1, "Default num_ipus must be 1"
        assert ipu_strategy.is_training == True, "Default is_training is True"
        assert ipu_strategy.enable_pipelining == False, \
            "Default enable_pipelining is False"
        assert ipu_strategy.enable_manual_shard == False, \
            "Default enable_manual_shard is False"

        ipu_strategy.num_ipus = 2
        assert ipu_strategy.num_ipus == 2, "Set num_ipus Failed"

        ipu_strategy.is_training = False
        assert ipu_strategy.is_training == False, "Set is_training Failed"

        ipu_strategy.enable_pipelining = True
        assert ipu_strategy.enable_pipelining == True, \
            "Set enable_pipelining Failed"

        ipu_strategy.enable_manual_shard = True
        assert ipu_strategy.enable_manual_shard == True, \
            "Set enable_manual_shard Failed"


if __name__ == "__main__":
    unittest.main()
