# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.contrib.mixed_precision import fp16_utils
import paddle
import paddle.nn as nn
import paddle.static as static
import numpy as np
sys.path.append("..")
from paddle.fluid.tests.test_mixed_precision import AMPTest

paddle.enable_static()


class AMPTestNpu(AMPTest):
    def setUp(self):
        self.place = paddle.NPUPlace(0)


if __name__ == '__main__':
    unittest.main()
