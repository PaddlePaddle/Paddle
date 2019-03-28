# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest
import paddle.fluid as fluid

fluid.core._set_eager_deletion_mode(0.0, 1.0, True)

# FIXME(zjl): It seems that this unittest fails randomly 
# when comparing all reduce last loss and reduce last loss
# e.g.: AssertionError: 1.0357145 != 1.0673475 within 0.01 delta
# Disable it temporarily.
'''
from test_parallel_executor_mnist import TestMNIST


class EagerDeletionTestMNIST(TestMNIST):
    pass
'''

if __name__ == '__main__':
    unittest.main()
