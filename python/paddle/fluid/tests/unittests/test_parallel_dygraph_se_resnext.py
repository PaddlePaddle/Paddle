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

from __future__ import print_function
import unittest
from test_dist_base import TestDistBase
import paddle.fluid as fluid


class TestParallelDygraphSeResNeXt(TestDistBase):
    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True

    def test_se_resnext(self):
        # TODO(Yancey1989): BN and Dropout is related with batchsize, so the delta is the 1,
        # try to remove the BN and Dropout in the network and using delta = 1e-5
        if fluid.core.is_compiled_with_cuda():
            self.check_with_place("parallel_dygraph_se_resnext.py", delta=1)


if __name__ == "__main__":
    unittest.main()
