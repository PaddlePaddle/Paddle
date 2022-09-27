#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import paddle.fluid as fluid
import paddle

from test_collective_base import TestDistBase

paddle.enable_static()


class TestReduceScatterAPI(TestDistBase):

    def _setup_config(self):
        pass

    def test_reducescatter(self, col_type="reduce_scatter"):
        self.check_with_place("collective_reducescatter.py", col_type)

    def test_reducescatter_with_error(self):
        nranks = 2
        tindata = fluid.data(name="tindata", shape=[5, 1000], dtype='float32')
        try:
            toutdata = fluid.layers.collective._c_reducescatter(tindata, nranks)
        except ValueError:
            pass


if __name__ == '__main__':
    unittest.main()
