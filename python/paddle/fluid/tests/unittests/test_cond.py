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

from __future__ import print_function

import unittest

import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor
from paddle.fluid.framework import default_startup_program, default_main_program


def true_func():
    return layers.fill_constant(shape=[2, 3], dtype='int32', value=2)


def false_func():
    return layers.fill_constant(shape=[3, 2], dtype='int32', value=-1)


class TestCond(unittest.TestCase):
    def test_cond(self):
        x = layers.fill_constant(shape=[1], dtype='float32', value=0.1)
        y = layers.fill_constant(shape=[1], dtype='float32', value=0.23)
        pred = layers.less_than(y, x)
        out = layers.cond(pred, true_func, false_func)

        #place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda() else fluid.CPUPlace()
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(default_startup_program())
        ret = exe.run(default_main_program(), fetch_list=[out.name])
        print(ret)


if __name__ == '__main__':
    unittest.main()
