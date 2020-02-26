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

from __future__ import print_function

from paddle.fluid.dygraph.jit import dygraph_to_static_output, dygraph_run_in_static_mode

import numpy as np
import unittest

import paddle.fluid as fluid

SEED = 2020


class MyPool2D(fluid.dygraph.Layer):
    def __init__(self):
        super(MyPool2D, self).__init__()
        self.pool2d = fluid.dygraph.Pool2D(
            pool_size=2, pool_type='avg', pool_stride=1, global_pooling=False)

    @dygraph_run_in_static_mode
    def forward(self, x):
        inputs = fluid.dygraph.to_variable(x)
        hidden = self.pool2d(inputs)
        return hidden, inputs


# without Param
class TestPool2D(unittest.TestCase):
    def setUp(self):
        self.dygraph_class = MyPool2D
        self.data = np.random.random((1, 2, 4, 4)).astype('float32')

    def run_dygraph_mode(self):
        with fluid.dygraph.guard():
            dy_layer = self.dygraph_class()
            for _ in range(1):
                _, prediction = dy_layer(self.data)
                return prediction

    def run_static_mode(self):
        startup_prog = fluid.Program()
        main_prog = fluid.Program()
        with fluid.program_guard(main_prog, startup_prog):
            dy_layer = self.dygraph_class()
            out, _ = dy_layer(self.data)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            res = exe.run(main_prog, fetch_list=out)
            return res

    def test_static_output(self):
        dygraph_res = self.run_dygraph_mode()
        static_res = self.run_static_mode()
        self.assertTrue(
            np.allclose(dygraph_res[0], static_res[0]),
            msg='dygraph is {}\n static_res is \n{}'.format(dygraph_res,
                                                            static_res))


if __name__ == '__main__':
    unittest.main()
