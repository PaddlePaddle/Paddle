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

from paddle.fluid.dygraph.jit import dygraph_to_static_output

import numpy as np
import unittest

import paddle.fluid as fluid

SEED = 2020


class Pool2D(fluid.dygraph.Layer):
    def __init__(self):
        super(Pool2D, self).__init__()
        self.pool2d = fluid.dygraph.Pool2D(
            pool_size=2, pool_type='avg', pool_stride=1, global_pooling=False)

    @dygraph_to_static_output
    def forward(self, x):
        inputs = fluid.dygraph.to_variable(x)

        # Add func `get_result` for testing arg_name_to_idx in ast transformation.
        def get_result(x):
            return self.pool2d(x)

        pre = get_result(inputs)
        return pre


class Linear(fluid.dygraph.Layer):
    def __init__(self):
        super(Linear, self).__init__()
        self.fc = fluid.dygraph.Linear(
            input_dim=10,
            output_dim=5,
            act='relu',
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.99)),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.5)))

    @dygraph_to_static_output
    def forward(self, x):
        inputs = fluid.dygraph.to_variable(x)
        pre = self.fc(inputs)
        return pre


class TestPool2D(unittest.TestCase):
    def setUp(self):
        self.dygraph_class = Pool2D
        self.data = np.random.random((1, 2, 4, 4)).astype('float32')

    def run_dygraph_mode(self):
        with fluid.dygraph.guard():
            dy_layer = self.dygraph_class()
            for _ in range(1):

                prediction = dy_layer(x=self.data)
                return prediction.numpy()

    def run_static_mode(self):
        startup_prog = fluid.Program()
        main_prog = fluid.Program()
        with fluid.program_guard(main_prog, startup_prog):
            dy_layer = self.dygraph_class()
            out = dy_layer(x=self.data)
            return out[0]

    def test_static_output(self):
        dygraph_res = self.run_dygraph_mode()
        static_res = self.run_static_mode()

        self.assertTrue(
            np.allclose(dygraph_res, static_res),
            msg='dygraph_res is {}\n static_res is \n{}'.format(dygraph_res,
                                                                static_res))
        return


class TestLinear(TestPool2D):
    def setUp(self):
        self.dygraph_class = Linear
        self.data = np.random.random((4, 10)).astype('float32')


if __name__ == '__main__':
    unittest.main()
