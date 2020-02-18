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

import numpy as np
import paddle.fluid as fluid
import unittest

from paddle.fluid.dygraph.jit import dygraph_to_static_output, _dygraph_to_static_output_

SEED = 2020
np.random.seed(SEED)


# Test custom class
class DygraphLayer(fluid.dygraph.Layer):
    def __init__(self):
        super(DygraphLayer, self).__init__()
        self.fc = fluid.dygraph.nn.Linear(
            input_dim=10,
            output_dim=5,
            act='relu',
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.99)),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.5)), )

    def forward(self, inputs):
        prediction = self.fc(inputs)
        return prediction


class TestDygraphBasicAPI(unittest.TestCase):
    '''
    Compare results of dynamic graph and transformed static graph function which only
    includes basic API.
    '''

    def setUp(self):
        self.input = np.random.random((4, 10)).astype('float32')
        self.dygraph_class = DygraphLayer

    def get_dygraph_output(self):
        with fluid.dygraph.guard():
            fluid.default_startup_program.random_seed = SEED
            fluid.default_main_program.random_seed = SEED
            data = fluid.dygraph.to_variable(self.input)

            res = self.dygraph_class()(data).numpy()

            return res

    def get_static_output(self):
        startup_program = fluid.Program()
        startup_program.random_seed = SEED
        main_program = fluid.Program()
        main_program.random_seed = SEED
        with fluid.program_guard(main_program, startup_program):
            data = fluid.layers.assign(self.input)
            func = _dygraph_to_static_output_(self.dygraph_class)
            static_out = func(data)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(startup_program)
        static_res = exe.run(main_program, fetch_list=static_out)
        return static_res[0]

    def test_transformed_static_result(self):
        dygraph_res = self.get_dygraph_output()
        static_res = self.get_static_output()
        print("dygraph_res\n", dygraph_res)
        print("static_res\n", static_res)
        self.assertTrue(np.array_equal(static_res, dygraph_res))


if __name__ == '__main__':
    unittest.main()
