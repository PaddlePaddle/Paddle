# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler
import paddle.fluid.core as core
import paddle.fluid.core.lite as lite
import paddle.fluid.layers as layers
import numpy as np
import unittest

from paddle.fluid.cxx_trainer import add_feed_fetch_op


def _as_lodtensor(data, place):
    # single tensor case
    tensor = core.LoDTensor()
    tensor.set(data, place)
    return tensor


data_label = [[
    0.753544, 0.772977, 0.646915, 0.747543, 0.528923, 0.0517749, 0.248678,
    0.75932, 0.960376, 0.606618
]]
data_a = [[
    0.874445, 0.21623, 0.713262, 0.702672, 0.396977, 0.828285, 0.932995,
    0.442674, 0.0321735, 0.484833, 0.045935, 0.21276, 0.556421, 0.131825,
    0.285626, 0.741409, 0.257467, 0.975958, 0.444006, 0.114553
]]

data_loss = [0.9876687]


class NaiveModelTest(unittest.TestCase):
    def test_model(self):

        start_prog = fluid.Program()
        main_prog = fluid.Program()

        start_prog.random_seed = 100
        main_prog.random_seed = 100

        with fluid.program_guard(main_prog, start_prog):
            a = fluid.layers.data(name="a", shape=[1, 20], dtype='float32')
            label = fluid.layers.data(name="label", shape=[10], dtype='float32')
            a1 = fluid.layers.fc(input=a, size=10, act=None, bias_attr=False)
            cost = fluid.layers.square_error_cost(a1, label)
            avg_cost = fluid.layers.mean(cost)

            optimizer = fluid.optimizer.SGD(learning_rate=0.001)
            optimizer.minimize(avg_cost)

            x86_place = lite.Place(lite.TargetType.kX86,
                                   lite.PrecisionType.kFloat,
                                   lite.DataLayoutType.kNCHW, 0)
            host_place = lite.Place(lite.TargetType.kHost,
                                    lite.PrecisionType.kFloat,
                                    lite.DataLayoutType.kNCHW, 0)
            scope = lite.Scope()

        trainer = lite.CXXTrainer(scope, x86_place, [x86_place, host_place])
        trainer.run_startup_program(start_prog.desc)

        cpu = fluid.core.CPUPlace()
        main_prog = add_feed_fetch_op(
            main_prog,
            feed=['a', 'label'],
            fetch_list={avg_cost},
            scope=scope,
            place=cpu)
        # print(main_prog)
        exe = trainer.build_main_program_executor(main_prog.desc)

        feed_data = [
            _as_lodtensor(np.array(data_a, object), cpu),
            _as_lodtensor(np.array(data_label, object), cpu)
        ]

        exe.run(feed_data)
        # print(np.array(exe.get_output(0).raw_tensor()))
        self.assertTrue(
            np.allclose(
                np.array(data_loss),
                np.array(exe.get_output(0).raw_tensor()),
                atol=1e-8),
            "lite result not equel to offline result")


if __name__ == '__main__':
    unittest.main()
