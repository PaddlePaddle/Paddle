#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid.core as core
from paddle.cost_model import CostModel

paddle.enable_static()

device = "gpu" if core.is_compiled_with_cuda() else "cpu"


class TestCostModel(unittest.TestCase):

    def test_profiler_measure_empty_program(self):
        cost_model = core.CostModel()
        empty_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        cost_data = cost_model.profile_measure(empty_program, startup_program,
                                               device, ["time"])
        self.assertEqual(cost_data.get_whole_time_ms(), 0)

    def test_profiler_measure_program(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            # TODO(zhhsplendid): support paddle.static.data, which is uninitialized data
            data = paddle.ones(name='X', shape=[16, 100], dtype='float32')
            hidden = paddle.static.nn.fc(data, 10)
            loss = paddle.mean(hidden)
        cost_model = core.CostModel()
        cost_data = cost_model.profile_measure(main_program, startup_program,
                                               device, ["time"])
        fc_op_time = cost_data.get_op_time_ms(0)
        mean_op_time = cost_data.get_op_time_ms(1)
        self.assertGreater(fc_op_time, 0)
        self.assertGreater(mean_op_time, 0)
        self.assertGreaterEqual(cost_data.get_whole_time_ms(),
                                fc_op_time + mean_op_time)

    def test_static_op_benchmark_cost_model(self):
        op_name = "abs"
        cost_model = CostModel()
        # init static data
        cost_model.static_cost_data()
        op_name = "abs"
        abs_op_cost = cost_model.get_static_op_time(op_name)
        abs_op_time = abs_op_cost["op_time"]
        abs_op_config = abs_op_cost["config"]
        print("abs_op_time:", abs_op_time)
        print("abs_op_config:", abs_op_config)
        self.assertGreater(float(abs_op_time), 0)
        conv2d_op_cost = cost_model.get_static_op_time("conv2d")
        conv2d_op_time = conv2d_op_cost["op_time"]
        conv2d_op_config = conv2d_op_cost["config"]
        self.assertGreater(float(conv2d_op_time), 0)
        print("conv2d_op_time:", conv2d_op_time)
        print("conv2d_op_config:", conv2d_op_config)

        conv2d_backward_op_cost = cost_model.get_static_op_time("conv2d",
                                                                forward=False)
        conv2d_backward_op_time = conv2d_backward_op_cost["op_time"]
        conv2d_backward_op_config = conv2d_backward_op_cost["config"]
        self.assertGreater(float(conv2d_backward_op_time), 0)
        print("conv2d_backward_op_time:", conv2d_backward_op_time)
        print("conv2d_backward_op_config:", conv2d_backward_op_config)

        conv2d_fp16_op_cost = cost_model.get_static_op_time("conv2d",
                                                            dtype="float16")
        conv2d_fp16_op_time = conv2d_fp16_op_cost["op_time"]
        conv2d_fp16_op_config = conv2d_fp16_op_cost["config"]
        self.assertGreater(float(conv2d_fp16_op_time), 0)
        print("conv2d_fp16_op_time:", conv2d_fp16_op_time)
        print("conv2d_fp16_op_config:", conv2d_fp16_op_config)


if __name__ == '__main__':
    unittest.main()
