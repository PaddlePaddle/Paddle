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
from paddle.base import core

paddle.enable_static()

device = "gpu" if core.is_compiled_with_cuda() else "cpu"


class TestCostModel(unittest.TestCase):
    def test_profiler_measure_program(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            # TODO(zhhsplendid): support paddle.static.data, which is uninitialized data
            data = paddle.ones(name='X', shape=[16, 100], dtype='float32')
            hidden = paddle.static.nn.fc(data, 10)
            loss = paddle.mean(hidden)
        cost_model = core.CostModel()
        cost_data = cost_model.profile_measure(
            main_program, startup_program, device, ["time"]
        )
        fc_op_time = cost_data.get_op_time_ms(0)
        mean_op_time = cost_data.get_op_time_ms(1)
        self.assertGreater(fc_op_time, 0)
        self.assertGreater(mean_op_time, 0)
        self.assertGreaterEqual(
            cost_data.get_whole_time_ms(), fc_op_time + mean_op_time
        )


if __name__ == '__main__':
    unittest.main()
