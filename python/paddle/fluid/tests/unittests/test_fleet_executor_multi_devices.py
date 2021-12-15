# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import os
import paddle
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet

paddle.enable_static()


class TestFleetExecutor(unittest.TestCase):
    def run_fleet_executor(self, place, fleet_opt=dict()):
        exe = paddle.static.Executor(place)
        empty_program = paddle.static.Program()
        with fluid.program_guard(empty_program, empty_program):
            x = fluid.layers.data(name='x', shape=[1], dtype=paddle.float32)
        empty_program._pipeline_opt = {
            "fleet_opt": fleet_opt,
            "section_program": empty_program
        }
        exe.run(empty_program, feed={'x': [1]})

    def test_dist_executor_on_multi_devices(self):
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ[
            "PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:7000,127.0.0.1:7001,127.0.0.1:7002,127.0.0.1:7003,127.0.0.1:7004,127.0.0.1:7005,127.0.0.1:7006,127.0.0.1:7007"
        strategy = fleet.DistributedStrategy()
        strategy.sharding_configs = {
            "dp_degree": 2,
            "mp_degree": 2,
            "pp_degree": 2
        }
        strategy.pipeline_configs = {"accumulate_steps": 8}
        fleet_opt = {
            "dist_strategy": strategy.sharding_configs,
            "num_micro_batches": strategy.pipeline_configs["accumulate_steps"]
        }
        if fluid.is_compiled_with_cuda():
            # TODO: Distribute test case is not supported for executor can not stop
            pass


if __name__ == "__main__":
    unittest.main()
