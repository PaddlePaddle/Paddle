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
import numpy as np
import paddle
import paddle.fluid as fluid

paddle.enable_static()


class TestFleetExecutor(unittest.TestCase):
    def fake_fleet_opt(self):
        # TODO: Fake for coverage will be removed in the future
        import paddle.distributed.fleet as fleet
        strategy = fleet.DistributedStrategy()
        strategy.sharding_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1
        }
        strategy.pipeline_configs = {"accumulate_steps": 1}
        fleet_opt = {
            "dist_strategy": strategy.sharding_configs,
            "num_micro_batches": strategy.pipeline_configs["accumulate_steps"]
        }
        return fleet_opt

    def run_fleet_executor(self, place, x_data, y_data):
        exe = paddle.static.Executor(place)
        empty_program = paddle.static.Program()
        with fluid.program_guard(empty_program, empty_program):
            x = fluid.layers.data(
                name='x', shape=x_data.shape, dtype=x_data.dtype)
            y = fluid.layers.data(
                name='y', shape=y_data.shape, dtype=y_data.dtype)
            z = x + y
            a = 2 * x + 3 * y
            loss = paddle.mean(a)
            base_lr = 0.1
            passes = [30, 60, 80, 90]
            steps_per_pass = 10
            bd = [steps_per_pass * p for p in passes]
            lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
            lr_val = paddle.optimizer.lr.PiecewiseDecay(
                boundaries=bd, values=lr)
            opt = paddle.optimizer.AdamW(
                learning_rate=lr_val,
                grad_clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0))
            opt.minimize(loss)
        # TODO: section_program will be removed in the future
        empty_program._pipeline_opt = {
            "fleet_opt": self.fake_fleet_opt(),
            "section_program": empty_program
        }
        res = exe.run(empty_program,
                      feed={'x': x_data,
                            'y': y_data},
                      fetch_list=[z.name, a.name])
        return res

    def test_executor_on_single_device(self):
        if fluid.is_compiled_with_cuda():
            shape = (10000, 3462)
            x_data = np.random.rand(*shape)
            y_data = np.random.rand(*shape)
            z_data = x_data + y_data
            a_data = 2 * x_data + 3 * y_data
            res = self.run_fleet_executor(fluid.CUDAPlace(0), x_data, y_data)
            self.assertTrue(np.allclose(res[0], z_data))
            self.assertTrue(np.allclose(res[1], a_data))


if __name__ == "__main__":
    unittest.main()
