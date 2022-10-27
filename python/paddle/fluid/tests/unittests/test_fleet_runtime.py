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

import unittest
import paddle


class TestFleetRuntime(unittest.TestCase):
    def test_fleet_runtime_base(self):
        import paddle.distributed.fleet.runtime

        base = paddle.distributed.fleet.runtime.runtime_base.RuntimeBase()
        base._run_worker()
        base._init_server()
        base._run_server()
        base._stop_worker()
        base._save_inference_model()
        base._save_persistables()

    def test_fleet_collective_runtime(self):
        import paddle.distributed.fleet.runtime

        collective_runtime = (
            paddle.distributed.fleet.runtime.CollectiveRuntime()
        )
        collective_runtime._init_worker()
        collective_runtime._run_worker()
        collective_runtime._init_worker()
        collective_runtime._run_server()
        collective_runtime._stop_worker()
        collective_runtime._save_inference_model()
        collective_runtime._save_persistables()

    def test_fleet_ps_runtime(self):
        ps_runtime = paddle.distributed.fleet.runtime.ParameterServerRuntime()
        self.assertRaises(
            Exception, ps_runtime._get_optimizer_status, "test_op", None
        )
        reshaped_names, origin_names = ps_runtime._get_optimizer_status(
            "adam", "param"
        )
        self.assertTrue(
            len(reshaped_names) == 2
            and reshaped_names[0] == 'param_moment1_0'
            and reshaped_names[1] == 'param_moment2_0'
        )
        self.assertTrue(
            len(origin_names) == 2
            and origin_names[0] == 'param_beta1_pow_acc_0'
            and origin_names[1] == 'param_beta2_pow_acc_0'
        )

        reshaped_names, origin_names = ps_runtime._get_optimizer_status(
            "sgd", "param"
        )
        self.assertTrue(len(reshaped_names) == 0 and len(origin_names) == 0)


if __name__ == "__main__":
    unittest.main()
