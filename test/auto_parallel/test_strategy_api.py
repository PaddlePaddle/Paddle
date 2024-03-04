#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# import yaml
import unittest

import paddle.distributed as dist


class TestStrategy(unittest.TestCase):
    def test_default_config(self):
        strategy = dist.Strategy()
        self.assertEqual(strategy.sharding.enable, False)
        self.assertEqual(strategy.sharding.stage, 1)
        self.assertEqual(strategy.sharding.degree, 8)

        self.assertEqual(strategy.gradient_merge.enable, False)
        self.assertEqual(strategy.gradient_merge.k_steps, 1)
        self.assertEqual(strategy.gradient_merge.avg, True)

        self.assertEqual(strategy.pipeline.enable, False)
        self.assertEqual(strategy.pipeline.schedule_mode, "1F1B")
        self.assertEqual(strategy.pipeline.micro_batch_size, 1)
        self.assertEqual(strategy.pipeline.accumulate_steps, 1)

        self.assertEqual(strategy.fused_passes.enable, False)
        self.assertEqual(strategy.fused_passes.gemm_epilogue, False)
        self.assertEqual(strategy.fused_passes.dropout_add, False)

    def test_modify_config(self):
        strategy = dist.Strategy()

        strategy.sharding.enable = True
        strategy.sharding.stage = 2
        strategy.sharding.degree = 16
        self.assertEqual(strategy.sharding.enable, True)
        self.assertEqual(strategy.sharding.stage, 2)
        self.assertEqual(strategy.sharding.degree, 16)

        strategy.gradient_merge.enable = True
        strategy.gradient_merge.k_steps = 2
        strategy.gradient_merge.avg = False
        self.assertEqual(strategy.gradient_merge.enable, True)
        self.assertEqual(strategy.gradient_merge.k_steps, 2)
        self.assertEqual(strategy.gradient_merge.avg, False)

        strategy.pipeline.enable = True
        strategy.pipeline.schedule_mode = "FThenB"
        strategy.pipeline.micro_batch_size = 2
        self.assertEqual(strategy.pipeline.enable, True)
        self.assertEqual(strategy.pipeline.schedule_mode, "FThenB")
        self.assertEqual(strategy.pipeline.micro_batch_size, 2)

        strategy.fused_passes.enable = True
        strategy.fused_passes.gemm_epilogue = True
        self.assertEqual(strategy.fused_passes.enable, True)
        self.assertEqual(strategy.fused_passes.gemm_epilogue, True)

    def test_init_from_dict(self):
        config = {
            "sharding": {"enable": True, "stage": 2},
            "gradient_merge": {"enable": True, "k_steps": 2},
            "fused_passes": {"enable": True, "gemm_epilogue": True},
            "pipeline": {"enable": True, "schedule_mode": "FThenB"},
        }
        strategy = dist.Strategy(config)
        self.assertEqual(strategy.sharding.enable, True)
        self.assertEqual(strategy.sharding.stage, 2)
        self.assertEqual(strategy.sharding.degree, 8)  # default
        self.assertEqual(strategy.gradient_merge.enable, True)
        self.assertEqual(strategy.gradient_merge.k_steps, 2)
        self.assertEqual(strategy.gradient_merge.avg, True)  # default
        self.assertEqual(strategy.fused_passes.enable, True)
        self.assertEqual(strategy.fused_passes.gemm_epilogue, True)
        self.assertEqual(strategy.fused_passes.dropout_add, False)  # default

    def test_error_init(self):
        with self.assertRaises(ValueError):
            config = [{"enable": True, "stage": 2}]
            err_strategy1 = dist.Strategy(config)

        with self.assertRaises(ValueError):
            config = {
                "sharding": {"enable": True, "stage": 2},
                "gradient_merge": {"enable": True, "k_steps": 2},
                "fused_passes": {
                    "enable": True,
                    "gemm_epilogue": True,
                    "dropout": True,
                },
                "pipeline": {"enable": True, "schedule_mode": "FThenB"},
            }
            err_strategy2 = dist.Strategy(config)


if __name__ == '__main__':
    unittest.main()
