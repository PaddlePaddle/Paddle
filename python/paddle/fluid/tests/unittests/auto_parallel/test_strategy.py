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
from paddle.distributed.fleet import auto


class TestStrategy(unittest.TestCase):
    def test_default_config(self):
        strategy = auto.Strategy()

        recompute = strategy.recompute
        self.assertEqual(recompute.enable, False)
        self.assertIsNone(recompute.checkpoints)

        amp = strategy.amp
        self.assertEqual(amp.enable, False)
        self.assertAlmostEqual(amp.init_loss_scaling, 32768.0)
        self.assertEqual(amp.incr_every_n_steps, 1000)
        self.assertEqual(amp.decr_every_n_nan_or_inf, 2)
        self.assertAlmostEqual(amp.incr_ratio, 2.0)
        self.assertAlmostEqual(amp.decr_ratio, 0.8)
        self.assertEqual(amp.use_dynamic_loss_scaling, True)
        self.assertEqual(amp.custom_black_list, [])
        self.assertEqual(amp.custom_white_list, [])
        self.assertEqual(amp.custom_black_varnames, [])
        self.assertEqual(amp.use_pure_fp16, False)
        self.assertEqual(amp.use_fp16_guard, True)
        self.assertEqual(amp.use_optimizer_fp16, False)

        sharding = strategy.sharding
        self.assertEqual(sharding.enable, False)
        self.assertEqual(sharding.stage, 1)
        self.assertEqual(sharding.degree, 8)
        self.assertAlmostEqual(sharding.overlap_grad_comm, False)
        self.assertAlmostEqual(sharding.bucket_size_numel, -1)
        self.assertAlmostEqual(sharding.partition_algor, "greedy_even")
        self.assertEqual(sharding.enable_tuning, False)
        self.assertEqual(sharding.tuning_range, [])

        gradient_merge = strategy.gradient_merge
        self.assertEqual(gradient_merge.enable, False)
        self.assertEqual(gradient_merge.k_steps, 1)
        self.assertEqual(gradient_merge.avg, True)

        qat = strategy.qat
        self.assertEqual(qat.enable, False)
        self.assertEqual(qat.channel_wise_abs_max, True)
        self.assertEqual(qat.weight_bits, 8)
        self.assertEqual(qat.activation_bits, 8)
        self.assertEqual(qat.not_quant_pattern, ['skip_quant'])
        self.assertIsNone(qat.algo)

        tuning = strategy.tuning
        self.assertEqual(tuning.enable, False)
        self.assertEqual(tuning.batch_size, 1)
        self.assertIsNone(tuning.dataset)
        self.assertEqual(tuning.profile_start_step, 1)
        self.assertEqual(tuning.profile_end_step, 1)
        self.assertEqual(tuning.run_after_tuning, True)
        self.assertEqual(tuning.verbose, True)

    def test_modify_config(self):
        strategy = auto.Strategy()

        recompute = strategy.recompute
        recompute.enable = True
        recompute.checkpoints = ["x"]
        self.assertEqual(recompute.enable, True)
        self.assertEqual(recompute.checkpoints, ["x"])

        amp = strategy.amp
        amp.enable = True
        amp.init_loss_scaling = 16384.0
        amp.incr_every_n_steps = 2000
        amp.decr_every_n_nan_or_inf = 4
        amp.incr_ratio = 4.0
        amp.decr_ratio = 0.4
        amp.use_dynamic_loss_scaling = False
        amp.custom_white_list = ["x"]
        amp.custom_black_list = ["y"]
        amp.custom_black_varnames = ["z"]
        amp.use_pure_fp16 = True
        amp.use_fp16_guard = False
        amp.use_optimizer_fp16 = True
        self.assertEqual(amp.enable, True)
        self.assertAlmostEqual(amp.init_loss_scaling, 16384.0)
        self.assertEqual(amp.incr_every_n_steps, 2000)
        self.assertEqual(amp.decr_every_n_nan_or_inf, 4)
        self.assertAlmostEqual(amp.incr_ratio, 4.0)
        self.assertAlmostEqual(amp.decr_ratio, 0.4)
        self.assertEqual(amp.use_dynamic_loss_scaling, False)
        self.assertEqual(amp.custom_white_list, ["x"])
        self.assertEqual(amp.custom_black_list, ["y"])
        self.assertEqual(amp.custom_black_varnames, ["z"])
        self.assertEqual(amp.use_pure_fp16, True)
        self.assertEqual(amp.use_fp16_guard, False)
        self.assertEqual(amp.use_optimizer_fp16, True)

        sharding = strategy.sharding
        sharding.enable = True
        sharding.stage = 2
        sharding.degree = 2
        sharding.segment_broadcast_MB = 64.0
        sharding.enable_tuning = True
        sharding.tuning_range = [1, 2, 3]
        self.assertEqual(sharding.enable, True)
        self.assertEqual(sharding.stage, 2)
        self.assertEqual(sharding.degree, 2)
        self.assertAlmostEqual(sharding.segment_broadcast_MB, 64.0)
        self.assertEqual(sharding.enable_tuning, True)
        self.assertEqual(sharding.tuning_range, [1, 2, 3])

        gradient_merge = strategy.gradient_merge
        gradient_merge.enable = True
        gradient_merge.k_steps = 4
        gradient_merge.avg = False
        self.assertEqual(gradient_merge.enable, True)
        self.assertEqual(gradient_merge.k_steps, 4)
        self.assertEqual(gradient_merge.avg, False)

    # def test_file_config(self):
    #     yaml_data = """
    #     all_ranks: false
    #     amp:
    #         custom_black_list:
    #         - y
    #         custom_black_varnames:
    #         - z
    #         custom_white_list:
    #         - x
    #         decr_every_n_nan_or_inf: 4
    #         decr_ratio: 0.4
    #         enable: false
    #         incr_every_n_steps: 2000
    #         incr_ratio: 4.0
    #         init_loss_scaling: 16384.0
    #         use_dynamic_loss_scaling: false
    #         use_fp16_guard: false
    #         use_optimizer_fp16: true
    #         use_pure_fp16: true
    #     auto_mode: semi
    #     gradient_merge:
    #         avg: false
    #         enable: false
    #         k_steps: 4
    #     gradient_scale: true
    #     qat:
    #         activation_bits: 8
    #         algo: null
    #         channel_wise_abs_max: true
    #         enable: false
    #         not_quant_pattern:
    #         - skip_quant
    #         weight_bits: 8
    #     recompute:
    #         checkpoints: null
    #         enable: false
    #         enable_tuning: false
    #     return_numpy: true
    #     seed: null
    #     sharding:
    #         enable: false
    #         enable_tuning: true
    #         segment_broadcast_MB: 64.0
    #         degree: 8
    #         stage: 2
    #         tuning_range: None
    #     split_data: false
    #     tuning:
    #         batch_size: 1
    #         dataset: null
    #         enable: false
    #         profile_end_step: 1
    #         profile_start_step: 1
    #         run_after_tuning: true
    #         verbose: true
    #     use_cache: true
    #     """
    #     yaml_path = "./strategy.yml"
    #     yaml_dict = yaml.load(yaml_data, Loader=yaml.Loader)
    #     with open(yaml_path, 'w') as outfile:
    #         yaml.dump(yaml_dict, outfile, default_flow_style=False)

    #     strategy = auto.Strategy(yaml_path)
    #     self.assertEqual(yaml_dict, strategy.to_dict())

    #     # Remove the created file
    #     if os.path.exists(yaml_path):
    #         os.remove(yaml_path)


if __name__ == '__main__':
    unittest.main()
