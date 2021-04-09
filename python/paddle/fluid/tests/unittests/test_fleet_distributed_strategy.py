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

import unittest
import paddle
import os


class TestStrategyConfig(unittest.TestCase):
    def test_amp(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.amp = True
        self.assertEqual(strategy.amp, True)
        strategy.amp = False
        self.assertEqual(strategy.amp, False)
        strategy.amp = "True"
        self.assertEqual(strategy.amp, False)

    def test_amp_configs(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        configs = {
            "init_loss_scaling": 32768,
            "decr_every_n_nan_or_inf": 2,
            "incr_every_n_steps": 1000,
            "incr_ratio": 2.0,
            "use_dynamic_loss_scaling": True,
            "decr_ratio": 0.5
        }
        strategy.amp_configs = configs
        self.assertEqual(strategy.amp_configs["init_loss_scaling"], 32768)

    def test_recompute(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.recompute = True
        self.assertEqual(strategy.recompute, True)
        strategy.recompute = False
        self.assertEqual(strategy.recompute, False)
        strategy.recompute = "True"
        self.assertEqual(strategy.recompute, False)

    def test_recompute_configs(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        configs = {"checkpoints": ["x", "y"]}
        strategy.recompute_configs = configs
        self.assertEqual(len(strategy.recompute_configs["checkpoints"]), 2)

    def test_pipeline(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.pipeline = True
        self.assertEqual(strategy.pipeline, True)
        strategy.pipeline = False
        self.assertEqual(strategy.pipeline, False)
        strategy.pipeline = "True"
        self.assertEqual(strategy.pipeline, False)

    def test_pipeline_configs(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        configs = {"micro_batch_size": 4}
        strategy.pipeline_configs = configs
        self.assertEqual(strategy.pipeline_configs["micro_batch_size"], 4)
        configs = {"accumulate_steps": 2}
        strategy.pipeline_configs = configs
        self.assertEqual(strategy.pipeline_configs["accumulate_steps"], 2)

    def test_localsgd(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.localsgd = True
        self.assertEqual(strategy.localsgd, True)
        strategy.localsgd = False
        self.assertEqual(strategy.localsgd, False)
        strategy.localsgd = "True"
        self.assertEqual(strategy.localsgd, False)

    def test_localsgd_configs(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        configs = {"k_steps": 4, "begin_step": 120}
        strategy.localsgd_configs = configs
        self.assertEqual(strategy.localsgd_configs["k_steps"], 4)
        self.assertEqual(strategy.localsgd_configs["begin_step"], 120)

    def test_adaptive_localsgd_configs(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        configs = {"init_k_steps": 1, "begin_step": 120}
        strategy.adaptive_localsgd_configs = configs
        self.assertEqual(strategy.adaptive_localsgd_configs["init_k_steps"], 1)
        self.assertEqual(strategy.adaptive_localsgd_configs["begin_step"], 120)

    def test_dgc(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.dgc = True
        self.assertEqual(strategy.dgc, True)
        strategy.dgc = False
        self.assertEqual(strategy.dgc, False)
        strategy.dgc = "True"
        self.assertEqual(strategy.dgc, False)

    def test_fp16_allreduce(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.fp16_allreduce = True
        self.assertEqual(strategy.fp16_allreduce, True)
        strategy.fp16_allreduce = False
        self.assertEqual(strategy.fp16_allreduce, False)
        with self.assertRaises(TypeError):
            strategy.fp16_allreduce = "True"
        self.assertEqual(strategy.fp16_allreduce, False)

    def test_sync_nccl_allreduce(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.sync_nccl_allreduce = True
        self.assertEqual(strategy.sync_nccl_allreduce, True)
        strategy.sync_nccl_allreduce = False
        self.assertEqual(strategy.sync_nccl_allreduce, False)
        strategy.sync_nccl_allreduce = "True"
        self.assertEqual(strategy.sync_nccl_allreduce, False)

    def test_nccl_comm_num(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.nccl_comm_num = 1
        self.assertEqual(strategy.nccl_comm_num, 1)
        strategy.nccl_comm_num = "2"
        self.assertEqual(strategy.nccl_comm_num, 1)

    def test_use_hierarchical_allreduce(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.use_hierarchical_allreduce = True
        self.assertEqual(strategy.use_hierarchical_allreduce, True)
        strategy.use_hierarchical_allreduce = False
        self.assertEqual(strategy.use_hierarchical_allreduce, False)
        strategy.use_hierarchical_allreduce = "True"
        self.assertEqual(strategy.use_hierarchical_allreduce, False)

    def test_hierarchical_allreduce_inter_nranks(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.hierarchical_allreduce_inter_nranks = 8
        self.assertEqual(strategy.hierarchical_allreduce_inter_nranks, 8)
        strategy.hierarchical_allreduce_inter_nranks = "4"
        self.assertEqual(strategy.hierarchical_allreduce_inter_nranks, 8)

    def test_sync_batch_norm(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.sync_batch_norm = True
        self.assertEqual(strategy.sync_batch_norm, True)
        strategy.sync_batch_norm = False
        self.assertEqual(strategy.sync_batch_norm, False)
        strategy.sync_batch_norm = "True"
        self.assertEqual(strategy.sync_batch_norm, False)

    def test_fuse_all_reduce_ops(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.fuse_all_reduce_ops = True
        self.assertEqual(strategy.fuse_all_reduce_ops, True)
        strategy.fuse_all_reduce_ops = False
        self.assertEqual(strategy.fuse_all_reduce_ops, False)
        strategy.fuse_all_reduce_ops = "True"
        self.assertEqual(strategy.fuse_all_reduce_ops, False)

    def test_fuse_grad_size_in_MB(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.fuse_grad_size_in_MB = 50
        self.assertEqual(strategy.fuse_grad_size_in_MB, 50)
        strategy.fuse_grad_size_in_MB = "40"
        self.assertEqual(strategy.fuse_grad_size_in_MB, 50)

    def test_last_comm_group_size_MB(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.last_comm_group_size_MB = 50
        self.assertEqual(strategy.last_comm_group_size_MB, 50)
        with self.assertRaises(ValueError):
            strategy.last_comm_group_size_MB = -1

    def test_find_unused_parameters(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.find_unused_parameters = True
        self.assertEqual(strategy.find_unused_parameters, True)
        strategy.find_unused_parameters = False
        self.assertEqual(strategy.find_unused_parameters, False)
        strategy.find_unused_parameters = "True"
        self.assertEqual(strategy.find_unused_parameters, False)

    def test_fuse_grad_size_in_TFLOPS(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy._fuse_grad_size_in_TFLOPS = 0.1
        self.assertGreater(strategy._fuse_grad_size_in_TFLOPS, 0.09)
        strategy._fuse_grad_size_in_TFLOPS = "0.3"
        self.assertGreater(strategy._fuse_grad_size_in_TFLOPS, 0.09)

    def test_gradient_merge(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.gradient_merge = True
        self.assertEqual(strategy.gradient_merge, True)
        strategy.gradient_merge = False
        self.assertEqual(strategy.gradient_merge, False)
        strategy.gradient_merge = "True"
        self.assertEqual(strategy.gradient_merge, False)

    def test_gradient_merge_configs(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        configs = {"k_steps": 4}
        strategy.gradient_merge_configs = configs
        self.assertEqual(strategy.gradient_merge_configs["k_steps"], 4)

    def test_lars(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.lars = True
        self.assertEqual(strategy.lars, True)
        strategy.lars = False
        self.assertEqual(strategy.lars, False)
        strategy.lars = "True"
        self.assertEqual(strategy.lars, False)

    def test_lamb(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.lamb = True
        self.assertEqual(strategy.lamb, True)
        strategy.lamb = False
        self.assertEqual(strategy.lamb, False)
        strategy.lamb = "True"
        self.assertEqual(strategy.lamb, False)

    def test_a_sync(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        self.assertEqual(strategy.a_sync, True)
        strategy.a_sync = False
        self.assertEqual(strategy.a_sync, False)

        with self.assertRaises(ValueError):
            strategy.a_sync = "True"

    def test_a_sync_configs(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        configs = {"k_steps": 1000}
        strategy.a_sync_configs = configs
        self.assertEqual(strategy.a_sync_configs["k_steps"], 1000)

    def test_elastic(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.elastic = True
        self.assertEqual(strategy.elastic, True)
        strategy.elastic = False
        self.assertEqual(strategy.elastic, False)
        strategy.elastic = "True"
        self.assertEqual(strategy.elastic, False)

    def test_auto(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.auto = True
        self.assertEqual(strategy.auto, True)
        strategy.auto = False
        self.assertEqual(strategy.auto, False)
        strategy.auto = "True"
        self.assertEqual(strategy.auto, False)

    def test_strategy_prototxt(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        strategy.localsgd = True
        strategy.dgc = True
        localsgd_configs = {"k_steps": 5, "begin_step": 1}
        strategy.localsgd_configs = localsgd_configs
        build_strategy = paddle.fluid.BuildStrategy()
        build_strategy.enable_sequential_execution = True
        build_strategy.nccl_comm_num = 10
        build_strategy.use_hierarchical_allreduce = True
        build_strategy.hierarchical_allreduce_inter_nranks = 1
        build_strategy.fuse_elewise_add_act_ops = True
        build_strategy.fuse_bn_act_ops = True
        build_strategy.enable_auto_fusion = True
        build_strategy.fuse_relu_depthwise_conv = True
        build_strategy.fuse_broadcast_ops = True
        build_strategy.fuse_all_optimizer_ops = True
        build_strategy.sync_batch_norm = True
        build_strategy.enable_inplace = True
        build_strategy.fuse_all_reduce_ops = True
        build_strategy.enable_backward_optimizer_op_deps = True
        build_strategy.trainers_endpoints = ["1", "2"]
        strategy.build_strategy = build_strategy
        exe_strategy = paddle.fluid.ExecutionStrategy()
        exe_strategy.num_threads = 10
        exe_strategy.num_iteration_per_drop_scope = 10
        exe_strategy.num_iteration_per_run = 10
        strategy.execution_strategy = exe_strategy
        strategy.save_to_prototxt("dist_strategy.prototxt")
        strategy2 = paddle.distributed.fleet.DistributedStrategy()
        strategy2.load_from_prototxt("dist_strategy.prototxt")
        self.assertEqual(strategy.dgc, strategy2.dgc)

    def test_build_strategy(self):
        build_strategy = paddle.fluid.BuildStrategy()
        build_strategy.enable_sequential_execution = True
        build_strategy.nccl_comm_num = 10
        build_strategy.use_hierarchical_allreduce = True
        build_strategy.hierarchical_allreduce_inter_nranks = 1
        build_strategy.fuse_elewise_add_act_ops = True
        build_strategy.fuse_bn_act_ops = True
        build_strategy.enable_auto_fusion = True
        build_strategy.fuse_relu_depthwise_conv = True
        build_strategy.fuse_broadcast_ops = True
        build_strategy.fuse_all_optimizer_ops = True
        build_strategy.sync_batch_norm = True
        build_strategy.enable_inplace = True
        build_strategy.fuse_all_reduce_ops = True
        build_strategy.enable_backward_optimizer_op_deps = True
        build_strategy.trainers_endpoints = ["1", "2"]

        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.build_strategy = build_strategy

    def test_execution_strategy(self):
        exe_strategy = paddle.fluid.ExecutionStrategy()
        exe_strategy.num_threads = 10
        exe_strategy.num_iteration_per_drop_scope = 10
        exe_strategy.num_iteration_per_run = 10

        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.execution_strategy = exe_strategy

    def test_unknown_strategy(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        with self.assertRaises(TypeError):
            strategy.unknown_key = 'UNK'

    def test_cudnn_exhaustive_search(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.cudnn_exhaustive_search = False
        self.assertEqual(strategy.cudnn_exhaustive_search, False)
        strategy.cudnn_exhaustive_search = "True"
        self.assertEqual(strategy.cudnn_exhaustive_search, False)

    def test_cudnn_batchnorm_spatial_persistent(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.cudnn_batchnorm_spatial_persistent = False
        self.assertEqual(strategy.cudnn_batchnorm_spatial_persistent, False)
        strategy.cudnn_batchnorm_spatial_persistent = "True"
        self.assertEqual(strategy.cudnn_batchnorm_spatial_persistent, False)

    def test_conv_workspace_size_limit(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.conv_workspace_size_limit = 1000
        self.assertEqual(strategy.conv_workspace_size_limit, 1000)
        strategy.conv_workspace_size_limit = "400"
        self.assertEqual(strategy.conv_workspace_size_limit, 1000)
        strategy._enable_env()

    def test_distributed_strategy_repr(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.recompute = True
        strategy.recompute_configs = {"checkpoints": ["a1", "a2", "a3"]}
        strategy.amp = True
        strategy.localsgd = True
        print(str(strategy))


if __name__ == '__main__':
    unittest.main()
