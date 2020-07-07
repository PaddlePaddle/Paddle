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
        strategy = paddle.fleet.DistributedStrategy()
        strategy.amp = True
        self.assertEqual(strategy.amp, True)
        strategy.amp = False
        self.assertEqual(strategy.amp, False)
        strategy.amp = "True"
        self.assertEqual(strategy.amp, False)

    def test_amp_loss_scaling(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.amp_loss_scaling = 32768
        self.assertEqual(strategy.amp_loss_scaling, 32768)
        strategy.amp_loss_scaling = 0.1
        self.assertEqual(strategy.amp_loss_scaling, 32768)

    def test_recompute(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.recompute = True
        self.assertEqual(strategy.recompute, True)
        strategy.recompute = False
        self.assertEqual(strategy.recompute, False)
        strategy.recompute = "True"
        self.assertEqual(strategy.recompute, False)

    def test_recompute_checkpoints(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.recompute_checkpoints = ["var1", "var2", "var3"]
        self.assertEqual(len(strategy.recompute_checkpoints), 3)
        import paddle.fluid as fluid
        program = fluid.Program()
        cur_block = program.current_block()
        var1 = cur_block.create_var(name="var4", shape=[1, 1], dtype="int32")
        var2 = cur_block.create_var(name="var5", shape=[1, 1], dtype="int32")
        var3 = cur_block.create_var(name="var6", shape=[1, 1], dtype="int32")
        strategy.recompute_checkpoints = [var1, var2, var3]
        self.assertEqual(len(strategy.recompute_checkpoints), 3)
        self.assertEqual(strategy.recompute_checkpoints[0], "var4")
        strategy.recompute_checkpoints = [var1, "var2", var3]
        self.assertEqual(strategy.recompute_checkpoints[1], "var5")

    def test_pipeline(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.pipeline = True
        self.assertEqual(strategy.pipeline, True)
        strategy.pipeline = False
        self.assertEqual(strategy.pipeline, False)
        strategy.pipeline = "True"
        self.assertEqual(strategy.pipeline, False)

    def test_pipeline_micro_batch(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.pipeline_micro_batch = 1
        self.assertEqual(strategy.pipeline_micro_batch, 1)
        strategy.pipeline_micro_batch = 0.1
        self.assertEqual(strategy.pipeline_micro_batch, 1)

    def test_localsgd(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.localsgd = True
        self.assertEqual(strategy.localsgd, True)
        strategy.localsgd = False
        self.assertEqual(strategy.localsgd, False)
        strategy.localsgd = "True"
        self.assertEqual(strategy.localsgd, False)

    def test_localsgd_k_step(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.localsgd_k_step = 1
        self.assertEqual(strategy.localsgd_k_step, 1)
        strategy.localsgd_k_step = "2"
        self.assertEqual(strategy.localsgd_k_step, 1)

    def test_dgc(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.dgc = True
        self.assertEqual(strategy.dgc, True)
        strategy.dgc = False
        self.assertEqual(strategy.dgc, False)
        strategy.dgc = "True"
        self.assertEqual(strategy.dgc, False)

    def test_hierachical_allreduce(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.hierachical_allreduce = True
        self.assertEqual(strategy.hierachical_allreduce, True)
        strategy.hierachical_allreduce = False
        self.assertEqual(strategy.hierachical_allreduce, False)
        strategy.hierachical_allreduce = "True"
        self.assertEqual(strategy.hierachical_allreduce, False)

    def test_nccl_comm_num(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.nccl_comm_num = 1
        self.assertEqual(strategy.nccl_comm_num, 1)
        strategy.nccl_comm_num = "2"
        self.assertEqual(strategy.nccl_comm_num, 1)

    def test_gradient_merge(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.gradient_merge = True
        self.assertEqual(strategy.gradient_merge, True)
        strategy.gradient_merge = False
        self.assertEqual(strategy.gradient_merge, False)
        strategy.gradient_merge = "True"
        self.assertEqual(strategy.gradient_merge, False)

    def test_gradient_merge_k_step(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.gradient_merge_k_step = 1
        self.assertEqual(strategy.gradient_merge_k_step, 1)
        strategy.gradient_merge_k_step = "2"
        self.assertEqual(strategy.gradient_merge_k_step, 1)

    def test_sequential_execution(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.sequential_execution = True
        self.assertEqual(strategy.sequential_execution, True)
        strategy.sequential_execution = False
        self.assertEqual(strategy.sequential_execution, False)
        strategy.sequential_execution = "True"
        self.assertEqual(strategy.sequential_execution, False)

    def test_lars(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.lars = True
        self.assertEqual(strategy.lars, True)
        strategy.lars = False
        self.assertEqual(strategy.lars, False)
        strategy.lars = "True"
        self.assertEqual(strategy.lars, False)

    def test_lamb(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.lamb = True
        self.assertEqual(strategy.lamb, True)
        strategy.lamb = False
        self.assertEqual(strategy.lamb, False)
        strategy.lamb = "True"
        self.assertEqual(strategy.lamb, False)

    def test_fuse_elewise_add_act_ops(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.fuse_elewise_add_act_ops = True
        self.assertEqual(strategy.fuse_elewise_add_act_ops, True)
        strategy.fuse_elewise_add_act_ops = False
        self.assertEqual(strategy.fuse_elewise_add_act_ops, False)
        strategy.fuse_elewise_add_act_ops = "True"
        self.assertEqual(strategy.fuse_elewise_add_act_ops, False)

    def test_fuse_bn_act_ops(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.fuse_bn_act_ops = True
        self.assertEqual(strategy.fuse_bn_act_ops, True)
        strategy.fuse_bn_act_ops = False
        self.assertEqual(strategy.fuse_bn_act_ops, False)
        strategy.fuse_bn_act_ops = "True"
        self.assertEqual(strategy.fuse_bn_act_ops, False)

    def test_enable_auto_fusion(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.enable_auto_fusion = True
        self.assertEqual(strategy.enable_auto_fusion, True)
        strategy.enable_auto_fusion = False
        self.assertEqual(strategy.enable_auto_fusion, False)
        strategy.enable_auto_fusion = "True"
        self.assertEqual(strategy.enable_auto_fusion, False)

    def test_fuse_relu_depthwise_conv(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.fuse_relu_depthwise_conv = True
        self.assertEqual(strategy.fuse_relu_depthwise_conv, True)
        strategy.fuse_relu_depthwise_conv = False
        self.assertEqual(strategy.fuse_relu_depthwise_conv, False)
        strategy.fuse_relu_depthwise_conv = "True"
        self.assertEqual(strategy.fuse_relu_depthwise_conv, False)

    def test_enable_inplace(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.enable_inplace = True
        self.assertEqual(strategy.enable_inplace, True)
        strategy.enable_inplace = False
        self.assertEqual(strategy.enable_inplace, False)
        strategy.enable_inplace = "True"
        self.assertEqual(strategy.enable_inplace, False)

    def test_fuse_all_reduce_ops(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.fuse_all_reduce_ops = True
        self.assertEqual(strategy.fuse_all_reduce_ops, True)
        strategy.fuse_all_reduce_ops = False
        self.assertEqual(strategy.fuse_all_reduce_ops, False)
        strategy.fuse_all_reduce_ops = "True"
        self.assertEqual(strategy.fuse_all_reduce_ops, False)

    def test_num_iteration_per_drop_scope(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.num_iteration_per_drop_scope = 1
        self.assertEqual(strategy.num_iteration_per_drop_scope, 1)
        strategy.num_iteration_per_drop_scope = 0.1
        self.assertEqual(strategy.num_iteration_per_drop_scope, 1)

    def test_sync_batch_norm(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.sync_batch_norm = True
        self.assertEqual(strategy.sync_batch_norm, True)
        strategy.sync_batch_norm = False
        self.assertEqual(strategy.sync_batch_norm, False)
        strategy.sync_batch_norm = "True"
        self.assertEqual(strategy.sync_batch_norm, False)

    def test_fuse_all_optimizer_ops(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.fuse_all_optimizer_ops = True
        self.assertEqual(strategy.fuse_all_optimizer_ops, True)
        strategy.fuse_all_optimizer_ops = False
        self.assertEqual(strategy.fuse_all_optimizer_ops, False)
        strategy.fuse_all_optimizer_ops = "True"
        self.assertEqual(strategy.fuse_all_optimizer_ops, False)

    def test_sync(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.sync = True
        self.assertEqual(strategy.sync, True)
        strategy.sync = False
        self.assertEqual(strategy.sync, False)
        strategy.sync = "True"
        self.assertEqual(strategy.sync, False)

    def test_async_k_step(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.async_k_step = 10000
        self.assertEqual(strategy.async_k_step, 10000)
        strategy.async_k_step = 0.1
        self.assertEqual(strategy.async_k_step, 10000)

    def test_send_queue_size(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.send_queue_size = 10000
        self.assertEqual(strategy.send_queue_size, 10000)
        strategy.send_queue_size = 0.1
        self.assertEqual(strategy.send_queue_size, 10000)

    def test_independent_recv_thread(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.independent_recv_thread = True
        self.assertEqual(strategy.independent_recv_thread, True)
        strategy.independent_recv_thread = False
        self.assertEqual(strategy.independent_recv_thread, False)
        strategy.independent_recv_thread = "True"
        self.assertEqual(strategy.independent_recv_thread, False)

    def test_min_send_grad_num_before_recv(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.min_send_grad_num_before_recv = 10000
        self.assertEqual(strategy.min_send_grad_num_before_recv, 10000)
        strategy.min_send_grad_num_before_recv = 0.1
        self.assertEqual(strategy.min_send_grad_num_before_recv, 10000)

    def test_thread_pool_size(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.thread_pool_size = 10000
        self.assertEqual(strategy.thread_pool_size, 10000)
        strategy.thread_pool_size = 0.1
        self.assertEqual(strategy.thread_pool_size, 10000)

    def test_send_wait_times(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.send_wait_times = 10000
        self.assertEqual(strategy.send_wait_times, 10000)
        strategy.send_wait_times = 0.1
        self.assertEqual(strategy.send_wait_times, 10000)

    def test_runtime_split_send_recv(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.runtime_split_send_recv = True
        self.assertEqual(strategy.runtime_split_send_recv, True)
        strategy.runtime_split_send_recv = False
        self.assertEqual(strategy.runtime_split_send_recv, False)
        strategy.runtime_split_send_recv = "True"
        self.assertEqual(strategy.runtime_split_send_recv, False)

    def use_thread_barrier(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.thread_barrier = True
        self.assertEqual(strategy.thread_barrier, True)
        strategy.thread_barrier = False
        self.assertEqual(strategy.thread_barrier, False)
        strategy.thread_barrier = "True"
        self.assertEqual(strategy.thread_barrier, False)

    def test_enable_backward_optimizer_op_deps(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.enable_backward_optimizer_op_deps = True
        self.assertEqual(strategy.enable_backward_optimizer_op_deps, True)
        strategy.enable_backward_optimizer_op_deps = False
        self.assertEqual(strategy.enable_backward_optimizer_op_deps, False)
        strategy.enable_backward_optimizer_op_deps = "True"
        self.assertEqual(strategy.enable_backward_optimizer_op_deps, False)

    def test_elastic(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.elastic = True
        self.assertEqual(strategy.elastic, True)
        strategy.elastic = False
        self.assertEqual(strategy.elastic, False)
        strategy.elastic = "True"
        self.assertEqual(strategy.elastic, False)

    def test_auto(self):
        strategy = paddle.fleet.DistributedStrategy()
        strategy.auto = True
        self.assertEqual(strategy.auto, True)
        strategy.auto = False
        self.assertEqual(strategy.auto, False)
        strategy.auto = "True"
        self.assertEqual(strategy.auto, False)


if __name__ == '__main__':
    unittest.main()
