# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
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

import os
import unittest

from fleet_meta_optimizer_base import TestFleetMetaOptimizer

import paddle
import paddle.distributed.fleet.meta_optimizers.sharding as sharding
from paddle.distributed.fleet.meta_optimizers.common import is_loss_grad_op

paddle.enable_static()


class TestFleetShardingMetaOptimizer(TestFleetMetaOptimizer):
    def test_sharding_optimizer(self):
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'sharding')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        parameters = [
            x.name for x in train_prog.list_vars() if x.persistable is True
        ]
        ops = [op.type for op in avg_cost.block.ops]
        vars = [x.name for x in train_prog.list_vars()]
        self.assertIn('@BroadCast', ''.join(vars))
        self.assertEqual(
            set(parameters),
            set(
                [
                    "fc_1.b_0",
                    "fc_2.b_0",
                    "fc_2.w_0",
                    "fc_1.b_0_velocity_0",
                    "fc_2.b_0_velocity_0",
                    "fc_2.w_0_velocity_0",
                    "learning_rate_0",
                ]
            ),
        )

        self.assertEqual(
            ops,
            [
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'c_sync_calc_stream',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_sync_comm_stream',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'softmax',
                'softmax_with_cross_entropy',
                'reduce_mean',
                'fill_constant',
                'reduce_mean_grad',
                'softmax_with_cross_entropy_grad',
                'softmax_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'c_sync_calc_stream',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_sync_comm_stream',
                'momentum',
                'momentum',
                'momentum',
            ],
        )

    def test_sharding_amp_optimizer(self):
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'sharding')
        self.set_strategy(strategy, 'amp')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        ops = [op.type for op in avg_cost.block.ops]
        vars = [x.name for x in train_prog.list_vars()]
        parameters = [x.name for x in train_prog.list_vars() if x.persistable]
        self.assertIn('@BroadCast', ''.join(vars))
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)
        self.assertEqual(
            set(parameters),
            set(
                [
                    "fc_1.b_0",
                    "fc_2.b_0",
                    "fc_2.w_0",
                    "fc_1.b_0_velocity_0",
                    "fc_2.b_0_velocity_0",
                    "fc_2.w_0_velocity_0",
                    "learning_rate_0",
                    "loss_scaling_0",
                    "num_bad_steps_0",
                    "num_good_steps_0",
                ]
            ),
        )

        self.assertEqual(
            ops,
            [
                'cast',
                'cast',
                'cast',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'c_sync_calc_stream',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_sync_comm_stream',
                'cast',
                'mul',
                'elementwise_add',
                'cast',
                'tanh',
                'cast',
                'mul',
                'elementwise_add',
                'cast',
                'tanh',
                'cast',
                'mul',
                'elementwise_add',
                'softmax',
                'cast',
                'softmax_with_cross_entropy',
                'reduce_mean',
                'elementwise_mul',
                'fill_constant',
                'elementwise_mul_grad',
                'reduce_mean_grad',
                'softmax_with_cross_entropy_grad',
                'cast',
                'softmax_grad',
                'elementwise_add_grad',
                'mul_grad',
                'cast',
                'tanh_grad',
                'cast',
                'elementwise_add_grad',
                'mul_grad',
                'cast',
                'tanh_grad',
                'cast',
                'elementwise_add_grad',
                'mul_grad',
                'c_sync_calc_stream',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_sync_comm_stream',
                'cast',
                'cast',
                'cast',
                'check_finite_and_unscale',
                'cast',
                'c_allreduce_max',
                'cast',
                'update_loss_scaling',
                'momentum',
                'momentum',
                'momentum',
            ],
        )

    def test_sharding_recompute_optimizer(self):
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'sharding')
        self.set_strategy(strategy, 'recompute')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)

        ops = [op.type for op in avg_cost.block.ops]
        vars = [x.name for x in train_prog.list_vars()]
        parameters = [x.name for x in train_prog.list_vars() if x.persistable]

        self.assertIn('@BroadCast', ''.join(vars))
        self.assertIn('subprog', ''.join(vars))
        self.assertEqual(
            set(parameters),
            set(
                [
                    "fc_1.b_0",
                    "fc_2.b_0",
                    "fc_2.w_0",
                    "fc_1.b_0_velocity_0",
                    "fc_2.b_0_velocity_0",
                    "fc_2.w_0_velocity_0",
                    "learning_rate_0",
                ]
            ),
        )

        self.assertEqual(
            ops,
            [
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'c_sync_calc_stream',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_sync_comm_stream',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'softmax',
                'softmax_with_cross_entropy',
                'reduce_mean',
                'fill_constant',
                'reduce_mean_grad',
                'softmax_with_cross_entropy_grad',
                'softmax_grad',
                'elementwise_add_grad',
                'mul_grad',
                'mul',
                'elementwise_add',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'mul',
                'elementwise_add',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'c_sync_calc_stream',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_sync_comm_stream',
                'momentum',
                'momentum',
                'momentum',
            ],
        )

    def test_sharding_amp_recompute_optimizer(self):
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'sharding')
        self.set_strategy(strategy, 'recompute')
        self.set_strategy(strategy, 'amp')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)

        ops = [op.type for op in avg_cost.block.ops]
        vars = [x.name for x in train_prog.list_vars()]
        parameters = [x.name for x in train_prog.list_vars() if x.persistable]

        self.assertIn('@BroadCast', ''.join(vars))
        self.assertIn('subprog', ''.join(vars))
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)

        self.assertEqual(
            set(parameters),
            set(
                [
                    "fc_1.b_0",
                    "fc_2.b_0",
                    "fc_2.w_0",
                    "fc_1.b_0_velocity_0",
                    "fc_2.b_0_velocity_0",
                    "fc_2.w_0_velocity_0",
                    "learning_rate_0",
                    "loss_scaling_0",
                    "num_bad_steps_0",
                    "num_good_steps_0",
                ]
            ),
        )
        self.assertEqual(
            ops,
            [
                'cast',
                'cast',
                'cast',
                'cast',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'c_sync_calc_stream',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_sync_comm_stream',
                'cast',
                'mul',
                'elementwise_add',
                'cast',
                'tanh',
                'cast',
                'mul',
                'elementwise_add',
                'cast',
                'tanh',
                'cast',
                'mul',
                'elementwise_add',
                'softmax',
                'cast',
                'softmax_with_cross_entropy',
                'reduce_mean',
                'elementwise_mul',
                'fill_constant',
                'elementwise_mul_grad',
                'reduce_mean_grad',
                'softmax_with_cross_entropy_grad',
                'cast',
                'softmax_grad',
                'elementwise_add_grad',
                'mul_grad',
                'cast',
                'cast',
                'mul',
                'elementwise_add',
                'cast',
                'tanh_grad',
                'cast',
                'elementwise_add_grad',
                'mul_grad',
                'cast',
                'mul',
                'elementwise_add',
                'cast',
                'tanh_grad',
                'cast',
                'elementwise_add_grad',
                'mul_grad',
                'c_sync_calc_stream',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_sync_comm_stream',
                'cast',
                'cast',
                'cast',
                'check_finite_and_unscale',
                'cast',
                'c_allreduce_max',
                'cast',
                'update_loss_scaling',
                'momentum',
                'momentum',
                'momentum',
            ],
        )

    def test_sharding_amp_asp_optimizer(self):
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'sharding')
        self.set_strategy(strategy, 'amp')
        self.set_strategy(strategy, 'asp')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)

        ops = [op.type for op in avg_cost.block.ops]
        vars = [x.name for x in train_prog.list_vars()]
        parameters = [x.name for x in train_prog.list_vars() if x.persistable]

        self.assertIn('@BroadCast', ''.join(vars))
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)

        self.assertEqual(
            set(parameters),
            set(
                [
                    'fc_2.b_0',
                    'num_good_steps_0',
                    'fc_2.w_0',
                    'loss_scaling_0',
                    'num_bad_steps_0',
                    'fc_2.w_0_velocity_0',
                    'fc_2.w_0.asp_mask',
                    'learning_rate_0',
                    'fc_1.b_0',
                    'fc_1.w_0.asp_mask',
                    'fc_0.w_0.asp_mask',
                    'fc_1.b_0_velocity_0',
                    'fc_2.b_0_velocity_0',
                ]
            ),
        )
        self.assertEqual(
            ops,
            [
                'cast',
                'cast',
                'cast',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'c_sync_calc_stream',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_sync_comm_stream',
                'cast',
                'mul',
                'elementwise_add',
                'cast',
                'tanh',
                'cast',
                'mul',
                'elementwise_add',
                'cast',
                'tanh',
                'cast',
                'mul',
                'elementwise_add',
                'softmax',
                'cast',
                'softmax_with_cross_entropy',
                'reduce_mean',
                'elementwise_mul',
                'fill_constant',
                'elementwise_mul_grad',
                'reduce_mean_grad',
                'softmax_with_cross_entropy_grad',
                'cast',
                'softmax_grad',
                'elementwise_add_grad',
                'mul_grad',
                'cast',
                'tanh_grad',
                'cast',
                'elementwise_add_grad',
                'mul_grad',
                'cast',
                'tanh_grad',
                'cast',
                'elementwise_add_grad',
                'mul_grad',
                'c_sync_calc_stream',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_sync_comm_stream',
                'cast',
                'cast',
                'cast',
                'check_finite_and_unscale',
                'cast',
                'c_allreduce_max',
                'cast',
                'update_loss_scaling',
                'momentum',
                'momentum',
                'momentum',
                'elementwise_mul',
            ],
        )

    def test_sharding_weight_decay(self):
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'sharding')
        regularization = paddle.fluid.regularizer.L2Decay(0.0001)
        self.optimizer(
            avg_cost,
            strategy,
            train_prog,
            startup_prog,
            regularization=regularization,
        )
        parameters = [x.name for x in train_prog.list_vars() if x.persistable]
        ops = [op.type for op in avg_cost.block.ops]
        vars = [x.name for x in train_prog.list_vars()]
        self.assertIn('@BroadCast', ''.join(vars))
        self.assertEqual(
            set(parameters),
            set(
                [
                    "fc_1.b_0",
                    "fc_2.b_0",
                    "fc_2.w_0",
                    "fc_1.b_0_velocity_0",
                    "fc_2.b_0_velocity_0",
                    "fc_2.w_0_velocity_0",
                    "learning_rate_0",
                ]
            ),
        )

        self.assertEqual(
            ops,
            [
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'c_sync_calc_stream',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_sync_comm_stream',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'softmax',
                'softmax_with_cross_entropy',
                'reduce_mean',
                'fill_constant',
                'reduce_mean_grad',
                'softmax_with_cross_entropy_grad',
                'softmax_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'c_sync_calc_stream',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_sync_comm_stream',
                'scale',
                'sum',
                'scale',
                'sum',
                'scale',
                'sum',
                'momentum',
                'momentum',
                'momentum',
            ],
        )

    def test_sharding_gradient_clip(self):
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'sharding')
        clip = paddle.fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0)
        self.optimizer(
            avg_cost, strategy, train_prog, startup_prog, grad_clip=clip
        )
        parameters = [x.name for x in train_prog.list_vars() if x.persistable]
        ops = [op.type for op in avg_cost.block.ops]
        vars = [x.name for x in train_prog.list_vars()]
        self.assertIn('@BroadCast', ''.join(vars))
        self.assertEqual(
            set(parameters),
            set(
                [
                    "fc_1.b_0",
                    "fc_2.b_0",
                    "fc_2.w_0",
                    "fc_1.b_0_velocity_0",
                    "fc_2.b_0_velocity_0",
                    "fc_2.w_0_velocity_0",
                    "learning_rate_0",
                ]
            ),
        )

        self.assertEqual(
            ops,
            [
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'c_sync_calc_stream',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_sync_comm_stream',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'softmax',
                'softmax_with_cross_entropy',
                'reduce_mean',
                'fill_constant',
                'reduce_mean_grad',
                'softmax_with_cross_entropy_grad',
                'softmax_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'c_sync_calc_stream',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_sync_comm_stream',
                'squared_l2_norm',
                'squared_l2_norm',
                'squared_l2_norm',
                'sum',
                'c_allreduce_sum',
                'sqrt',
                'fill_constant',
                'elementwise_max',
                'elementwise_div',
                'elementwise_mul',
                'elementwise_mul',
                'elementwise_mul',
                'momentum',
                'momentum',
                'momentum',
            ],
        )

    def test_sharding_clone_for_test(self):
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'sharding')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        sharding.utils.comm_analyse(train_prog)
        test_prog = train_prog.clone(for_test=True)
        # assume sharding_ring_id = 1
        sharding.utils.add_sync_comm(test_prog, 1)
        ops = [op.type for op in test_prog.global_block().ops]

        self.assertEqual(
            ops,
            [
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'c_sync_calc_stream',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_sync_comm_stream',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'softmax',
                'softmax_with_cross_entropy',
                'reduce_mean',
            ],
        )


class TestFleetShardingHybridOptimizer(TestFleetMetaOptimizer):
    def setUp(self):
        os.environ["PADDLE_TRAINER_ID"] = "3"
        os.environ[
            "PADDLE_TRAINER_ENDPOINTS"
        ] = "127.0.0.1:36001,127.0.0.1:36002,127.0.0.1:36003,127.0.0.1:36004"

        # pre-assigned ring id
        self.mp_ring_id = 0
        self.sharding_ring_id = 1
        self.dp_ring_id = 2
        self.global_ring_id = 3
        self.pp_pair_ring_id = 20

    def test_sharding_with_mp(self):
        # NOTE(JZ-LIANG) MP parallelism need user to build model with MP API
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, _ = self.net(train_prog, startup_prog)
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.sharding = True
        strategy.sharding_configs = {
            "sharding_segment_strategy": "segment_broadcast_MB",
            "segment_broadcast_MB": 0.2,
            "segment_anchors": None,
            "sharding_degree": 2,
            "hybrid_dp": False,
            "gradient_merge_acc_step": 1,
            "mp_degree": 2,
        }
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        startup_prog_ops = startup_prog.global_block().ops
        main_prog_ops = train_prog.global_block().ops

        # should has ring id for MP
        created_ring_ids = [
            op.desc.attr("ring_id")
            for op in startup_prog_ops
            if op.type == "c_comm_init"
        ]
        self.assertIn(self.mp_ring_id, created_ring_ids)

        # check correctness of MP group
        sharding_group_waiting_port = None
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_0"
            ):
                sharding_group_waiting_ports = op.desc.attr("other_endpoints")

        self.assertEqual(sharding_group_waiting_ports, ['127.0.0.1:36003'])

        # check correctness of sharding group
        sharding_group_waiting_port = None
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_1"
            ):
                dp_group_waiting_ports = op.desc.attr("other_endpoints")

        self.assertEqual(dp_group_waiting_ports, ['127.0.0.1:36002'])

    def test_sharding_hybrid_dp(self):
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, _ = self.net(train_prog, startup_prog)
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.sharding = True
        strategy.sharding_configs = {
            "sharding_segment_strategy": "segment_broadcast_MB",
            "segment_broadcast_MB": 0.2,
            "segment_anchors": None,
            "sharding_degree": 2,
            "dp_degree": 2,
            "hybrid_dp": True,
            "gradient_merge_acc_step": 1,
            "mp_degree": 1,
        }

        strategy.fuse_all_reduce_ops = False
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        startup_prog_ops = startup_prog.global_block().ops
        main_prog_ops = train_prog.global_block().ops

        # check ring id for outter dp
        created_ring_ids = [
            op.desc.attr("ring_id")
            for op in startup_prog_ops
            if op.type == "c_comm_init"
        ]
        self.assertIn(self.dp_ring_id, created_ring_ids)

        # check correctness of sharding group
        sharding_group_waiting_port = None
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_0"
            ):
                sharding_group_waiting_ports = op.desc.attr("other_endpoints")

        self.assertEqual(sharding_group_waiting_ports, ['127.0.0.1:36003'])

        # check correctness of dp group
        sharding_group_waiting_port = None
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_1"
            ):
                dp_group_waiting_ports = op.desc.attr("other_endpoints")
        self.assertEqual(dp_group_waiting_ports, ['127.0.0.1:36002'])

        # check loss scale for sharding hybrid dp
        for op in main_prog_ops:
            if is_loss_grad_op(op):
                self.assertEqual(op.type, 'fill_constant')
                self.assertTrue(op.has_attr('value'))
                scale = (
                    strategy.sharding_configs['sharding_degree']
                    * strategy.sharding_configs['dp_degree']
                )
                loss_scale = 1.0 / scale
                self.assertAlmostEqual(float(op.attr('value')), loss_scale)

        # check program (allreudce)
        ops = [op.type for op in main_prog_ops]
        self.assertEqual(
            ops,
            [
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'c_sync_calc_stream',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_sync_comm_stream',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'softmax',
                'softmax_with_cross_entropy',
                'reduce_mean',
                'fill_constant',
                'reduce_mean_grad',
                'softmax_with_cross_entropy_grad',
                'softmax_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'c_sync_calc_stream',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_sync_comm_stream',
                'c_allreduce_sum',
                'c_allreduce_sum',
                'c_allreduce_sum',
                'c_sync_comm_stream',
                'momentum',
                'momentum',
                'momentum',
            ],
        )

    def test_sharding_hybrid_dp_gm(self):
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, _ = self.net(train_prog, startup_prog)
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.sharding = True
        strategy.sharding_configs = {
            "sharding_segment_strategy": "segment_broadcast_MB",
            "segment_broadcast_MB": 0.2,
            "segment_anchors": None,
            "sharding_degree": 2,
            "dp_degree": 2,
            "hybrid_dp": True,
            "gradient_merge_acc_step": 4,
            "mp_degree": 1,
        }
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        startup_prog_ops = startup_prog.global_block().ops
        main_prog_ops = train_prog.global_block().ops

        # check ring id for outter dp
        created_ring_ids = [
            op.desc.attr("ring_id")
            for op in startup_prog_ops
            if op.type == "c_comm_init"
        ]
        self.assertIn(self.dp_ring_id, created_ring_ids)

        # check correctness of sharding group
        sharding_group_waiting_port = None
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_0"
            ):
                sharding_group_waiting_ports = op.desc.attr("other_endpoints")

        self.assertEqual(sharding_group_waiting_ports, ['127.0.0.1:36003'])

        # check correctness of dp group
        sharding_group_waiting_port = None
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_1"
            ):
                dp_group_waiting_ports = op.desc.attr("other_endpoints")
        self.assertEqual(dp_group_waiting_ports, ['127.0.0.1:36002'])

        # check program
        fw_bw_ops = [op.type for op in train_prog.blocks[0].ops]
        opt_ops = [op.type for op in train_prog.blocks[2].ops]
        self.assertEqual(
            fw_bw_ops,
            [
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'c_sync_calc_stream',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_sync_comm_stream',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'softmax',
                'softmax_with_cross_entropy',
                'reduce_mean',
                'fill_constant',
                'reduce_mean_grad',
                'softmax_with_cross_entropy_grad',
                'softmax_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'c_sync_calc_stream',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_sync_comm_stream',
                'elementwise_add',
                'elementwise_add',
                'elementwise_add',
                'increment',
                'elementwise_mod',
                'equal',
                'conditional_block',
            ],
        )
        self.assertEqual(
            opt_ops,
            [
                'c_allreduce_sum',
                'c_allreduce_sum',
                'c_allreduce_sum',
                'scale',
                'scale',
                'scale',
                'momentum',
                'momentum',
                'momentum',
                'fill_constant',
                'fill_constant',
                'fill_constant',
            ],
        )

        # # check loss scale for gradient merge
        scale_ = -1
        for op in train_prog.blocks[2].ops:
            if op.type == "scale":
                scale_ = float(op.desc.attr("scale"))
                self.assertEqual(scale_, 0.25)

    def test_sharding_with_pp(self):
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, strategy = self.pp_net(train_prog, startup_prog)
        strategy.sharding = True
        strategy.sharding_configs = {
            "sharding_segment_strategy": "segment_broadcast_MB",
            "segment_broadcast_MB": 0.1,
            "sharding_degree": 2,
            "hybrid_dp": False,
            "gradient_merge_acc_step": 4,
            "mp_degree": 1,
            "pp_degree": 2,
        }
        strategy.pipeline = True
        strategy.pipeline_configs = {
            "schedule_mode": "1F1B",
            "micro_batch_size": 2,
            "accumulate_steps": 4,
        }
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        train_prog = train_prog._pipeline_opt['section_program']
        startup_prog = startup_prog._pipeline_opt['startup_program']

        startup_prog_ops = startup_prog.global_block().ops
        main_prog_ops = train_prog.global_block().ops

        # check program
        startup_prog_op_types = [op.type for op in startup_prog_ops]
        main_prog_op_types = [op.type for op in main_prog_ops]
        print(startup_prog_op_types)
        # global, sharding, pp_send, pp_recv
        self.assertEqual(
            startup_prog_op_types,
            [
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
            ],
        )

        self.assertEqual(
            main_prog_op_types,
            [
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'c_sync_calc_stream',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_sync_comm_stream',
                'recv_v2',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'softmax',
                'softmax_with_cross_entropy',
                'reduce_mean',
                'fill_constant',
                'reduce_mean_grad',
                'softmax_with_cross_entropy_grad',
                'softmax_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'c_sync_calc_stream',
                'send_v2',
                'c_sync_calc_stream',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_sync_comm_stream',
                'fill_constant',
                'sum',
                'fill_constant',
                'sum',
                'fill_constant',
                'sum',
                'fill_constant',
                'sum',
                'fill_constant',
                'sum',
                'c_sync_comm_stream',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
            ],
        )

        # should has ring id for pp
        created_ring_ids = [
            op.desc.attr("ring_id")
            for op in startup_prog_ops
            if op.type == "c_comm_init"
        ]
        self.assertIn(self.sharding_ring_id, created_ring_ids)
        self.assertIn(self.pp_pair_ring_id, created_ring_ids)

        # check correctness of pp group
        sharding_group_waiting_port = None
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_0"
            ):
                sharding_group_waiting_ports = op.desc.attr("other_endpoints")

        self.assertEqual(sharding_group_waiting_ports, ['127.0.0.1:36003'])

        # check correctness of sharding group
        sharding_group_waiting_port = None
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_1"
            ):
                dp_group_waiting_ports = op.desc.attr("other_endpoints")

        self.assertEqual(dp_group_waiting_ports, ['127.0.0.1:36002'])

    def test_sharding_dp_with_allreduce_fuse(self):
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, _ = self.net(train_prog, startup_prog)
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.sharding = True
        strategy.sharding_configs = {
            "sharding_segment_strategy": "segment_broadcast_MB",
            "segment_broadcast_MB": 0.1,
            "segment_anchors": None,
            "sharding_degree": 2,
            "dp_degree": 2,
            "hybrid_dp": True,
            "gradient_merge_acc_step": 1,
            "mp_degree": 1,
        }
        strategy.fuse_all_reduce_ops = True
        strategy.fuse_grad_size_in_MB = 2
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)

        main_prog_ops = train_prog.global_block().ops
        main_prog_op_types = [op.type for op in main_prog_ops]

        assert 'c_allreduce_sum' in main_prog_op_types
        assert 'coalesce_tensor' in main_prog_op_types

        for op in main_prog_ops:
            if op.type == 'c_allreduce_sum':
                assert 'FusedGrad' in op.input_arg_names[0]

    def test_hybrid_with_mp_pp_amp_gclip(self):
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, strategy = self.pp_net(train_prog, startup_prog)
        self.set_strategy(strategy, 'amp')
        strategy.sharding = True
        strategy.sharding_configs = {
            "sharding_degree": 1,
            "mp_degree": 2,
            "pp_degree": 2,
            "dp_degree": 1,
        }
        strategy.pipeline = True
        strategy.pipeline_configs = {
            "schedule_mode": "1F1B",
            "micro_batch_size": 2,
            "accumulate_steps": 4,
        }
        clip = paddle.fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0)
        self.optimizer(
            avg_cost, strategy, train_prog, startup_prog, grad_clip=clip
        )
        train_prog = train_prog._pipeline_opt['section_program']
        startup_prog = startup_prog._pipeline_opt['startup_program']

        startup_prog_ops = startup_prog.global_block().ops
        main_prog_ops = train_prog.global_block().ops

        # check program
        startup_prog_op_types = [op.type for op in startup_prog_ops]
        main_prog_op_types = [op.type for op in main_prog_ops]

        # ring: mp, pp_group, pp_pair, pp_pair
        self.assertEqual(
            startup_prog_op_types,
            [
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
            ],
        )

        self.assertEqual(
            main_prog_op_types,
            [
                'partial_recv',
                'partial_allgather',
                'cast',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'cast',
                'tanh',
                'cast',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'cast',
                'tanh',
                'cast',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'cast',
                'tanh',
                'cast',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'softmax',
                'cast',
                'softmax_with_cross_entropy',
                'reduce_mean',
                'elementwise_mul',
                'fill_constant',
                'elementwise_mul_grad',
                'reduce_mean_grad',
                'softmax_with_cross_entropy_grad',
                'cast',
                'softmax_grad',
                'elementwise_add_grad',
                'mul_grad',
                'cast',
                'tanh_grad',
                'cast',
                'elementwise_add_grad',
                'mul_grad',
                'cast',
                'tanh_grad',
                'cast',
                'elementwise_add_grad',
                'mul_grad',
                'cast',
                'tanh_grad',
                'cast',
                'elementwise_add_grad',
                'mul_grad',
                'cast',
                'c_sync_calc_stream',
                'partial_send',
                'fill_constant',
                'cast',
                'sum',
                'fill_constant',
                'cast',
                'sum',
                'fill_constant',
                'cast',
                'sum',
                'fill_constant',
                'cast',
                'sum',
                'fill_constant',
                'cast',
                'sum',
                'fill_constant',
                'cast',
                'sum',
                'fill_constant',
                'cast',
                'sum',
                'fill_constant',
                'cast',
                'sum',
                'c_sync_comm_stream',
                'check_finite_and_unscale',
                'cast',
                'c_allreduce_max',
                'c_allreduce_max',
                'cast',
                'update_loss_scaling',
                'fill_constant',
                'c_allreduce_sum',
                'c_allreduce_sum',
                'sqrt',
                'fill_constant',
                'elementwise_max',
                'elementwise_div',
                'elementwise_mul',
                'elementwise_mul',
                'elementwise_mul',
                'elementwise_mul',
                'elementwise_mul',
                'elementwise_mul',
                'elementwise_mul',
                'elementwise_mul',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
            ],
        )

        # pp + mp, partial send recv
        self.assertIn('partial_recv', main_prog_op_types)
        self.assertIn('partial_allgather', main_prog_op_types)
        self.assertIn('partial_send', main_prog_op_types)

        # amp check_finite_and_unscale, allreduce(mp)->allreduce(pp)
        self.assertEqual(main_prog_op_types.count('c_allreduce_max'), 2)

        # global gradient clip, allreduce(mp)->allreduce(pp)
        self.assertEqual(main_prog_op_types.count('c_allreduce_sum'), 2)

        # should has ring id for pp
        created_ring_ids = [
            op.desc.attr("ring_id")
            for op in startup_prog_ops
            if op.type == "c_comm_init"
        ]
        self.assertIn(self.mp_ring_id, created_ring_ids)
        self.assertIn(self.pp_pair_ring_id, created_ring_ids)

        # check correctness of pp group
        sharding_group_waiting_port = None
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_0"
            ):
                mp_group_waiting_ports = op.desc.attr("other_endpoints")

        self.assertEqual(mp_group_waiting_ports, ['127.0.0.1:36003'])

        # check correctness of sharding group
        sharding_group_waiting_port = None
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_1"
            ):
                pp_group_waiting_ports = op.desc.attr("other_endpoints")

        self.assertEqual(pp_group_waiting_ports, ['127.0.0.1:36002'])

    def test_hybrid_with_mp_pp_amp_gclip_for_optimizer(self):
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, strategy = self.pp_net(train_prog, startup_prog)
        self.set_strategy(strategy, 'amp')
        strategy.sharding = True
        strategy.sharding_configs = {
            "sharding_degree": 1,
            "mp_degree": 2,
            "pp_degree": 2,
            "dp_degree": 1,
        }
        strategy.pipeline = True
        strategy.pipeline_configs = {
            "schedule_mode": "1F1B",
            "micro_batch_size": 2,
            "accumulate_steps": 4,
        }
        clip = paddle.fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0)
        self.optimizer(
            avg_cost,
            strategy,
            train_prog,
            startup_prog,
            grad_clip=clip,
            name="adamw",
        )
        train_prog = train_prog._pipeline_opt['section_program']
        startup_prog = startup_prog._pipeline_opt['startup_program']

        startup_prog_ops = startup_prog.global_block().ops
        main_prog_ops = train_prog.global_block().ops

        # check program
        startup_prog_op_types = [op.type for op in startup_prog_ops]
        main_prog_op_types = [op.type for op in main_prog_ops]

        # ring: mp, pp_group, pp_pair, pp_pair
        self.assertEqual(
            startup_prog_op_types,
            [
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
            ],
        )

        self.assertEqual(
            main_prog_op_types,
            [
                'partial_recv',
                'partial_allgather',
                'cast',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'cast',
                'tanh',
                'cast',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'cast',
                'tanh',
                'cast',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'cast',
                'tanh',
                'cast',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'softmax',
                'cast',
                'softmax_with_cross_entropy',
                'reduce_mean',
                'elementwise_mul',
                'fill_constant',
                'elementwise_mul_grad',
                'reduce_mean_grad',
                'softmax_with_cross_entropy_grad',
                'cast',
                'softmax_grad',
                'elementwise_add_grad',
                'mul_grad',
                'cast',
                'tanh_grad',
                'cast',
                'elementwise_add_grad',
                'mul_grad',
                'cast',
                'tanh_grad',
                'cast',
                'elementwise_add_grad',
                'mul_grad',
                'cast',
                'tanh_grad',
                'cast',
                'elementwise_add_grad',
                'mul_grad',
                'cast',
                'c_sync_calc_stream',
                'partial_send',
                'fill_constant',
                'cast',
                'sum',
                'fill_constant',
                'cast',
                'sum',
                'fill_constant',
                'cast',
                'sum',
                'fill_constant',
                'cast',
                'sum',
                'fill_constant',
                'cast',
                'sum',
                'fill_constant',
                'cast',
                'sum',
                'fill_constant',
                'cast',
                'sum',
                'fill_constant',
                'cast',
                'sum',
                'c_sync_comm_stream',
                'check_finite_and_unscale',
                'cast',
                'c_allreduce_max',
                'c_allreduce_max',
                'cast',
                'update_loss_scaling',
                'memcpy',
                'fill_constant',
                'c_allreduce_sum',
                'c_allreduce_sum',
                'sqrt',
                'fill_constant',
                'elementwise_max',
                'elementwise_div',
                'elementwise_mul',
                'elementwise_mul',
                'elementwise_mul',
                'elementwise_mul',
                'elementwise_mul',
                'elementwise_mul',
                'elementwise_mul',
                'elementwise_mul',
                'adamw',
                'adamw',
                'adamw',
                'adamw',
                'adamw',
                'adamw',
                'adamw',
                'adamw',
            ],
        )

        # pp + mp, partial send recv
        self.assertIn('partial_recv', main_prog_op_types)
        self.assertIn('partial_allgather', main_prog_op_types)
        self.assertIn('partial_send', main_prog_op_types)

        # amp check_finite_and_unscale, allreduce(mp)->allreduce(pp)
        self.assertEqual(main_prog_op_types.count('c_allreduce_max'), 2)

        # global gradient clip, allreduce(mp)->allreduce(pp)
        self.assertEqual(main_prog_op_types.count('c_allreduce_sum'), 2)

        # should has ring id for pp
        created_ring_ids = [
            op.desc.attr("ring_id")
            for op in startup_prog_ops
            if op.type == "c_comm_init"
        ]
        self.assertIn(self.mp_ring_id, created_ring_ids)
        self.assertIn(self.pp_pair_ring_id, created_ring_ids)

        # check correctness of pp group
        sharding_group_waiting_port = None
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_0"
            ):
                mp_group_waiting_ports = op.desc.attr("other_endpoints")

        self.assertEqual(mp_group_waiting_ports, ['127.0.0.1:36003'])

        # check correctness of sharding group
        sharding_group_waiting_port = None
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_1"
            ):
                pp_group_waiting_ports = op.desc.attr("other_endpoints")

        self.assertEqual(pp_group_waiting_ports, ['127.0.0.1:36002'])

    def test_hybrid_with_pp_dp_amp_fp16allreduce(self):
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, strategy = self.pp_net(train_prog, startup_prog)
        strategy.amp = True
        strategy.amp_configs = {
            'custom_black_varnames': ['fc_6.b_0'],
        }
        strategy.sharding = True
        strategy.sharding_configs = {
            "sharding_degree": 1,
            "mp_degree": 1,
            "pp_degree": 2,
            "dp_degree": 2,
        }
        strategy.pipeline = True
        strategy.pipeline_configs = {
            "schedule_mode": "1F1B",
            "micro_batch_size": 2,
            "accumulate_steps": 4,
        }
        strategy.fp16_allreduce = True
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        train_prog = train_prog._pipeline_opt['section_program']
        startup_prog = startup_prog._pipeline_opt['startup_program']

        startup_prog_ops = startup_prog.global_block().ops
        main_prog_ops = train_prog.global_block().ops

        # check program
        startup_prog_op_types = [op.type for op in startup_prog_ops]
        main_prog_op_types = [op.type for op in main_prog_ops]

        # ring: mp, pp_group, pp_pair, pp_pair
        self.assertEqual(
            startup_prog_op_types,
            [
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
            ],
        )

        self.assertEqual(
            main_prog_op_types,
            [
                'recv_v2',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'tanh',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'tanh',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'tanh',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'softmax',
                'softmax_with_cross_entropy',
                'reduce_mean',
                'elementwise_mul',
                'fill_constant',
                'elementwise_mul_grad',
                'reduce_mean_grad',
                'softmax_with_cross_entropy_grad',
                'softmax_grad',
                'elementwise_add_grad',
                'cast',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'c_sync_calc_stream',
                'send_v2',
                'fill_constant',
                'cast',
                'sum',
                'fill_constant',
                'sum',
                'fill_constant',
                'sum',
                'fill_constant',
                'sum',
                'fill_constant',
                'sum',
                'fill_constant',
                'sum',
                'fill_constant',
                'sum',
                'fill_constant',
                'sum',
                'coalesce_tensor',
                'c_allreduce_sum',
                'cast',
                'cast',
                'cast',
                'cast',
                'cast',
                'cast',
                'cast',
                'cast',
                'c_sync_comm_stream',
                'check_finite_and_unscale',
                'cast',
                'c_allreduce_max',
                'cast',
                'update_loss_scaling',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
            ],
        )

        # amp check_finite_and_unscale, allreduce(pp)
        self.assertEqual(main_prog_op_types.count('c_allreduce_max'), 1)

        # should has ring id for pp
        created_ring_ids = [
            op.desc.attr("ring_id")
            for op in startup_prog_ops
            if op.type == "c_comm_init"
        ]
        self.assertIn(self.pp_pair_ring_id, created_ring_ids)
        self.assertIn(self.dp_ring_id, created_ring_ids)

        # check correctness of pp group
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_0"
            ):
                pp_group_waiting_ports = op.desc.attr("other_endpoints")

        self.assertEqual(pp_group_waiting_ports, ['127.0.0.1:36003'])

        # check correctness of dp group
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_3"
            ):
                dp_group_waiting_ports = op.desc.attr("other_endpoints")

        self.assertEqual(dp_group_waiting_ports, ['127.0.0.1:36002'])

    def test_hybrid_with_sharding_pp_amp_fp16allreduce_in_optimize(self):
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, strategy = self.pp_net(train_prog, startup_prog)
        strategy.amp = True
        strategy.amp_configs = {
            'custom_black_varnames': ['fc_6.b_0'],
        }
        strategy.sharding = True
        strategy.sharding_configs = {
            "segment_broadcast_MB": 0.1,
            "sharding_degree": 2,
            "mp_degree": 1,
            "pp_degree": 2,
            "dp_degree": 1,
            'pp_allreduce_in_optimize': True,
        }
        strategy.pipeline = True
        strategy.pipeline_configs = {
            "schedule_mode": "1F1B",
            "micro_batch_size": 2,
            "accumulate_steps": 4,
        }
        strategy.fp16_allreduce = True
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        train_prog = train_prog._pipeline_opt['section_program']
        startup_prog = startup_prog._pipeline_opt['startup_program']

        startup_prog_ops = startup_prog.global_block().ops
        main_prog_ops = train_prog.global_block().ops

        # check program
        startup_prog_op_types = [op.type for op in startup_prog_ops]
        main_prog_op_types = [op.type for op in main_prog_ops]

        # ring: sharding, pp_group, pp_pair, pp_pair
        self.assertEqual(
            startup_prog_op_types,
            [
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
            ],
        )

        # FIXME(wangxi): some bug in sharding+pp with pp_allreduce_in_optimize
        # self.assertEqual(main_prog_op_types, [])

        # amp check_finite_and_unscale, allreduce(pp)
        self.assertEqual(main_prog_op_types.count('c_allreduce_max'), 2)

        # should has ring id for pp
        created_ring_ids = [
            op.desc.attr("ring_id")
            for op in startup_prog_ops
            if op.type == "c_comm_init"
        ]
        self.assertIn(self.sharding_ring_id, created_ring_ids)
        self.assertIn(self.pp_pair_ring_id, created_ring_ids)

        # check correctness of sharding group
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_0"
            ):
                sharding_group_waiting_ports = op.desc.attr("other_endpoints")

        self.assertEqual(sharding_group_waiting_ports, ['127.0.0.1:36003'])

        # check correctness of pp group
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_1"
            ):
                pp_group_waiting_ports = op.desc.attr("other_endpoints")

        self.assertEqual(pp_group_waiting_ports, ['127.0.0.1:36002'])

    def test_hybrid_with_pp_dp_amp_fp16allreduce_optimize_cast(self):
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, strategy = self.pp_net(train_prog, startup_prog)
        strategy.amp = True
        strategy.amp_configs = {
            'custom_black_varnames': ['fc_6.b_0'],
        }
        strategy.sharding = True
        strategy.sharding_configs = {
            "sharding_degree": 1,
            "mp_degree": 1,
            "pp_degree": 2,
            "dp_degree": 2,
            "optimize_cast": True,
        }
        strategy.pipeline = True
        strategy.pipeline_configs = {
            "schedule_mode": "1F1B",
            "micro_batch_size": 2,
            "accumulate_steps": 4,
        }
        strategy.fp16_allreduce = True
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        train_prog = train_prog._pipeline_opt['section_program']
        startup_prog = startup_prog._pipeline_opt['startup_program']

        startup_prog_ops = startup_prog.global_block().ops
        main_prog_ops = train_prog.global_block().ops

        # check program
        startup_prog_op_types = [op.type for op in startup_prog_ops]
        main_prog_op_types = [op.type for op in main_prog_ops]

        # ring: mp, pp_group, pp_pair, pp_pair
        self.assertEqual(
            startup_prog_op_types,
            [
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_broadcast',
                'cast',
                'c_broadcast',
                'cast',
                'c_broadcast',
                'cast',
                'c_broadcast',
                'cast',
                'c_broadcast',
                'cast',
                'c_broadcast',
                'cast',
                'c_broadcast',
                'cast',
                'c_broadcast',
            ],
        )

        self.assertEqual(
            main_prog_op_types,
            [
                'recv_v2',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'cast',
                'elementwise_add',
                'softmax',
                'softmax_with_cross_entropy',
                'reduce_mean',
                'elementwise_mul',
                'fill_constant',
                'elementwise_mul_grad',
                'reduce_mean_grad',
                'softmax_with_cross_entropy_grad',
                'softmax_grad',
                'elementwise_add_grad',
                'cast',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'c_sync_calc_stream',
                'send_v2',
                'fill_constant',
                'cast',
                'sum',
                'fill_constant',
                'sum',
                'fill_constant',
                'sum',
                'fill_constant',
                'sum',
                'fill_constant',
                'sum',
                'fill_constant',
                'sum',
                'fill_constant',
                'sum',
                'fill_constant',
                'sum',
                'coalesce_tensor',
                'c_allreduce_sum',
                'cast',
                'cast',
                'cast',
                'cast',
                'cast',
                'cast',
                'cast',
                'cast',
                'c_sync_comm_stream',
                'check_finite_and_unscale',
                'cast',
                'c_allreduce_max',
                'cast',
                'update_loss_scaling',
                'momentum',
                'cast',
                'momentum',
                'cast',
                'momentum',
                'cast',
                'momentum',
                'cast',
                'momentum',
                'cast',
                'momentum',
                'cast',
                'momentum',
                'momentum',
                'cast',
            ],
        )

        # amp check_finite_and_unscale, allreduce(pp)
        self.assertEqual(main_prog_op_types.count('c_allreduce_max'), 1)

        # should has ring id for pp
        created_ring_ids = [
            op.desc.attr("ring_id")
            for op in startup_prog_ops
            if op.type == "c_comm_init"
        ]
        self.assertIn(self.pp_pair_ring_id, created_ring_ids)
        self.assertIn(self.dp_ring_id, created_ring_ids)

        # check correctness of pp group
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_0"
            ):
                pp_group_waiting_ports = op.desc.attr("other_endpoints")

        self.assertEqual(pp_group_waiting_ports, ['127.0.0.1:36003'])

        # check correctness of dp group
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_3"
            ):
                dp_group_waiting_ports = op.desc.attr("other_endpoints")

        self.assertEqual(dp_group_waiting_ports, ['127.0.0.1:36002'])

    def test_hybrid_with_pp_dp_amp_fp16allreduce_optimize_offload(self):
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, strategy = self.pp_net(train_prog, startup_prog)
        strategy.amp = True
        strategy.amp_configs = {
            'custom_black_varnames': ['fc_6.b_0'],
        }
        strategy.sharding = True
        strategy.sharding_configs = {
            "sharding_degree": 1,
            "mp_degree": 1,
            "pp_degree": 2,
            "dp_degree": 2,
            "optimize_offload": True,
        }
        strategy.pipeline = True
        strategy.pipeline_configs = {
            "schedule_mode": "1F1B",
            "micro_batch_size": 2,
            "accumulate_steps": 4,
        }
        strategy.fp16_allreduce = True
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        train_prog = train_prog._pipeline_opt['section_program']
        startup_prog = startup_prog._pipeline_opt['startup_program']

        startup_prog_ops = startup_prog.global_block().ops
        main_prog_ops = train_prog.global_block().ops

        # check program
        startup_prog_op_types = [op.type for op in startup_prog_ops]
        main_prog_op_types = [op.type for op in main_prog_ops]

        # ring: mp, pp_group, pp_pair, pp_pair
        self.assertEqual(
            startup_prog_op_types,
            [
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_broadcast',
                'cast',
                'memcpy',
                'c_broadcast',
                'cast',
                'memcpy',
                'c_broadcast',
                'cast',
                'memcpy',
                'c_broadcast',
                'cast',
                'memcpy',
                'c_broadcast',
                'cast',
                'memcpy',
                'c_broadcast',
                'cast',
                'memcpy',
                'c_broadcast',
                'cast',
                'memcpy',
                'c_broadcast',
            ],
        )

        self.assertEqual(
            main_prog_op_types,
            [
                'recv_v2',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'cast',
                'elementwise_add',
                'softmax',
                'softmax_with_cross_entropy',
                'reduce_mean',
                'elementwise_mul',
                'fill_constant',
                'elementwise_mul_grad',
                'reduce_mean_grad',
                'softmax_with_cross_entropy_grad',
                'softmax_grad',
                'elementwise_add_grad',
                'cast',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'c_sync_calc_stream',
                'send_v2',
                'fill_constant',
                'cast',
                'sum',
                'fill_constant',
                'sum',
                'fill_constant',
                'sum',
                'fill_constant',
                'sum',
                'fill_constant',
                'sum',
                'fill_constant',
                'sum',
                'fill_constant',
                'sum',
                'fill_constant',
                'sum',
                'coalesce_tensor',
                'c_allreduce_sum',
                'cast',
                'cast',
                'cast',
                'cast',
                'cast',
                'cast',
                'cast',
                'cast',
                'c_sync_comm_stream',
                'check_finite_and_unscale',
                'cast',
                'c_allreduce_max',
                'cast',
                'update_loss_scaling',
                'memcpy',
                'momentum',
                'cast',
                'memcpy',
                'memcpy',
                'momentum',
                'cast',
                'memcpy',
                'memcpy',
                'momentum',
                'cast',
                'memcpy',
                'memcpy',
                'momentum',
                'cast',
                'memcpy',
                'memcpy',
                'momentum',
                'cast',
                'memcpy',
                'memcpy',
                'momentum',
                'cast',
                'memcpy',
                'momentum',
                'memcpy',
                'momentum',
                'cast',
                'memcpy',
            ],
        )

        # amp check_finite_and_unscale, allreduce(pp)
        self.assertEqual(main_prog_op_types.count('c_allreduce_max'), 1)

        # should has ring id for pp
        created_ring_ids = [
            op.desc.attr("ring_id")
            for op in startup_prog_ops
            if op.type == "c_comm_init"
        ]
        self.assertIn(self.pp_pair_ring_id, created_ring_ids)
        self.assertIn(self.dp_ring_id, created_ring_ids)

        # check correctness of pp group
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_0"
            ):
                pp_group_waiting_ports = op.desc.attr("other_endpoints")

        self.assertEqual(pp_group_waiting_ports, ['127.0.0.1:36003'])

        # check correctness of dp group
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_3"
            ):
                dp_group_waiting_ports = op.desc.attr("other_endpoints")

        self.assertEqual(dp_group_waiting_ports, ['127.0.0.1:36002'])

    def test_hybrid_with_pp_dp_amp_fp16allreduce_optimize_cast_with_gradient_fuse(
        self,
    ):
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, strategy = self.pp_net(train_prog, startup_prog)
        strategy.amp = True
        strategy.amp_configs = {
            'custom_black_varnames': ['fc_6.b_0'],
        }
        strategy.sharding = True
        strategy.sharding_configs = {
            "sharding_degree": 1,
            "mp_degree": 1,
            "pp_degree": 2,
            "dp_degree": 2,
            "optimize_cast": True,
        }
        strategy.pipeline = True
        strategy.pipeline_configs = {
            "schedule_mode": "1F1B",
            "micro_batch_size": 2,
            "accumulate_steps": 4,
        }
        strategy.fp16_allreduce = True
        strategy.fuse_grad_merge = True
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        train_prog = train_prog._pipeline_opt['section_program']
        startup_prog = startup_prog._pipeline_opt['startup_program']

        startup_prog_ops = startup_prog.global_block().ops
        main_prog_ops = train_prog.global_block().ops

        # check program
        startup_prog_op_types = [op.type for op in startup_prog_ops]
        main_prog_op_types = [op.type for op in main_prog_ops]

        # ring: mp, pp_group, pp_pair, pp_pair
        self.assertEqual(
            startup_prog_op_types,
            [
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_broadcast',
                'cast',
                'c_broadcast',
                'cast',
                'c_broadcast',
                'cast',
                'c_broadcast',
                'cast',
                'c_broadcast',
                'cast',
                'c_broadcast',
                'cast',
                'c_broadcast',
                'cast',
                'c_broadcast',
            ],
        )

        self.assertEqual(
            main_prog_op_types,
            [
                'recv_v2',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'cast',
                'elementwise_add',
                'softmax',
                'softmax_with_cross_entropy',
                'reduce_mean',
                'elementwise_mul',
                'coalesce_tensor',
                'coalesce_tensor',
                'coalesce_tensor',
                'coalesce_tensor',
                'fill_constant',
                'elementwise_mul_grad',
                'reduce_mean_grad',
                'softmax_with_cross_entropy_grad',
                'softmax_grad',
                'elementwise_add_grad',
                'cast',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'c_sync_calc_stream',
                'send_v2',
                'sum',
                'cast',
                'sum',
                'c_allreduce_sum',
                'c_allreduce_sum',
                'cast',
                'cast',
                'cast',
                'cast',
                'cast',
                'cast',
                'cast',
                'cast',
                'c_sync_comm_stream',
                'check_finite_and_unscale',
                'cast',
                'c_allreduce_max',
                'cast',
                'update_loss_scaling',
                'momentum',
                'cast',
                'momentum',
                'cast',
                'momentum',
                'cast',
                'momentum',
                'cast',
                'momentum',
                'cast',
                'momentum',
                'cast',
                'momentum',
                'momentum',
                'cast',
            ],
        )

        # amp check_finite_and_unscale, allreduce(pp)
        self.assertEqual(main_prog_op_types.count('c_allreduce_max'), 1)

        # should has ring id for pp
        created_ring_ids = [
            op.desc.attr("ring_id")
            for op in startup_prog_ops
            if op.type == "c_comm_init"
        ]
        self.assertIn(self.pp_pair_ring_id, created_ring_ids)
        self.assertIn(self.dp_ring_id, created_ring_ids)

        # check correctness of pp group
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_0"
            ):
                pp_group_waiting_ports = op.desc.attr("other_endpoints")

        self.assertEqual(pp_group_waiting_ports, ['127.0.0.1:36003'])

        # check correctness of dp group
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_3"
            ):
                dp_group_waiting_ports = op.desc.attr("other_endpoints")

        self.assertEqual(dp_group_waiting_ports, ['127.0.0.1:36002'])

    def test_hybrid_with_pp_dp_amp_with_gradient_fuse(self):
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, strategy = self.pp_net(train_prog, startup_prog)
        strategy.amp = True
        strategy.amp_configs = {
            'custom_black_varnames': ['fc_6.b_0'],
        }
        strategy.sharding = True
        strategy.sharding_configs = {
            "sharding_degree": 1,
            "mp_degree": 1,
            "pp_degree": 2,
            "dp_degree": 2,
        }
        strategy.pipeline = True
        strategy.pipeline_configs = {
            "schedule_mode": "1F1B",
            "micro_batch_size": 2,
            "accumulate_steps": 4,
        }
        strategy.fuse_grad_merge = True
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        train_prog = train_prog._pipeline_opt['section_program']
        startup_prog = startup_prog._pipeline_opt['startup_program']

        startup_prog_ops = startup_prog.global_block().ops
        main_prog_ops = train_prog.global_block().ops

        # check program
        startup_prog_op_types = [op.type for op in startup_prog_ops]
        main_prog_op_types = [op.type for op in main_prog_ops]

        # ring: mp, pp_group, pp_pair, pp_pair
        self.assertEqual(
            startup_prog_op_types,
            [
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
            ],
        )

        self.assertEqual(
            main_prog_op_types,
            [
                'recv_v2',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'tanh',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'tanh',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'tanh',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'softmax',
                'softmax_with_cross_entropy',
                'reduce_mean',
                'elementwise_mul',
                'coalesce_tensor',
                'coalesce_tensor',
                'coalesce_tensor',
                'coalesce_tensor',
                'fill_constant',
                'elementwise_mul_grad',
                'reduce_mean_grad',
                'softmax_with_cross_entropy_grad',
                'softmax_grad',
                'elementwise_add_grad',
                'cast',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'c_sync_calc_stream',
                'send_v2',
                'cast',
                'sum',
                'sum',
                'c_allreduce_sum',
                'c_allreduce_sum',
                'c_sync_comm_stream',
                'check_finite_and_unscale',
                'cast',
                'c_allreduce_max',
                'cast',
                'update_loss_scaling',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
            ],
        )

        # amp check_finite_and_unscale, allreduce(pp)
        self.assertEqual(main_prog_op_types.count('c_allreduce_max'), 1)

        # should has ring id for pp
        created_ring_ids = [
            op.desc.attr("ring_id")
            for op in startup_prog_ops
            if op.type == "c_comm_init"
        ]
        self.assertIn(self.pp_pair_ring_id, created_ring_ids)
        self.assertIn(self.dp_ring_id, created_ring_ids)

        # check correctness of pp group
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_0"
            ):
                pp_group_waiting_ports = op.desc.attr("other_endpoints")

        self.assertEqual(pp_group_waiting_ports, ['127.0.0.1:36003'])

        # check correctness of dp group
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_3"
            ):
                dp_group_waiting_ports = op.desc.attr("other_endpoints")

        self.assertEqual(dp_group_waiting_ports, ['127.0.0.1:36002'])

    def test_hybrid_with_pp_dp_amp_with_gradient_fuse_and_avg_after_sum(self):
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, strategy = self.pp_net(train_prog, startup_prog)
        strategy.amp = True
        strategy.amp_configs = {
            'custom_black_varnames': ['fc_6.b_0'],
        }
        strategy.sharding = True
        strategy.sharding_configs = {
            "sharding_degree": 1,
            "mp_degree": 1,
            "pp_degree": 2,
            "dp_degree": 2,
        }
        strategy.pipeline = True
        strategy.pipeline_configs = {
            "schedule_mode": "1F1B",
            "micro_batch_size": 2,
            "accumulate_steps": 4,
        }
        strategy.gradient_scale_configs = {
            'scale_strategy': 'avg',
            'scale_gradient': True,
        }
        strategy.fuse_grad_merge = True
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        train_prog = train_prog._pipeline_opt['section_program']
        startup_prog = startup_prog._pipeline_opt['startup_program']

        startup_prog_ops = startup_prog.global_block().ops
        main_prog_ops = train_prog.global_block().ops

        # check program
        startup_prog_op_types = [op.type for op in startup_prog_ops]
        main_prog_op_types = [op.type for op in main_prog_ops]

        self.assertEqual(
            startup_prog_op_types,
            [
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
            ],
        )

        self.assertEqual(
            main_prog_op_types,
            [
                'recv_v2',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'tanh',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'tanh',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'tanh',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'softmax',
                'softmax_with_cross_entropy',
                'reduce_mean',
                'elementwise_mul',
                'coalesce_tensor',
                'coalesce_tensor',
                'coalesce_tensor',
                'coalesce_tensor',
                'fill_constant',
                'elementwise_mul_grad',
                'reduce_mean_grad',
                'softmax_with_cross_entropy_grad',
                'softmax_grad',
                'elementwise_add_grad',
                'cast',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'c_sync_calc_stream',
                'send_v2',
                'cast',
                'sum',
                'sum',
                'c_allreduce_sum',
                'c_allreduce_sum',
                'c_sync_comm_stream',
                'scale',
                'check_finite_and_unscale',
                'cast',
                'c_allreduce_max',
                'cast',
                'update_loss_scaling',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
            ],
        )

    def test_hybrid_with_pp_dp_with_gradient_fuse_and_avg_after_sum(self):
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, strategy = self.pp_net(train_prog, startup_prog)
        strategy.sharding = True
        strategy.sharding_configs = {
            "sharding_degree": 1,
            "mp_degree": 1,
            "pp_degree": 2,
            "dp_degree": 2,
        }
        strategy.pipeline = True
        strategy.pipeline_configs = {
            "schedule_mode": "1F1B",
            "micro_batch_size": 2,
            "accumulate_steps": 4,
        }
        strategy.gradient_scale_configs = {
            'scale_strategy': 'avg',
            'scale_gradient': True,
        }
        strategy.fuse_grad_merge = True
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        train_prog = train_prog._pipeline_opt['section_program']
        startup_prog = startup_prog._pipeline_opt['startup_program']

        startup_prog_ops = startup_prog.global_block().ops
        main_prog_ops = train_prog.global_block().ops

        # check program
        startup_prog_op_types = [op.type for op in startup_prog_ops]
        main_prog_op_types = [op.type for op in main_prog_ops]

        self.assertEqual(
            startup_prog_op_types,
            [
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
            ],
        )

        self.assertEqual(
            main_prog_op_types,
            [
                'recv_v2',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'tanh',
                'mul',
                'elementwise_add',
                'softmax',
                'softmax_with_cross_entropy',
                'reduce_mean',
                'coalesce_tensor',
                'coalesce_tensor',
                'fill_constant',
                'reduce_mean_grad',
                'softmax_with_cross_entropy_grad',
                'softmax_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'c_sync_calc_stream',
                'send_v2',
                'sum',
                'c_allreduce_sum',
                'c_sync_comm_stream',
                'scale',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
            ],
        )

    def test_hybrid_with_pp_dp_with_amp_no_dynamic_gradient_fuse_and_avg_after_sum(
        self,
    ):
        train_prog, startup_prog = (
            paddle.fluid.Program(),
            paddle.fluid.Program(),
        )
        avg_cost, strategy = self.pp_net(train_prog, startup_prog)
        strategy.sharding = True
        strategy.sharding_configs = {
            "sharding_degree": 1,
            "mp_degree": 1,
            "pp_degree": 2,
            "dp_degree": 2,
        }
        strategy.amp = True
        strategy.amp_configs = {
            'custom_black_varnames': ['fc_6.b_0'],
            'use_dynamic_loss_scaling': False,
        }
        strategy.pipeline = True
        strategy.pipeline_configs = {
            "schedule_mode": "1F1B",
            "micro_batch_size": 2,
            "accumulate_steps": 4,
        }
        strategy.gradient_scale_configs = {
            'scale_strategy': 'avg',
            'scale_gradient': True,
        }
        strategy.fuse_grad_merge = True
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        train_prog = train_prog._pipeline_opt['section_program']
        startup_prog = startup_prog._pipeline_opt['startup_program']

        startup_prog_ops = startup_prog.global_block().ops
        main_prog_ops = train_prog.global_block().ops

        # check program
        startup_prog_op_types = [op.type for op in startup_prog_ops]
        main_prog_op_types = [op.type for op in main_prog_ops]

        self.assertEqual(
            startup_prog_op_types,
            [
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'uniform_random',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'fill_constant',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_gen_nccl_id',
                'c_comm_init',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
                'c_broadcast',
            ],
        )

        self.assertEqual(
            main_prog_op_types,
            [
                'recv_v2',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'tanh',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'tanh',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'tanh',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'softmax',
                'softmax_with_cross_entropy',
                'reduce_mean',
                'elementwise_mul',
                'coalesce_tensor',
                'coalesce_tensor',
                'coalesce_tensor',
                'coalesce_tensor',
                'fill_constant',
                'elementwise_mul_grad',
                'reduce_mean_grad',
                'softmax_with_cross_entropy_grad',
                'softmax_grad',
                'elementwise_add_grad',
                'cast',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'tanh_grad',
                'elementwise_add_grad',
                'mul_grad',
                'c_sync_calc_stream',
                'send_v2',
                'cast',
                'sum',
                'sum',
                'c_allreduce_sum',
                'c_allreduce_sum',
                'c_sync_comm_stream',
                'scale',
                'scale',
                'check_finite_and_unscale',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
            ],
        )


if __name__ == "__main__":
    unittest.main()
