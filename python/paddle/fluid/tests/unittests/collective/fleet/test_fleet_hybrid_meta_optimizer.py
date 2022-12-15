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

import os
import unittest

from fleet_meta_optimizer_base import TestFleetMetaOptimizer

import paddle
import paddle.static as static
from paddle.distributed.fleet.meta_optimizers.common import is_loss_grad_op

paddle.enable_static()


class TestFleetHybridOptimizer(TestFleetMetaOptimizer):
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
        self._debug = False

    def test_opt_sharding_with_pp(self):
        train_prog, startup_prog = static.Program(), static.Program()
        avg_cost, strategy = self.pp_net(train_prog, startup_prog)

        self.set_strategy(strategy, 'pipeline')
        strategy.sharding = True
        strategy.sharding_configs = {
            "sharding_degree": 1,
            "pp_degree": 2,
            "dp_degree": 2,
            "_dp_as_optimizer_sharding": True,
        }
        strategy.fuse_all_reduce_ops = False

        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        train_prog = train_prog._pipeline_opt['section_program']
        startup_prog = startup_prog._pipeline_opt['startup_program']

        self.debug_program(train_prog, startup_prog)

        startup_prog_ops = startup_prog.global_block().ops
        main_prog_ops = train_prog.global_block().ops

        # check program
        startup_prog_op_types = [op.type for op in startup_prog_ops]
        main_prog_op_types = [op.type for op in main_prog_ops]

        # global, sharding, pp_send, pp_recv
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
                'fill_constant',
                'sum',
                'c_reduce_sum',
                'c_reduce_sum',
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
                'momentum',
                'momentum',
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

        # should has ring id for pp
        created_ring_ids = [
            op.desc.attr("ring_id")
            for op in startup_prog_ops
            if op.type == "c_comm_init"
        ]
        self.assertIn(self.dp_ring_id, created_ring_ids)
        self.assertIn(self.pp_pair_ring_id, created_ring_ids)

        # check correctness of pp group
        pp_group_waiting_prots = None
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_0"
            ):
                pp_group_waiting_prots = op.desc.attr("other_endpoints")
        self.assertEqual(pp_group_waiting_prots, ['127.0.0.1:36003'])

        # check correctness of sharding group
        dp_group_waiting_ports = None
        for op in startup_prog_ops:
            if (
                op.type == "c_gen_nccl_id"
                and op.desc.output_arg_names()[0] == "comm_id_3"
            ):
                dp_group_waiting_ports = op.desc.attr("other_endpoints")
        self.assertEqual(dp_group_waiting_ports, ['127.0.0.1:36002'])

    def test_opt_sharding_with_pp_with_allreduce_fuse(self):
        train_prog, startup_prog = static.Program(), static.Program()
        avg_cost, strategy = self.pp_net(train_prog, startup_prog)

        self.set_strategy(strategy, 'pipeline')
        strategy.sharding = True
        strategy.sharding_configs = {
            "sharding_degree": 1,
            "pp_degree": 2,
            "dp_degree": 2,
            "_dp_as_optimizer_sharding": True,
        }
        strategy.fuse_all_reduce_ops = True
        strategy.fuse_grad_size_in_MB = 32

        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        train_prog = train_prog._pipeline_opt['section_program']
        startup_prog = startup_prog._pipeline_opt['startup_program']

        startup_prog_ops = startup_prog.global_block().ops
        main_prog_ops = train_prog.global_block().ops

        # check program
        startup_prog_op_types = [op.type for op in startup_prog_ops]
        main_prog_op_types = [op.type for op in main_prog_ops]

        # global, sharding, pp_send, pp_recv
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
                'fill_constant',
                'sum',
                'coalesce_tensor',
                'c_reduce_sum',
                'coalesce_tensor',
                'c_reduce_sum',
                'c_sync_comm_stream',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'coalesce_tensor',
                'c_broadcast',
                'coalesce_tensor',
                'c_broadcast',
            ],
        )

    def test_opt_sharding_with_pp_amp_gclip(self):
        train_prog, startup_prog = static.Program(), static.Program()
        avg_cost, strategy = self.pp_net(train_prog, startup_prog)

        self.set_strategy(strategy, 'amp')
        self.set_strategy(strategy, 'pipeline')

        strategy.sharding = True
        strategy.sharding_configs = {
            "sharding_degree": 1,
            "pp_degree": 2,
            "dp_degree": 2,
            "_dp_as_optimizer_sharding": True,
        }
        strategy.fuse_all_reduce_ops = True
        strategy.fuse_grad_size_in_MB = 32
        clip = paddle.fluid.clip.GradientClipByGlobalNorm(1.0)

        self.optimizer(
            avg_cost, strategy, train_prog, startup_prog, grad_clip=clip
        )
        train_prog = train_prog._pipeline_opt['section_program']
        startup_prog = startup_prog._pipeline_opt['startup_program']
        self.debug_program(train_prog, startup_prog)

        startup_prog_ops = startup_prog.global_block().ops
        main_prog_ops = train_prog.global_block().ops

        # check program
        startup_prog_op_types = [op.type for op in startup_prog_ops]
        main_prog_op_types = [op.type for op in main_prog_ops]

        # global, sharding, pp_send, pp_recv
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
                'send_v2',
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
                'coalesce_tensor',
                'c_reduce_sum',
                'coalesce_tensor',
                'c_reduce_sum',
                'c_sync_comm_stream',
                'check_finite_and_unscale',
                'cast',
                'c_allreduce_max',
                'c_allreduce_max',
                'cast',
                'update_loss_scaling',
                'squared_l2_norm',
                'squared_l2_norm',
                'squared_l2_norm',
                'squared_l2_norm',
                'squared_l2_norm',
                'sum',
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
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'coalesce_tensor',
                'c_broadcast',
                'coalesce_tensor',
                'c_broadcast',
            ],
        )

    def test_opt_sharding_with_pp_amp_gclip_fuse_gm(self):
        train_prog, startup_prog = static.Program(), static.Program()
        avg_cost, strategy = self.pp_net(train_prog, startup_prog)

        self.set_strategy(strategy, 'amp')
        self.set_strategy(strategy, 'pipeline')

        strategy.sharding = True
        strategy.sharding_configs = {
            "sharding_degree": 1,
            "pp_degree": 2,
            "dp_degree": 2,
            "_dp_as_optimizer_sharding": True,
        }
        strategy.fuse_all_reduce_ops = True
        strategy.fuse_grad_size_in_MB = 32
        strategy.fuse_grad_merge = True
        clip = paddle.fluid.clip.GradientClipByGlobalNorm(1.0)

        self.optimizer(
            avg_cost, strategy, train_prog, startup_prog, grad_clip=clip
        )
        train_prog = train_prog._pipeline_opt['section_program']
        startup_prog = startup_prog._pipeline_opt['startup_program']
        self.debug_program(train_prog, startup_prog)

        startup_prog_ops = startup_prog.global_block().ops
        main_prog_ops = train_prog.global_block().ops

        # check program
        startup_prog_op_types = [op.type for op in startup_prog_ops]
        main_prog_op_types = [op.type for op in main_prog_ops]

        # global, sharding, pp_send, pp_recv
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
                'coalesce_tensor',
                'coalesce_tensor',
                'coalesce_tensor',
                'coalesce_tensor',
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
                'send_v2',
                'cast',
                'sum',
                'cast',
                'sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_sync_comm_stream',
                'check_finite_and_unscale',
                'cast',
                'c_allreduce_max',
                'c_allreduce_max',
                'cast',
                'update_loss_scaling',
                'squared_l2_norm',
                'squared_l2_norm',
                'squared_l2_norm',
                'squared_l2_norm',
                'squared_l2_norm',
                'sum',
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
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'momentum',
                'coalesce_tensor',
                'c_broadcast',
                'coalesce_tensor',
                'c_broadcast',
            ],
        )

    def test_opt_sharding_with_pp_amp_ckp_fuse_gm_optcast(self):
        train_prog, startup_prog = static.Program(), static.Program()
        avg_cost, strategy = self.pp_net(train_prog, startup_prog)

        self.set_strategy(strategy, 'pipeline')
        self.set_strategy(strategy, 'amp')
        strategy.amp_configs = {
            'custom_black_varnames': ['fc_6.b_0'],
        }
        strategy.recompute = True
        strategy.recompute_configs = {
            "checkpoints": [
                "fc_0.tmp_2",
                "fc_1.tmp_2",
                "fc_2.tmp_2",
                "fc_3.tmp_2",
            ]
        }

        strategy.sharding = True
        strategy.sharding_configs = {
            "sharding_degree": 1,
            "pp_degree": 2,
            "dp_degree": 2,
            "_dp_as_optimizer_sharding": True,
            'optimize_cast': True,
        }
        strategy.fuse_all_reduce_ops = True
        strategy.fuse_grad_size_in_MB = 32
        strategy.fuse_grad_merge = True

        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        train_prog = train_prog._pipeline_opt['section_program']
        startup_prog = startup_prog._pipeline_opt['startup_program']

        # self._debug = True
        self.debug_program(train_prog, startup_prog)

        startup_prog_ops = startup_prog.global_block().ops
        main_prog_ops = train_prog.global_block().ops

        # check program
        startup_prog_op_types = [op.type for op in startup_prog_ops]
        main_prog_op_types = [op.type for op in main_prog_ops]

        # global, sharding, pp_send, pp_recv
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
                'cast',
                'tanh',
                'cast',
                'mul',
                'cast',
                'elementwise_add',
                'cast',
                'softmax',
                'cast',
                'softmax_with_cross_entropy',
                'reduce_mean',
                'elementwise_mul',
                'coalesce_tensor',
                'coalesce_tensor',
                'coalesce_tensor',
                'coalesce_tensor',
                'coalesce_tensor',
                'coalesce_tensor',
                'fill_constant',
                'elementwise_mul_grad',
                'reduce_mean_grad',
                'softmax_with_cross_entropy_grad',
                'cast',
                'softmax_grad',
                'cast',
                'elementwise_add_grad',
                'cast',
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
                'cast',
                'mul',
                'elementwise_add',
                'cast',
                'tanh_grad',
                'cast',
                'elementwise_add_grad',
                'mul_grad',
                'cast',
                'c_sync_calc_stream',
                'send_v2',
                'cast',
                'sum',
                'sum',
                'cast',
                'sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_reduce_sum',
                'c_sync_comm_stream',
                'check_finite_and_unscale',
                'cast',
                'c_allreduce_max',
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
                'momentum',
                'cast',
                'coalesce_tensor',
                'c_broadcast',
                'c_broadcast',
                'coalesce_tensor',
                'c_broadcast',
            ],
        )


class TestFleetHybridOptimizerBoundary(TestFleetMetaOptimizer):
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
        self._debug = False

    def test_opt_sharding_with_pp_amp_gclip_boundary(self):
        """
        test optimizer sharding without parameter
        test loss grad scale value
        """
        train_prog, startup_prog = static.Program(), static.Program()
        avg_cost, strategy = self.boundary_net(train_prog, startup_prog)

        self.set_strategy(strategy, 'amp')
        self.set_strategy(strategy, 'pipeline')
        strategy.sharding = True
        strategy.sharding_configs = {
            "sharding_degree": 1,
            "pp_degree": 2,
            "dp_degree": 2,
            "_dp_as_optimizer_sharding": True,
        }
        strategy.fuse_all_reduce_ops = True
        strategy.fuse_grad_size_in_MB = 32
        clip = paddle.fluid.clip.GradientClipByGlobalNorm(1.0)

        self.optimizer(
            avg_cost, strategy, train_prog, startup_prog, grad_clip=clip
        )
        train_prog = train_prog._pipeline_opt['section_program']
        startup_prog = startup_prog._pipeline_opt['startup_program']
        self.debug_program(train_prog, startup_prog)

        startup_prog_ops = startup_prog.global_block().ops
        main_prog_ops = train_prog.global_block().ops

        # check program
        startup_prog_op_types = [op.type for op in startup_prog_ops]
        main_prog_op_types = [op.type for op in main_prog_ops]

        # check loss scale for hybrid
        for op in main_prog_ops:
            if is_loss_grad_op(op):
                self.assertEqual(op.type, 'fill_constant')
                self.assertTrue(op.has_attr('value'))
                scale = (
                    strategy.pipeline_configs['accumulate_steps']
                    * strategy.sharding_configs['dp_degree']
                )
                loss_scale = 1.0 / scale
                self.assertAlmostEqual(float(op.attr('value')), loss_scale)

        # global, sharding, pp_send, pp_recv
        self.assertEqual(
            startup_prog_op_types,
            [
                'uniform_random',
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
            ],
        )

        self.assertEqual(
            main_prog_op_types,
            [
                'recv_v2',
                'cast',
                'matmul_v2',
                'cast',
                'reduce_mean',
                'elementwise_mul',
                'fill_constant',
                'elementwise_mul_grad',
                'reduce_mean_grad',
                'cast',
                'matmul_v2_grad',
                'c_sync_calc_stream',
                'send_v2',
                'fill_constant',
                'cast',
                'sum',
                'c_reduce_sum',
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
                'c_broadcast',
            ],
        )

    def test_opt_sharding_with_pp_amp_gclip_boundary_card1(self):
        """test optimizer sharding without parameter in card0"""
        os.environ["PADDLE_TRAINER_ID"] = "1"
        train_prog, startup_prog = static.Program(), static.Program()
        avg_cost, strategy = self.boundary_net(train_prog, startup_prog)

        self.set_strategy(strategy, 'amp')
        self.set_strategy(strategy, 'pipeline')
        strategy.sharding = True
        strategy.sharding_configs = {
            "sharding_degree": 1,
            "pp_degree": 2,
            "dp_degree": 2,
            "_dp_as_optimizer_sharding": True,
        }
        strategy.fuse_all_reduce_ops = True
        strategy.fuse_grad_size_in_MB = 32
        clip = paddle.fluid.clip.GradientClipByGlobalNorm(1.0)

        self.optimizer(
            avg_cost, strategy, train_prog, startup_prog, grad_clip=clip
        )
        train_prog = train_prog._pipeline_opt['section_program']
        startup_prog = startup_prog._pipeline_opt['startup_program']
        self.debug_program(train_prog, startup_prog)

        startup_prog_ops = startup_prog.global_block().ops
        main_prog_ops = train_prog.global_block().ops

        # check program
        startup_prog_op_types = [op.type for op in startup_prog_ops]
        main_prog_op_types = [op.type for op in main_prog_ops]

        # global, sharding, pp_send, pp_recv
        self.assertEqual(
            startup_prog_op_types,
            [
                'uniform_random',
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
            ],
        )

        self.assertEqual(
            main_prog_op_types,
            [
                'recv_v2',
                'cast',
                'matmul_v2',
                'cast',
                'reduce_mean',
                'elementwise_mul',
                'fill_constant',
                'elementwise_mul_grad',
                'reduce_mean_grad',
                'cast',
                'matmul_v2_grad',
                'c_sync_calc_stream',
                'send_v2',
                'fill_constant',
                'cast',
                'sum',
                'c_reduce_sum',
                'c_sync_comm_stream',
                'check_finite_and_unscale',
                'cast',
                'c_allreduce_max',
                'c_allreduce_max',
                'cast',
                'update_loss_scaling',
                'squared_l2_norm',
                'sum',
                'c_allreduce_sum',
                'c_allreduce_sum',
                'sqrt',
                'fill_constant',
                'elementwise_max',
                'elementwise_div',
                'elementwise_mul',
                'momentum',
                'c_broadcast',
            ],
        )


if __name__ == "__main__":
    unittest.main()
