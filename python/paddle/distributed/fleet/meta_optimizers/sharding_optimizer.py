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

from paddle.fluid import unique_name, core
import paddle.fluid as fluid

from paddle.distributed.fleet.meta_optimizers.common import OpRole, OP_ROLE_VAR_KEY, CollectiveHelper
from paddle.distributed.fleet.meta_optimizers.common import is_backward_op, is_optimizer_op, is_update_op
from paddle.distributed.fleet.meta_optimizers.meta_optimizer_base import MetaOptimizerBase
from paddle.distributed.fleet.meta_optimizers.sharding.shard import Shard, ProgramSegment
from paddle.distributed.fleet.meta_optimizers.sharding.fp16_helper import FP16Utils
from paddle.distributed.fleet.meta_optimizers.sharding.weight_decay_helper import WeightDecayHelper
from paddle.distributed.fleet.meta_optimizers.sharding.gradient_clip_helper import GradientClipHelper
from .sharding.offload_helper import OffloadHelper
from paddle.distributed.fleet.meta_optimizers.sharding.prune import ProgramDeps
from paddle.distributed.fleet.meta_optimizers.sharding.utils import *

import logging
from functools import reduce

__all__ = ["ShardingOptimizer"]


class ShardingOptimizer(MetaOptimizerBase):
    """Sharding Optimizer."""

    def __init__(self, optimizer):
        super(ShardingOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        self.meta_optimizers_white_list = [
            "RecomputeOptimizer",
            "AMPOptimizer",
            "LarsOptimizer",
            "LambOptimizer",
            # "ModelParallelOptimizer",
            "PipelineOptimizer",
        ]
        self.meta_optimizers_black_list = ["GraphExecutionOptimizer", ]
        self._main_program = None
        self._startup_program = None
        self._segments = []
        # params and fp16 params is for broadcast
        self._params = set([])
        self._broadcast_vars = set([])
        # reduced grads to param name
        self._reduced_grads_to_param = {}
        self._shard = Shard()

        # use sharding as outer parallelism (e.g. inner:Megatron & outer sharding)
        self._as_outer_parallelism = False
        self._inner_parallelism_size = None

    def _can_apply(self):
        if not self.role_maker._is_collective:
            return False
        if self.role_maker._worker_num() <= 1:
            return False
        return self.user_defined_strategy.sharding

    def _disable_strategy(self, dist_strategy):
        dist_strategy.sharding = False
        dist_strategy.sharding_configs = {}

    def _enable_strategy(self, dist_strategy, context):
        dist_strategy.sharding = True
        dist_strategy.sharding_configs = {"fuse_broadcast_MB": 32}

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        """Implementation of minimize."""
        # TODO: (JZ-LIANG) support multiple comm in future
        # self._nrings = self.user_defined_strategy.nccl_comm_num
        self._nrings_sharding = 1
        self._nrings_dp = 1
        self._fuse_broadcast_MB = self.user_defined_strategy.sharding_configs[
            "fuse_broadcast_MB"]
        self.hybrid_dp = self.user_defined_strategy.sharding_configs[
            "hybrid_dp"]
        self._as_outer_parallelism = self.user_defined_strategy.sharding_configs[
            "as_outer_parallelism"]
        self._inner_parallelism_size = int(
            self.user_defined_strategy.sharding_configs["parallelism"])
        self.use_pipeline = self.user_defined_strategy.sharding_configs[
            "use_pipeline"]
        self.acc_steps = self.user_defined_strategy.sharding_configs[
            "acc_steps"]
        self.schedule_mode = self.user_defined_strategy.sharding_configs[
            "schedule_mode"]
        self.pp_bz = self.user_defined_strategy.sharding_configs["pp_bz"]
        self.pp_allreduce_in_optimize = self.user_defined_strategy.sharding_configs[
            "pp_allreduce_in_optimize"]
        self.optimize_offload = self.user_defined_strategy.sharding_configs[
            "optimize_offload"]

        if self.inner_opt is None:
            raise ValueError(
                "self.inner_opt of ShardingOptimizer should not be None.")
        if self.use_pipeline:
            pp_optimizer = fluid.optimizer.PipelineOptimizer(self.inner_opt,
                                                             self.acc_steps)
            main_program = loss.block.program
            main_program._pipeline_opt = dict()
            main_program._pipeline_opt['schedule_mode'] = self.schedule_mode
            main_program._pipeline_opt['pp_bz'] = self.pp_bz
            pp_rank = self.role_maker._worker_index() // (
                self.user_defined_strategy.sharding_configs[
                    'sharding_group_size'] * self._inner_parallelism_size)
            main_program._pipeline_opt['local_rank'] = pp_rank
            main_program._pipeline_opt[
                'global_rank'] = self.role_maker._worker_index()
            main_program._pipeline_opt['use_sharding'] = True
            main_program._pipeline_opt['ring_id'] = 20
            optimize_ops, params_grads, program_list, self.pipeline_pair, self.pp_ring_map = pp_optimizer.minimize(
                loss, startup_program, parameter_list, no_grad_set)
            self.pipeline_nodes = len(program_list)
        else:
            optimize_ops, params_grads = self.inner_opt.minimize(
                loss, startup_program, parameter_list, no_grad_set)

        if startup_program is None:
            startup_program = default_startup_program()
        if self.use_pipeline:
            startup_program = startup_program._pipeline_opt['startup_program']
            #main_program = main_program._pipeline_opt['section_program']['program']
            print("pp_rank:", pp_rank)
            main_program = program_list[pp_rank]['program']
            with open("main_%d" % self.role_maker._worker_index(), 'w') as f:
                f.writelines(str(main_program))
            main_block = main_program.global_block()
            new_params_grads = []
            for param, grad in params_grads:
                if main_block.has_var(param.name):
                    new_params_grads.append((param, grad))
            params_grads = new_params_grads

        else:
            main_block = loss.block
        startup_block = startup_program.global_block()
        self._main_program = main_block.program
        self._startup_program = startup_program

        if self.use_pipeline:
            pp_optimizer._rename_gradient_var_name(main_block)
            with open("main_%d" % self.role_maker._worker_index(), 'w') as f:
                f.writelines(str(main_program))

        # step1: set_up
        self._set_up(params_grads)

        # step2: split_program
        self._split_program(main_block)

        # step3: add broadcast and reduce ops
        self._add_broadcast_allreduce(main_block)
        main_block._sync_with_cpp()
        startup_block._sync_with_cpp()

        # step4: insert reduce_sum for grad
        # grad_scale_coeff = self.role_maker._worker_num()
        # if self._as_outer_parallelism:
        #     grad_scale_coeff = grad_scale_coeff / self._inner_parallelism_size
        # insert_scale_loss_grad_ops(main_block, scale=1.0 / grad_scale_coeff)
        sharding_group_size = self.user_defined_strategy.sharding_configs[
            'sharding_group_size']
        insert_scale_loss_grad_ops(main_block, scale=1.0 / sharding_group_size)
        main_block._sync_with_cpp()

        # step5: remove unneeded ops and vars from block
        self._prune_main_program(main_block)
        self._prune_startup_program(startup_block)
        if self.hybrid_dp:
            self._initialization_broadcast(startup_program)

        if self.use_pipeline:
            # pp_optimizer._rename_gradient_var_name(main_block)
            # crop ops
            for idx, op in reversed(list(enumerate(main_block.ops))):
                if is_update_op(op):
                    op_role_var = op.attr('op_role_var')
                    param_name = op_role_var[0]
                    if not self._shard.has_param(param_name):
                        main_block._remove_op(idx)

            for idx, op in reversed(list(enumerate(main_block.ops))):
                if op.type != 'cast': continue
                in_name = op.input_arg_names[0]
                if in_name not in self._params: continue
                #if self._shard.has_param(param_name): continue
                if in_name not in main_block.vars:
                    main_block._remove_op(idx)
            accumulated_grad_names = pp_optimizer._accumulate_gradients(
                main_block)
            # accumulated_grad_names = sorted(accumulated_grad_names)
            if self.pp_allreduce_in_optimize:
                print("persistable FP32 grad: ")
                print(accumulated_grad_names)
                first_optimize_op_index = get_first_check_finite_and_unscale_op_idx(
                    main_block)
                insert_reduce_ops(
                    main_block,
                    first_optimize_op_index,
                    self.sharding_ring_id,
                    accumulated_grad_names,
                    self._shard,
                    core.op_proto_and_checker_maker.OpRole.Optimize,
                    use_calc_stream=True)

        main_block._sync_with_cpp()

        # TODO(wangxi): add optimize offload
        if self.optimize_offload:
            logging.info("Sharding with optimize offload !")
            offload_helper = OffloadHelper()
            offload_helper.offload(main_block, startup_block)
            offload_helper.offload_fp32param(main_block, startup_block)

        with open("start_sharding_%d" % self.role_maker._worker_index(),
                  'w') as f:
            f.writelines(str(startup_block.program))
        with open("main_sharding_%d" % self.role_maker._worker_index(),
                  'w') as f:
            f.writelines(str(main_block.program))

        # check op dependecy
        check_broadcast(main_block)
        #check_allreduce_sum(main_block, self._shard, self.sharding_ring_id,
        #                    self.dp_ring_id)
        #check_allreduce_sum(main_block, self._shard, self.dp_ring_id)
        # self._wait()
        return optimize_ops, params_grads

    def _set_up(self, params_grads):
        # step 1: initialize nccl
        self.global_word_size = self.role_maker._worker_num()
        self.global_rank = self.role_maker._worker_index()
        self.endpoints = self.role_maker._get_trainer_endpoints()
        self.current_endpoint = self.endpoints[self.global_rank]
        self._collective_helper = CollectiveHelper(self.role_maker,
                                                   self._nrings_sharding)
        # config sharding & dp groups
        self._init_comm()

        # global
        if self._as_outer_parallelism:
            print("global_group_endpoints:", self.global_group_endpoints)
            print("global_rank:", self.global_rank)
            print("global_ring_id:", self.global_group_id)
            self._collective_helper._init_communicator(
                self._startup_program, self.current_endpoint,
                self.global_group_endpoints, self.global_rank,
                self.global_group_id, False)

        if self._as_outer_parallelism:
            print("mp_group_endpoints:", self.mp_group_endpoints)
            print("mp_rank:", self.mp_rank)
            print("mp_ring_id:", self.mp_group_id)
            self._collective_helper._init_communicator(
                self._startup_program, self.current_endpoint,
                self.mp_group_endpoints, self.mp_rank, self.mp_group_id, False)

        # sharding
        print("sharding_group_endpoints:", self.sharding_group_endpoints)
        print("sharding_rank:", self.sharding_rank)
        print("sharding_ring_id:", self.sharding_ring_id)
        self._collective_helper._init_communicator(
            self._startup_program, self.current_endpoint,
            self.sharding_group_endpoints, self.sharding_rank,
            self.sharding_ring_id, False)

        # dp
        if self.hybrid_dp:
            self._collective_helper._init_communicator(
                self._startup_program, self.current_endpoint,
                self.dp_group_endpoints, self.dp_rank, self.dp_ring_id, False)
        # pp
        if self.use_pipeline:
            print("pp_group_endpoints:", self.pp_group_endpoints)
            print("pp_rank:", self.pp_rank)
            print("pp_ring_id:", self.pp_ring_id)
            if self.schedule_mode == 0:  # GPipe
                self._collective_helper._init_communicator(
                    self._startup_program, self.current_endpoint,
                    self.pp_group_endpoints, self.pp_rank, self.pp_ring_id,
                    False)
                self._collective_helper._init_communicator(
                    self._startup_program, self.current_endpoint,
                    self.pp_group_endpoints, self.pp_rank, self.pp_ring_id + 2,
                    False)
            else:
                for pair in self.pipeline_pair:
                    pair_key = pair[0] * 1000 + pair[1]
                    ring_id = self.pp_ring_map[pair_key]
                    print("pp pair:{}, ring_id: {}".format(pair, ring_id))
                    if self.pp_rank not in pair: continue
                    pp_group_endpoints = [
                        self.pp_group_endpoints[pair[0]],
                        self.pp_group_endpoints[pair[1]],
                    ]
                    if pair[0] < pair[1]:
                        start_ring_id = self.pp_ring_id + pair[1] - pair[0] - 1
                    else:
                        start_ring_id = self.pp_ring_id + 2 + pair[0] - pair[
                            1] - 1
                    pp_rank = 0 if self.pp_rank == pair[0] else 1
                    self._collective_helper._init_communicator(
                        self._startup_program, self.current_endpoint,
                        pp_group_endpoints, pp_rank, ring_id, False, False)

        startup_block = self._startup_program.global_block()
        startup_block._sync_with_cpp()

        # step 2: split params
        self._params = set([x[0].name for x in params_grads])
        self._shard.setup(params_grads, self.sharding_rank,
                          self.sharding_group_size)

        # step 3: get broadcast vars
        self._broadcast_vars = self._shard.find_broadcast_params(
            self._main_program.global_block())

    def _wait(self, ):
        # only the first parallelsm group that init nccl need to be wait. 
        if self._as_outer_parallelism:
            endpoints = self.role_maker._get_trainer_endpoints()
            current_endpoint = endpoints[self.role_maker._worker_index()]
        else:
            endpoints = self.sharding_group_endpoints[:]
            current_endpoint = self.sharding_group_endpoints[self.sharding_rank]

        if self._as_outer_parallelism:
            if self.role_maker._worker_index() == 0:
                self._collective_helper._wait(current_endpoint, endpoints)
        else:
            if self.sharding_rank == 0:
                self._collective_helper._wait(current_endpoint, endpoints)

    # def _wait(self, ):
    #     # only the first parallelsm group that init nccl need to be wait. 
    #     if self._as_outer_parallelism:
    #         endpoints = self.role_maker._get_trainer_endpoints()
    #     else:
    #         endpoints = self.sharding_group_endpoints[:]
    #     current_endpoint = endpoints[self.role_maker._worker_index()]

    #     if self._as_outer_parallelism:
    #         if self.role_maker._worker_index() == 0:
    #             self._collective_helper._wait(current_endpoint, endpoints)
    #     else:
    #         if self.sharding_rank == 0:
    #             self._collective_helper._wait(current_endpoint, endpoints)

    def _split_program(self, block):
        for op_idx, op in reversed(list(enumerate(block.ops))):
            if int(op.attr('op_role')) != int(OpRole.Optimize):
                last_backward_op_idx = op_idx + 1
                break
        segment = ProgramSegment(block)
        segment._end_idx = last_backward_op_idx
        for op_idx in reversed(range(last_backward_op_idx)):
            op = block.ops[op_idx]
            assert (int(op.attr('op_role')) != int(OpRole.Optimize))
            if segment._param_mem >= self._fuse_broadcast_MB:
                segment._start_idx = op_idx + 1
                self._segments.insert(0, segment)
                segment = ProgramSegment(block)
                segment._end_idx = op_idx + 1

            # find broadcast vars
            for input_name in op.desc.input_arg_names():
                if input_name not in self._broadcast_vars:
                    continue
                if input_name in segment._param2broadcast:
                    # skip broadcast because it reuse the old broadcast var
                    broadcast_name = segment._param2broadcast[input_name]
                    if input_name != broadcast_name:
                        op._rename_input(input_name, broadcast_name)
                    continue
                if self._shard.has_param(input_name):
                    broadcast_var_name = input_name
                else:
                    broadcast_var_name = unique_name.generate(input_name +
                                                              "@BroadCast")
                    segment._fill_constant_vars.append(broadcast_var_name)
                segment._param2broadcast[input_name] = broadcast_var_name
                segment._broadcast_vars.append((broadcast_var_name,
                                                self._shard.device(input_name)))
                segment._param_mem += get_var_size(
                    self._main_program.global_block().var(input_name))

            # find reduce vars
            if self.use_pipeline and self.pp_allreduce_in_optimize:
                # place pipeline gradient allreduce in optimize
                pass
            else:
                if is_backward_op(op) and \
                        OP_ROLE_VAR_KEY in op.attr_names:
                    op_role_var = op.all_attrs()[OP_ROLE_VAR_KEY]
                    if len(op_role_var) != 0:
                        assert len(op_role_var) % 2 == 0
                        for i in range(0, len(op_role_var), 2):
                            param, reduced_grad = op_role_var[i], op_role_var[
                                i + 1]
                            segment._allreduce_vars.append(reduced_grad)
                            #assert (
                            #    reduced_grad not in self._reduced_grads_to_param)
                            self._reduced_grads_to_param[reduced_grad] = param

            # find cast op
            if FP16Utils.is_fp16_cast_op(block, op, self._params):
                fp32_param = op.desc.input_arg_names()[0]
                fp16_param = op.desc.output_arg_names()[0]
                if self._shard.has_param(fp32_param):
                    segment._cast_ops[fp16_param] = fp32_param

        if segment._param_mem > 0:
            segment._start_idx = 0
            self._segments.insert(0, segment)
        return

    def _prune_main_program(self, block):
        """
        calculate deps from allredce op to optimize op,
        remove ops and vars not needed in this worker

        1. prune regularization (weight decay)
        2. prune cast_fp32_to_fp16; update amp_infine_checking
        3. prune gradient_clip related; update global_norm_sum
        4. prune optimizer op + param + gradient
            
        """
        weightdecay_helper = WeightDecayHelper()
        weightdecay_helper.prune_weight_decay(block, self._shard)
        # NOTE (JZ-LIANG) the sync of FoundInfinite should among one entire Model Parallelism
        # group. and each Data Parallelism group should have its own sync of FoundInfinite
        Model_Paramllelism_ring_id = self.sharding_ring_id
        if self._as_outer_parallelism:
            Model_Paramllelism_ring_id = self.global_group_id
        FP16Utils.prune_fp16(block, self._shard, self._reduced_grads_to_param,
                             Model_Paramllelism_ring_id)
        gradientclip_helper = GradientClipHelper(Model_Paramllelism_ring_id)
        gradientclip_helper.prune_gradient_clip(block, self._shard)

        # build prog deps
        reduced_grads = []
        for idx, op in enumerate(block.ops):
            input_names = op.desc.input_arg_names()
            output_names = op.desc.output_arg_names()
            if op.type == "c_allreduce_sum":
                assert (len(output_names) == 1)
                output_name = output_names[0]
                reduced_grads.append(output_name)

        # prune optimizer state and param
        pruned_opti_vars = []
        for var_name in list(block.vars.keys()):
            if self._shard.is_opti_var(var_name) and \
              not self._shard.has_opt_var(var_name):
                pruned_opti_vars.append(var_name)
        program_deps = ProgramDeps(block, reduced_grads, pruned_opti_vars)

        # Init
        for var_name in program_deps._end_vars:
            program_deps._should_removed_var.add(var_name)

        # Prune
        for idx, op in reversed(list(enumerate(block.ops))):
            if op.type in [
                    "c_allreduce_sum",
                    "c_sync_comm_stream",
                    "c_calc_comm_stream",
                    "c_gen_nccl_id",
                    "c_comm_init",
                    'send_v2',
                    'recv_v2',
            ]:
                pass
            elif op.type == "conditional_block":
                assert (op.desc.has_attr("sub_block"))
                subblock_idx = op.desc.attr("sub_block").id
                subblock_deps = program_deps.get_sub_block_deps(subblock_idx)
                # only prune amp subblock
                if subblock_deps is None or not self._is_amp_subblock(op):
                    continue
                # init
                reversed_output_vars = []
                for output_name in op.desc.output("Out"):
                    if output_name in program_deps._should_removed_var:
                        subblock_deps._should_removed_var.add(output_name)
                        program_deps.crop_output_var_from_op(idx, output_name)
                    else:
                        reversed_output_vars.append(output_name)
                # prune
                for sub_op_idx, _ in reversed(
                        list(enumerate(subblock_deps._block.ops))):
                    if subblock_deps.should_remove_op(sub_op_idx):
                        subblock_deps.remove_op(sub_op_idx)
                reversed_input_vars = []
                for input_name in op.desc.input('Input'):
                    if input_name not in subblock_deps._should_removed_var:
                        reversed_input_vars.append(input_name)
                    else:
                        program_deps.crop_input_var_from_op(idx, input_name)
                op.desc.set_input('Input', reversed_input_vars)
                op.desc.set_output('Out', reversed_output_vars)
            else:
                # if all outputs of this op are in _should_removed_var
                # _should_removed_var: opt state not cur shard
                if program_deps.should_remove_op(idx):
                    program_deps.remove_op(idx)

        block._sync_with_cpp()
        for idx, op in reversed(list(enumerate(block.ops))):
            if op.type == 'concat' and is_optimizer_op(op):
                # remove inputs that not on this card
                reserved_x = []
                for var_name in op.desc.input("X"):
                    if block.has_var(var_name): reserved_x.append(var_name)
                op.desc.set_input('X', reserved_x)
        block._sync_with_cpp()
        return

    def _add_broadcast_allreduce(self, block):
        """
        _add_broadcast_allreduce

        if combined with pipeline(grad accumulate), 
        the grad allreduce should be done in optimize role
        """
        if len(self._segments) < 1:
            return
        # sharding
        if self.use_pipeline and self.pp_allreduce_in_optimize:
            for idx in range(len(self._segments)):
                assert len(self._segments[idx]._allreduce_vars) == 0

        # fix the _end_idx for segments[-1] if pp is used.
        new_end_idx = self._segments[-1]._end_idx
        for idx in range(self._segments[-1]._end_idx - 1,
                         self._segments[-1]._start_idx - 1, -1):
            op = block.ops[idx]
            if op.type == "fill_constant" or op.type == "sum":
                if "MERGED" in op.output_arg_names[0]: new_end_idx = idx + 1
            elif op.type == "cast":
                if "@TMP" in op.output_arg_names[0]: new_end_idx = idx + 1
        self._segments[-1]._end_idx = new_end_idx

        if self._segments[-1]._allreduce_vars:
            shard_allredue_vars = self._shard.filter_grads(self._segments[-1]
                                                           ._allreduce_vars)
            if self.hybrid_dp and len(shard_allredue_vars) >= 1:
                insert_sync_comm_ops(block, self._segments[-1]._end_idx,
                                     self.dp_ring_id, shard_allredue_vars)
                insert_allreduce_ops(block, self._segments[-1]._end_idx,
                                     self.dp_ring_id, shard_allredue_vars)
            insert_sync_comm_ops(block, self._segments[-1]._end_idx,
                                 self.sharding_ring_id,
                                 self._segments[-1]._allreduce_vars)
            # allreduce --> reduce
            insert_reduce_ops(
                block,
                self._segments[-1]._end_idx,
                self.sharding_ring_id,
                self._segments[-1]._allreduce_vars,
                self._shard,
                op_role=OpRole.Backward,
                use_calc_stream=False)

        for idx, segment in reversed(list(enumerate(self._segments))):
            allreduce_vars = self._segments[
                idx - 1]._allreduce_vars if idx > 0 else []
            broadcast_vars = self._segments[idx +
                                            1]._broadcast_vars if idx < len(
                                                self._segments) - 1 else []
            fill_constant_vars = self._segments[
                idx + 2]._fill_constant_vars if idx < len(
                    self._segments) - 2 else []
            cast_ops = self._segments[idx + 2]._cast_ops if idx < len(
                self._segments) - 2 else {}

            for op_idx in reversed(range(segment._start_idx, segment._end_idx)):
                op = block.ops[op_idx]
                for input_name in op.desc.input_arg_names():
                    if input_name in segment._param2broadcast and \
                        input_name != segment._param2broadcast[input_name]:
                        op._rename_input(input_name,
                                         segment._param2broadcast[input_name])

            for param_name, broadcast_name in segment._param2broadcast.items():
                if param_name != broadcast_name:
                    block.create_var(
                        name=broadcast_name,
                        shape=self._main_program.global_block().var(
                            param_name).shape,
                        dtype=self._main_program.global_block().var(param_name)
                        .dtype,
                        persistable=False)

            # step1: remove cast ops
            block._sync_with_cpp()
            segment._end_idx += FP16Utils.remove_cast_op(block, self._params,
                                                         segment, 0)

            # step2: add Sync ops
            shard_allredue_vars = self._shard.filter_grads(allreduce_vars)
            if self.hybrid_dp and len(shard_allredue_vars) >= 1:
                insert_sync_comm_ops(block, segment._end_idx, self.dp_ring_id,
                                     shard_allredue_vars)

                broad_cast_vars = [x[0] for x in broadcast_vars]
                if len(broad_cast_vars) > 0:
                    insert_sync_comm_ops(block, segment._end_idx,
                                         self.sharding_ring_id, broad_cast_vars)
            else:
                comm_dep_vars = allreduce_vars + [x[0] for x in broadcast_vars]
                if len(comm_dep_vars) > 0:
                    insert_sync_comm_ops(block, segment._end_idx,
                                         self.sharding_ring_id, comm_dep_vars)

            calc_dep_vars = fill_constant_vars + [
                k for k, v in cast_ops.items()
            ] + self._segments[idx]._allreduce_vars

            if len(calc_dep_vars) > 0:
                insert_sync_calc_op(block, segment._end_idx,
                                    [calc_dep_vars[-1]])

            # step3: insert `fill_constant` ops 
            insert_fill_constant_ops(block, segment._end_idx,
                                     fill_constant_vars)

            # step4: add `cast` ops     
            print("cast_ops:", cast_ops)
            insert_cast_ops(block, segment._end_idx, cast_ops)

            # step5: add broadcast ops
            insert_broadcast_ops(block, segment._start_idx,
                                 self.sharding_ring_id, broadcast_vars)
            # step6: add all_reduce ops
            # dp
            if self.hybrid_dp and len(shard_allredue_vars) >= 1:
                insert_allreduce_ops(block, segment._start_idx, self.dp_ring_id,
                                     shard_allredue_vars)
                insert_sync_comm_ops(block, segment._start_idx,
                                     self.sharding_ring_id, allreduce_vars)
            # sharding
            # allreduce --> reduce
            insert_reduce_ops(
                block,
                segment._start_idx,
                self.sharding_ring_id,
                allreduce_vars,
                self._shard,
                op_role=OpRole.Backward,
                use_calc_stream=False)

            block._sync_with_cpp()

        if self._segments[0]._broadcast_vars:
            broadcast_vars = [x[0] for x in self._segments[0]._broadcast_vars]
            insert_sync_comm_ops(block, self._segments[0]._start_idx,
                                 self.sharding_ring_id, broadcast_vars)
            insert_broadcast_ops(block, self._segments[0]._start_idx,
                                 self.sharding_ring_id,
                                 self._segments[0]._broadcast_vars)

        fill_constant_vars = []
        for x in self._segments[:2]:
            fill_constant_vars += x._fill_constant_vars

        # Join
        cast_ops = {}
        for x in self._segments[:2]:
            for k, v in x._cast_ops.items():
                cast_ops[k] = v

        calc_deps_vars = fill_constant_vars + [k for k, v in cast_ops.items()]
        if fill_constant_vars or cast_ops:
            insert_sync_calc_op(block, self._segments[0]._start_idx,
                                [calc_deps_vars[-1]])

        if fill_constant_vars:
            insert_fill_constant_ops(block, self._segments[0]._start_idx,
                                     fill_constant_vars)

        if cast_ops:
            insert_cast_ops(block, self._segments[0]._start_idx, cast_ops)

        return

    def _prune_startup_program(self, block):
        for idx, op in reversed(list(enumerate(block.ops))):
            for output_name in op.desc.output_arg_names():
                if self._shard.has_var(output_name):
                    continue
                #TODO why do we remove op, when only one var is removed
                block._remove_op(idx, sync=False)
                break

        for var_name in list(block.vars.keys()):
            if self._shard.has_var(var_name):
                continue
            block._remove_var(var_name, sync=False)
        block._sync_with_cpp()

    def _init_comm(self):

        if self.hybrid_dp:
            assert self._as_outer_parallelism == False, "hybrid dp is conflict when using sharding as outer parallelism"
            self.sharding_group_size = self.user_defined_strategy.sharding_configs[
                "sharding_group_size"]
            self.sharding_ring_id = 0
            self.sharding_rank = self.global_rank % self.sharding_group_size

            self.dp_group_size = self.global_word_size // self.sharding_group_size
            self.dp_rank = self.global_rank // self.sharding_group_size
            self.dp_ring_id = self.sharding_rank + 1

            self.sharding_group_endpoints = [
                ep for idx, ep in enumerate(self.endpoints)
                if (idx // self.sharding_group_size) == self.dp_rank
            ]
            self.dp_group_endpoints = [
                ep for idx, ep in enumerate(self.endpoints)
                if (idx % self.sharding_group_size) == self.sharding_rank
            ]

            assert self.global_word_size > self.sharding_group_size, \
                "global_word_size: {} should be larger than sharding_group_size: {}".format(self.global_word_size, self.sharding_group_size)
            assert self.global_word_size % self.sharding_group_size == 0, \
                "global_word_size: {} should be divisible to the sharding_group_size: {}".format(self.global_word_size, self.sharding_group_size)
            assert self.dp_group_size *  self.sharding_group_size == self.global_word_size, \
                "global_word_size: {} should be equal to the product of sharding_group_size: {} and dp_group_size: {}".format(
                self.global_word_size,
                self.sharding_group_size,
                self.dp_group_size)
            self.pp_ring_id = -1
            self.pp_rank = -1
            self.pp_group_size = None
            self.pp_group_endpoints = None

            # sharding parallelism is the only model parallelism in the current setting
            self.mp_group_id = self.sharding_ring_id
            self.mp_rank = self.sharding_rank
            self.mp_group_size = self.sharding_group_size
            self.mp_group_endpoints = self.sharding_group_endpoints[:]

            logging.info("Using Sharing&DP mode !")
        else:
            if self._as_outer_parallelism and not self.use_pipeline:
                self.sharding_ring_id = 1
                assert self.global_word_size > self._inner_parallelism_size, \
                    "global_word_size: {} should be larger than inner_parallelism_size: {}".format(self.global_word_size, self._inner_parallelism_size)
                assert self.global_word_size % self._inner_parallelism_size == 0, \
                    "global_word_size: {} should be divisible to the inner_parallelism_size: {}".format(self.global_word_size, self._inner_parallelism_size)
                self.sharding_rank = self.global_rank // self._inner_parallelism_size
                self.sharding_group_size = self.role_maker._worker_num(
                ) // self._inner_parallelism_size
                _offset = self.global_rank % self._inner_parallelism_size
                self.sharding_group_endpoints = [
                    ep for idx, ep in enumerate(self.endpoints)
                    if idx % self._inner_parallelism_size == _offset
                ]

                # the current entire model parallelism group is the combination of innert & sharding parallelism
                self.mp_group_id = 2
                self.mp_rank = self.global_rank
                self.mp_group_size = self.role_maker._worker_num()
                self.mp_group_endpoints = self.endpoints[:]
                logging.info("Using Sharing as Outer parallelism mode !")

                # print(
                #     "init the nccl comm for megatron paramllelism, this should be done in Megatron Metaoptimizer"
                # )
                # partition_idx = self.global_rank // self._inner_parallelism_size
                # magetron_endpoints = self.endpoints[
                #     partition_idx * self._inner_parallelism_size:partition_idx *
                #     self._inner_parallelism_size + self._inner_parallelism_size]
                # magetron_rank = self.global_rank % self._inner_parallelism_size

                # self._collective_helper._init_communicator(
                #     program=self._startup_program,
                #     current_endpoint=self.current_endpoint,
                #     endpoints=magetron_endpoints,
                #     rank=magetron_rank,
                #     ring_id=0,
                #     wait_port=True)
                # logging.info("megatron group size: {}".format(
                #     self._inner_parallelism_size))
                # logging.info("megatron rank: {}".format(magetron_rank))
                # logging.info("megatron endpoints: {}".format(
                #     magetron_endpoints))
            if self.use_pipeline:
                if self._inner_parallelism_size == 1:
                    self.sharding_ring_id = 0
                    self.sharding_group_size = self.user_defined_strategy.sharding_configs[
                        'sharding_group_size']
                    self.sharding_rank = self.global_rank % self.sharding_group_size
                    assert self.sharding_group_size * self.pipeline_nodes * self._inner_parallelism_size == self.role_maker._worker_num(
                    )
                    self.pp_ring_id = 20
                    self.pp_rank = self.global_rank // (
                        self.sharding_group_size * self._inner_parallelism_size)
                    self.sharding_group_endpoints = [
                        ep for idx, ep in enumerate(self.endpoints)
                        if (idx // self.sharding_group_size) == self.pp_rank
                    ]
                    self.pp_group_size = self.pipeline_nodes
                    self.pp_group_endpoints = [
                        ep for idx, ep in enumerate(self.endpoints)
                        if (idx % self.sharding_group_size
                            ) == self.sharding_rank
                    ]
                else:
                    self.mp_group_id = 0
                    self.sharding_ring_id = 1
                    self.pp_ring_id = 20
                    self.mp_rank = self.global_rank % self._inner_parallelism_size
                    self.mp_group = self.global_rank // self._inner_parallelism_size
                    self.mp_group_endpoints = [
                        ep for idx, ep in enumerate(self.endpoints)
                        if idx // self._inner_parallelism_size == self.mp_group
                    ]
                    print("megatron_group_endpoints:", self.mp_group_endpoints)
                    print("megatron_rank:", self.mp_rank)
                    # self.cards_per_node = 8
                    self.sharding_group_size = self.user_defined_strategy.sharding_configs[
                        'sharding_group_size']
                    self.sharding_rank = (
                        self.global_rank //
                        self._inner_parallelism_size) % self.sharding_group_size
                    self.sharding_group_id = self.global_rank // (
                        self._inner_parallelism_size * self.sharding_group_size)
                    self.megatron_rank = self.global_rank % self._inner_parallelism_size
                    self.sharding_group_endpoints = [
                        ep for idx, ep in enumerate(self.endpoints)
                        if (idx // (self._inner_parallelism_size *
                                    self.sharding_group_size)
                            ) == self.sharding_group_id and idx %
                        self._inner_parallelism_size == self.megatron_rank
                    ]
                    print("sharding_endpoint:", self.sharding_group_endpoints)
                    print("sharding_rank:", self.sharding_rank)
                    assert self.sharding_group_size * self.pipeline_nodes * self._inner_parallelism_size == self.role_maker._worker_num(
                    )
                    self.pp_rank = self.global_rank // (
                        self.sharding_group_size *
                        self._inner_parallelism_size) % self.pipeline_nodes
                    offset = self.sharding_group_size * self._inner_parallelism_size
                    # TODO: Adjust for dp
                    idx_with_pp_0 = self.global_rank % (
                        self.sharding_group_size * self._inner_parallelism_size)
                    self.pp_group_endpoints = []
                    for i in range(self.pipeline_nodes):
                        self.pp_group_endpoints.append(self.endpoints[
                            idx_with_pp_0])
                        idx_with_pp_0 += offset
                    print("pp_group_endpoints:", self.pp_group_endpoints)
                    print("pp_rank:", self.pp_rank)

                    #self.pp_group_endpoints = [
                    #    ep for idx, ep in enumerate(self.endpoints)
                    #    if (idx % self.sharding_group_size) == self.sharding_rank
                    #]
                self.global_group_id = 3
                self.global_rank = self.global_rank
                self.global_group_size = self.role_maker._worker_num()
                self.global_group_endpoints = self.endpoints[:]
                logging.info("Using Sharing as Outer parallelism mode !")
                self.dp_ring_id = -1
                self.dp_rank = -1
                self.dp_group_size = None
                self.dp_group_endpoints = None

                logging.info("Using Sharing with pipeline !")
            #else:
            #    self.sharding_ring_id = 0
            #    self.sharding_rank = self.global_rank
            #    self.sharding_group_size = self.role_maker._worker_num()
            #    self.sharding_group_endpoints = self.endpoints

            #    # sharding parallelism is the only model parallelism in the current setting
            #    self.mp_group_id = self.sharding_ring_id
            #    self.mp_rank = self.sharding_rank
            #    self.mp_group_size = self.sharding_group_size
            #    self.mp_group_endpoints = self.sharding_group_endpoints[:]

            #    logging.info("Using Sharing alone mode !")

            self.dp_ring_id = -1
            self.dp_rank = -1
            self.dp_group_size = None
            self.dp_group_endpoints = None

            #self.pp_ring_id = -1
            #self.pp_rank = -1
            #self.pp_group_size = None
            #self.pp_group_endpoints = None
            #self.dp_ring_id = -1
            #self.dp_rank = -1
            #self.dp_group_size = None
            #self.dp_group_endpoints = None

            logging.info("Using Sharing alone mode !")

        #logging.info("global word size: {}".format(self.global_word_size))
        #logging.info("global rank: {}".format(self.global_rank))
        #logging.info("sharding group_size: {}".format(self.sharding_group_size))
        #logging.info("sharding rank: {}".format(self.sharding_rank))
        #logging.info("current model parallelism group_size: {}".format(
        #    self.mp_group_size))
        #logging.info("current model parallelism rank: {}".format(self.mp_rank))
        #logging.info("dp group size: {}".format(self.dp_group_size))
        #logging.info("dp rank: {}".format(self.dp_rank))
        #logging.info("current endpoint: {}".format(self.current_endpoint))
        #logging.info("global word endpoints: {}".format(self.endpoints))
        #logging.info("sharding group endpoints: {}".format(
        #    self.sharding_group_endpoints))
        #logging.info("current model parallelism group endpoints: {}".format(
        #    self.mp_group_endpoints))
        #logging.info("dp group endpoints: {}".format(self.dp_group_endpoints))

        return

    def _initialization_broadcast(self, startup_prog):
        """
        this funtion is to ensure the initialization between dp group to be 
        identical when hybrid-dp is used.
        """
        block = startup_prog.global_block()
        params = []
        for param in block.iter_parameters():
            params.append(param)
            block.append_op(
                type='c_broadcast',
                inputs={'X': param},
                outputs={'Out': param},
                attrs={
                    'ring_id': self.dp_ring_id,
                    'root': 0,
                    OP_ROLE_KEY: OpRole.Forward
                })
        block.append_op(
            type='c_sync_comm_stream',
            inputs={'X': params},
            outputs={'Out': params},
            attrs={'ring_id': self.dp_ring_id,
                   OP_ROLE_KEY: OpRole.Forward})
