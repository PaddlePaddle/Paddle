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
from paddle.distributed.fleet.meta_optimizers.common import is_backward_op
from paddle.distributed.fleet.meta_optimizers.meta_optimizer_base import MetaOptimizerBase
from paddle.distributed.fleet.meta_optimizers.sharding.shard import Shard, ProgramSegment
from paddle.distributed.fleet.meta_optimizers.sharding.fp16_helper import FP16Utils
from paddle.distributed.fleet.meta_optimizers.sharding.weight_decay_helper import WeightDecayHelper
from paddle.distributed.fleet.meta_optimizers.sharding.gradient_clip_helper import GradientClipHelper
from paddle.distributed.fleet.meta_optimizers.sharding.prune import ProgramDeps
from paddle.distributed.fleet.meta_optimizers.sharding.utils import *

from functools import reduce

__all__ = ["ShardingOptimizer"]


class ShardingOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(ShardingOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        self.meta_optimizers_white_list = [
            "RecomputeOptimizer",
            "AMPOptimizer",
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
        self._nrings = self.user_defined_strategy.nccl_comm_num
        self._fuse_broadcast_MB = self.user_defined_strategy.sharding_configs[
            "fuse_broadcast_MB"]

        if self.inner_opt is None:
            raise ValueError(
                "self.inner_opt of ShardingOptimizer should not be None.")
        optimize_ops, params_grads = self.inner_opt.minimize(
            loss, startup_program, parameter_list, no_grad_set)

        if startup_program is None:
            startup_program = default_startup_program()
        main_block = loss.block
        startup_block = startup_program.global_block()
        self._main_program = main_block.program
        self._startup_program = startup_program

        # step1: set_up
        self._set_up(params_grads)

        # step2: split_program
        self._split_program(main_block)

        # step3: add broadcast and reduce ops
        self._add_broadcast_allreduce(main_block)
        main_block._sync_with_cpp()
        startup_block._sync_with_cpp()

        # step4: insert reduce_sum for grad
        insert_scale_loss_grad_ops(
            main_block, scale=1.0 / self.role_maker._worker_num())
        main_block._sync_with_cpp()

        # step5: remove unneeded ops and vars from block
        self._prune_main_program(main_block)
        self._prune_startup_program(startup_block)

        # check op dependecy
        check_broadcast(main_block)
        check_allreduce_sum(main_block)
        self._wait()
        return optimize_ops, params_grads

    def _set_up(self, params_grads):
        # step 1: initialize nccl
        worker_idx = self.role_maker._worker_index()
        endpoints = self.role_maker._get_trainer_endpoints()
        current_endpoint = endpoints[worker_idx]
        self._collective_helper = CollectiveHelper(self.role_maker,
                                                   self._nrings)
        for ring_id in range(self._nrings):
            self._collective_helper._init_communicator(
                self._startup_program, current_endpoint, endpoints, worker_idx,
                ring_id, None)
        startup_block = self._startup_program.global_block()
        startup_block._sync_with_cpp()

        # step 2: split params
        self._params = set([x[0].name for x in params_grads])
        self._shard.setup(params_grads, worker_idx,
                          self.role_maker._worker_num())

        # step 3: get broadcast vars
        self._broadcast_vars = self._shard.find_broadcast_params(
            self._main_program.global_block())

    def _wait(self, ):
        endpoints = self.role_maker._get_trainer_endpoints()
        current_endpoint = endpoints[self.role_maker._worker_index()]
        if self.role_maker._worker_index() == 0:
            self._collective_helper._wait(current_endpoint, endpoints)

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
            if is_backward_op(op) and \
                    OP_ROLE_VAR_KEY in op.attr_names:
                op_role_var = op.all_attrs()[OP_ROLE_VAR_KEY]
                if len(op_role_var) != 0:
                    assert len(op_role_var) % 2 == 0
                    for i in range(0, len(op_role_var), 2):
                        param, reduced_grad = op_role_var[i], op_role_var[i + 1]
                        segment._allreduce_vars.append(reduced_grad)
                        assert (
                            reduced_grad not in self._reduced_grads_to_param)
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
        """
        weightdecay_helper = WeightDecayHelper()
        weightdecay_helper.prune_weight_decay(block, self._shard)
        FP16Utils.prune_fp16(block, self._shard, self._reduced_grads_to_param,
                             self._nrings)
        gradientclip_helper = GradientClipHelper()
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
                    "c_allreduce_sum", "c_sync_comm_stream",
                    "c_calc_comm_stream", "c_gen_nccl_id", "c_comm_init"
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
                if program_deps.should_remove_op(idx):
                    program_deps.remove_op(idx)

        block._sync_with_cpp()
        return

    def _add_broadcast_allreduce(self, block):
        """
        _add_broadcast_allreduce
        """
        ring_id = -1
        if len(self._segments) < 1:
            return

        if self._segments[-1]._allreduce_vars:
            insert_sync_comm_ops(block, self._segments[-1]._end_idx,
                                 self._nrings,
                                 self._segments[-1]._allreduce_vars)
            insert_allreduce_ops(block, self._segments[-1]._end_idx,
                                 self._nrings,
                                 self._segments[-1]._allreduce_vars)

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
            comm_dep_vars = allreduce_vars + [x[0] for x in broadcast_vars]
            if len(comm_dep_vars) > 0:
                insert_sync_comm_ops(
                    block,
                    segment._end_idx,
                    self._nrings,
                    comm_dep_vars, )
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
            insert_cast_ops(block, segment._end_idx, cast_ops)

            # step5: add broadcast ops
            insert_broadcast_ops(block, segment._start_idx, self._nrings,
                                 broadcast_vars)

            # step6: add all_reduce ops
            insert_allreduce_ops(block, segment._start_idx, self._nrings,
                                 allreduce_vars)

            block._sync_with_cpp()

        if self._segments[0]._broadcast_vars:
            insert_sync_comm_ops(
                block, self._segments[0]._start_idx, self._nrings,
                [x[0] for x in self._segments[0]._broadcast_vars])
            insert_broadcast_ops(block, self._segments[0]._start_idx,
                                 self._nrings,
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
