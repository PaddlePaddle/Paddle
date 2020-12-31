#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
from __future__ import division

import paddle.fluid as fluid
from paddle.fluid import core, unique_name
from ..base.private_helper_function import wait_server_ready
from paddle.fluid.optimizer import PipelineOptimizer as PO
from .meta_optimizer_base import MetaOptimizerBase
from .common import OpRole, OP_ROLE_KEY, OP_ROLE_VAR_KEY, CollectiveHelper, is_update_op, is_loss_grad_op, is_backward_op, is_optimizer_op


def _get_node_num(endpoints):
    ss = set()
    for ep in endpoints:
        ip = ep.split(":")[0].strip()
        if ip not in ss:
            ss.add(ip)
    return len(ss)


class PipelineHelper(object):
    def __init__(self, role_maker, wait_port='6174'):
        self.wait_port = wait_port
        self.role_maker = role_maker

    def update_startup_program(self,
                               startup_program=None,
                               inner_parallelism=None):
        self.startup_program = startup_program

        nranks = self.role_maker._worker_num()
        rank = self.role_maker._worker_index()
        endpoints = self.role_maker._get_trainer_endpoints()
        current_endpoint = endpoints[rank]
        node_num = _get_node_num(endpoints)
        assert nranks % node_num == 0

        # Create ring 0 for all gpus in the same pipeline
        if inner_parallelism > 1:
            pipeline_rank = rank % inner_parallelism
            pipeline_id = rank // inner_parallelism
            start_index = pipeline_id * inner_parallelism
            pipeline_endpoints = endpoints[start_index:start_index +
                                           inner_parallelism]
            self._init_communicator(self.startup_program, current_endpoint,
                                    pipeline_endpoints, pipeline_rank, 0,
                                    self.wait_port)

        pipeline_num = len(endpoints) // inner_parallelism
        if pipeline_num == 1: return
        # Create rings for gpus with the same pipeline id for data parallel
        eps = []
        pipeline_rank = rank % inner_parallelism
        ring_id = pipeline_rank + 1
        for i in range(pipeline_num):
            eps.append(endpoints[i * inner_parallelism + pipeline_rank])
        # rank in a ring of gpus with the same pipeline id for data parallel
        dp_rank = rank // inner_parallelism
        self._init_communicator(self.startup_program, current_endpoint, eps,
                                dp_rank, ring_id, self.wait_port)
        self._broadcast_params(ring_id)

    def _init_communicator(self, program, current_endpoint, endpoints, rank,
                           ring_id, wait_port):
        nranks = len(endpoints)
        other_endpoints = endpoints[:]
        other_endpoints.remove(current_endpoint)
        if rank == 0 and wait_port:
            wait_server_ready(other_endpoints)

        block = program.global_block()
        nccl_id_var = block.create_var(
            name=unique_name.generate('nccl_id'),
            persistable=True,
            type=core.VarDesc.VarType.RAW)
        block.append_op(
            type='c_gen_nccl_id',
            inputs={},
            outputs={'Out': nccl_id_var},
            attrs={
                'rank': rank,
                'endpoint': current_endpoint,
                'other_endpoints': other_endpoints,
                OP_ROLE_KEY: OpRole.Forward,
            })
        block.append_op(
            type='c_comm_init',
            inputs={'X': nccl_id_var},
            outputs={},
            attrs={
                'nranks': nranks,
                'rank': rank,
                'ring_id': ring_id,
                OP_ROLE_KEY: OpRole.Forward,
            })

    def _broadcast_params(self, ring_id):
        block = self.startup_program.global_block()
        for var_name in block.vars:
            if "nccl_id" in var_name: continue
            param = block.var(var_name)
            if not param.persistable:
                continue

            block.append_op(
                type='c_broadcast',
                inputs={'X': param},
                outputs={'Out': param},
                attrs={
                    'ring_id': ring_id,
                    'root': 0,
                    OP_ROLE_KEY: OpRole.Forward
                })

        block.append_op(
            type='c_sync_comm_stream',
            inputs={'X': param},
            outputs={'Out': param},
            attrs={'ring_id': ring_id,
                   OP_ROLE_KEY: OpRole.Forward})


class PipelineOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(PipelineOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = []
        self.meta_optimizers_black_list = ["GraphExecutionOptimizer", ]

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        super(PipelineOptimizer, self)._set_basic_info(
            loss, role_maker, user_defined_optimizer, user_defined_strategy)
        self.num_microbatches = user_defined_strategy.pipeline_configs[
            'micro_batch']

    def _can_apply(self):
        if not self.role_maker._is_collective:
            return False

        if self.user_defined_strategy.pipeline == True:
            return True
        return False

    def _disable_strategy(self, dist_strategy):
        dist_strategy.pipeline = False
        dist_strategy.pipeline_configs = {}

    def _enable_strategy(self, dist_strategy, context):
        dist_strategy.pipeline = True
        dist_strategy.pipeline_configs = {"micro_batch": 1, }

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        endpoints = self.role_maker._get_trainer_endpoints()
        current_endpoint = endpoints[self.role_maker._worker_index()]
        self.wrapped_opt = PO(self.inner_opt,
                              num_microbatches=self.num_microbatches)
        node_num = _get_node_num(endpoints)
        gpus_per_node = len(endpoints) // node_num
        self.startup_program = startup_program
        if startup_program is None:
            self.startup_program = fluid.default_startup_program()

        self.rank = self.role_maker._worker_index()
        self.nranks = self.role_maker._worker_num()
        assert self.nranks % node_num == 0

        loss.block.program._pipeline_opt = dict()
        loss.block.program._pipeline_opt['local_rank'] = self.rank
        optimize_ops, params_grads, prog_list = self.wrapped_opt.minimize(
            loss, startup_program, parameter_list, no_grad_set)
        assert prog_list

        self.main_program_list = prog_list
        self.main_program = loss.block.program
        self.inner_parallelism = loss.block.program._pipeline_opt[
            'inner_parallelism']
        assert self.nranks % self.inner_parallelism == 0

        pipeline_helper = PipelineHelper(self.role_maker)
        pipeline_helper.update_startup_program(
            self.startup_program._pipeline_opt["startup_program"],
            self.inner_parallelism)

        pipeline_num = self.nranks // self.inner_parallelism
        self._transpile_main_program(loss, pipeline_num, self.inner_parallelism)
        return optimize_ops, params_grads

    def _transpile_main_program(self, loss, pipeline_num, inner_parallelism):
        if pipeline_num <= 1: return
        self._insert_loss_grad_ops(loss, pipeline_num)
        for ring_id in range(1, inner_parallelism + 1):
            self._insert_allreduce_ops(ring_id)

    def _insert_loss_grad_ops(self, loss, pipeline_num):
        """
        In order to keep the learning rate consistent in different numbers of
        training workers, we scale the loss grad by the number of workers
        """
        block = self.main_program_list[-1]['program'].global_block()
        for idx, op in reversed(list(enumerate(block.ops))):
            if is_loss_grad_op(op):
                loss_grad_var = block.vars[op.output_arg_names[0]]
                block._insert_op(
                    idx + 1,
                    type='scale',
                    inputs={'X': loss_grad_var},
                    outputs={'Out': loss_grad_var},
                    attrs={
                        'scale': 1.0 / pipeline_num,
                        OP_ROLE_KEY: OpRole.Backward
                    })

    def _insert_allreduce_ops(self, ring_id):
        block = self.main_program_list[ring_id - 1]['program'].global_block()
        origin_block = self.main_program.global_block()
        grad = None
        for idx, op in reversed(list(enumerate(block.ops))):
            if is_backward_op(op) and \
                    OP_ROLE_VAR_KEY in op.attr_names:
                op_role_var = op.all_attrs()[OP_ROLE_VAR_KEY]
                if len(op_role_var) == 0:
                    continue
                assert len(op_role_var) % 2 == 0
                offset = idx
                for i in range(0, len(op_role_var), 2):
                    param = block.vars[op_role_var[i]]
                    grad = block.vars[op_role_var[i + 1]]
                    origin_param = origin_block.vars[op_role_var[i]]
                    if origin_param.is_distributed:
                        continue
                    if offset == idx:
                        offset += 1
                        block._insert_op(
                            offset,
                            type='c_sync_calc_stream',
                            inputs={'X': grad},
                            outputs={'Out': grad},
                            attrs={OP_ROLE_KEY: OpRole.Backward})
                        offset += 1

                    block._insert_op(
                        offset,
                        type='c_allreduce_sum',
                        inputs={'X': grad},
                        outputs={'Out': grad},
                        attrs={
                            'ring_id': ring_id,
                            OP_ROLE_KEY: OpRole.Backward
                        })

        if grad is None:
            return

        for idx, op in enumerate(block.ops):
            if is_optimizer_op(op):
                block._insert_op(
                    idx,
                    type='c_sync_comm_stream',
                    inputs={'X': grad},
                    outputs={'Out': grad},
                    attrs={'ring_id': ring_id,
                           OP_ROLE_KEY: OpRole.Backward})
            break
