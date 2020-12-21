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

        endpoints = self.role_maker._get_trainer_endpoints()
        current_endpoint = endpoints[self.role_maker._worker_index()]
        node_num = _get_node_num(endpoints)
        assert len(endpoints) % node_num == 0
        nranks = self.role_maker._worker_num()
        rank = self.role_maker._worker_index()

        # Create ring 0 for all gpus in a pipeline
        pipeline_endpoints = []
        pipeline_rank = rank % inner_parallelism
        pipeline_id = rank // inner_parallelism
        for idx, ep in enumerate(endpoints):
            if idx // inner_parallelism == pipeline_id:
                pipeline_endpoints.append(ep)
        self._init_communicator(self.startup_program, current_endpoint,
                                pipeline_endpoints, pipeline_rank, 0,
                                self.wait_port)

        pipeline_num = len(endpoints) // inner_parallelism
        if pipeline_num == 1: return
        # Create rings for gpus with the same gpu id
        eps = []
        local_rank = self.role_maker._worker_index() % inner_parallelism
        ring_id = local_rank + 1
        for i in range(pipeline_num):
            eps.append(endpoints[i * inner_parallelism + local_rank])
        temp_rank = self.role_maker._worker_index() // inner_parallelism
        self._init_communicator(self.startup_program, current_endpoint, eps,
                                temp_rank, ring_id, self.wait_port)
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
        for param in block.iter_parameters():
            if param.is_distributed:
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
        self.meta_optimizers_black_list = []

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

    def _get_local_rank(self, current_endpoint, endpoints):
        cur_node_endpoints = []
        cur_ip = current_endpoint.split(':')[0].strip()
        for ep in endpoints:
            if cur_ip == ep.split(':')[0].strip():
                cur_node_endpoints.append(ep)
        return cur_node_endpoints.index(current_endpoint)

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        endpoints = self.role_maker._get_trainer_endpoints()
        current_endpoint = endpoints[self.role_maker._worker_index()]
        self.local_rank = self._get_local_rank(current_endpoint, endpoints)
        self.wrapped_opt = PO(self.inner_opt,
                              num_microbatches=self.num_microbatches,
                              start_cpu_core_id=self.local_rank)
        node_num = _get_node_num(endpoints)
        gpus_per_node = len(endpoints) // node_num
        self.startup_program = startup_program
        self.local_rank = self._get_local_rank(current_endpoint, endpoints)
        if startup_program is None:
            self.startup_program = fluid.default_startup_program()

        loss.block.program._pipeline_opt = dict()
        loss.block.program._pipeline_opt['local_rank'] = self.local_rank
        optimize_ops, params_grads, prog_list = \
            self.wrapped_opt.minimize(loss, startup_program,
                                      parameter_list, no_grad_set)

        assert prog_list
        self.main_program_list = prog_list
        self.main_program = loss.block.program
        self.inner_parallelism = loss.block.program._pipeline_opt[
            'inner_parallelism']
        nranks = len(endpoints)
        self.nranks = nranks
        self.nrings = len(self.main_program_list)

        self.rank = self.role_maker._worker_index()
        self.endpoints = endpoints
        self.current_endpoint = current_endpoint

        pipeline_helper = PipelineHelper(self.role_maker)
        pipeline_helper.update_startup_program(
            self.startup_program._pipeline_opt["startup_program"],
            self.inner_parallelism)

        self._transpile_main_program(loss, node_num, gpus_per_node)
        return optimize_ops, params_grads

    def _transpile_main_program(self, loss, node_num, gpus_per_node):
        self._insert_loss_grad_ops(loss, gpus_per_node, node_num)
        for ring_id in range(1, gpus_per_node + 1):
            self._insert_allreduce_ops(ring_id)

    def _insert_loss_grad_ops(self, loss, gpus_per_node, node_num):
        """
        In order to keep the learning rate consistent in different numbers of
        training workers, we scale the loss grad by the number of workers
        """
        block = self.main_program_list[gpus_per_node - 1][
            'program'].global_block()
        for idx, op in reversed(list(enumerate(block.ops))):
            if is_loss_grad_op(op):
                loss_grad_var = block.vars[op.output_arg_names[0]]
                block._insert_op(
                    idx + 1,
                    type='scale',
                    inputs={'X': loss_grad_var},
                    outputs={'Out': loss_grad_var},
                    attrs={
                        'scale': 1.0 / node_num,
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
                        type='c_sync_calc_stream',
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
                    idx + ring_id,
                    type='c_sync_comm_stream',
                    inputs={'X': grad},
                    outputs={'Out': grad},
                    attrs={'ring_id': ring_id,
                           OP_ROLE_KEY: OpRole.Backward})
            break
