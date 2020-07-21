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

import paddle
from paddle.fluid.framework import core
from paddle.fluid import compiler
from .meta_optimizer_base import MetaOptimizerBase
from ..base.private_helper_function import wait_server_ready


def get_build_strategy(dist_strategy):
    build_strategy = paddle.BuildStrategy()
    build_strategy.enable_sequential_execution = \
        dist_strategy.sequential_execution
    build_strategy.remove_unnecessary_lock = True
    build_strategy.fuse_elewise_add_act_ops = \
        dist_strategy.fuse_elewise_add_act_ops
    build_strategy.fuse_bn_act_ops = \
        dist_strategy.fuse_bn_act_ops
    build_strategy.enable_auto_fusion = \
        dist_strategy.enable_auto_fusion
    build_strategy.fuse_relu_depthwise_conv = \
        dist_strategy.fuse_relu_depthwise_conv
    build_strategy.fuse_broadcast_ops = \
        dist_strategy.fuse_broadcast_ops
    build_strategy.sync_batch_norm = \
        dist_strategy.sync_batch_norm
    return build_strategy


def get_execution_strategy(dist_strategy):
    execution_strategy = paddle.ExecutionStrategy()
    execution_strategy.num_threads = \
        dist_strategy.num_threads
    execution_strategy.num_iteration_per_drop_scope = \
        dist_strategy.num_iteration_per_drop_scope
    execution_strategy.num_iteration_per_run = \
        dist_strategy.num_iteration_per_run
    execution_strategy.use_thread_barrier = \
        dist_strategy.use_thread_barrier
    return execution_strategy


class GraphExecutionOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(GraphExecutionOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = []

    def _is_graph_out(self):
        return True

    def _can_apply(self):
        """
        Basically, this is PE, and almost all programs can be executed here
        """
        return True

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        pass

    # should fix the variable 
    def _setup_nccl_op(self, startup_program, main_program):
        trainer_endpoints = self.role_maker.get_trainer_endpoints()
        trainers = trainer_endpoints
        trainer_id = self.role_maker.worker_index()
        current_endpoint = self.role_maker.get_trainer_endpoints()[trainer_id]
        trainer_endpoints_env = ",".join(trainer_endpoints)
        trainers_num = self.role_maker.worker_num()
        trainer_endpoints.remove(current_endpoint)
        if trainer_id == 0:
            wait_server_ready(trainer_endpoints)
        nccl_id_var = startup_program.global_block().create_var(
            name="NCCLID", persistable=True, type=core.VarDesc.VarType.RAW)
        for i in range(1, self.user_defined_strategy.nccl_comm_num):
            startup_program.global_block().create_var(
                name="NCCLID_{}".format(i),
                persistable=True,
                type=core.VarDesc.VarType.RAW)

        if self.user_defined_strategy.hierachical_allreduce:
            for i in range(0, self.user_defined_strategy.nccl_comm_num):
                startup_program.global_block().create_var(
                    name="Hierarchical_inter_NCCLID_{}".format(i),
                    persistable=True,
                    type=core.VarDesc.VarType.RAW)
                startup_program.global_block().create_var(
                    name="Hierarchical_exter_NCCLID_{}".format(i),
                    persistable=True,
                    type=core.VarDesc.VarType.RAW)

        startup_program.global_block().append_op(
            type="gen_nccl_id",
            inputs={},
            outputs={"NCCLID": nccl_id_var},
            attrs={
                "trainers": trainers,
                "trainer_id": trainer_id,
                "nccl_comm_num": self.user_defined_strategy.nccl_comm_num,
                "use_hierarchical_allreduce":
                self.user_defined_strategy.hierachical_allreduce,
                "hierarchical_allreduce_inter_ranks":
                self.user_defined_strategy.hierachical_allreduce_inter_ranks
            })

    def _try_to_compile(self, startup_program, main_program, loss):
        build_strategy = get_build_strategy(self.user_defined_strategy)
        exe_strategy = get_execution_strategy(self.user_defined_strategy)
        node_num = self.role_maker.worker_num()
        if self.role_maker._is_collective:
            assert node_num >= 1, "nccl2 node_num must >= 1, now:{}" % node_num

        if node_num <= 1:
            # local mode
            if self.user_defined_strategy.nccl_comm_num > 1:
                logging.warn("set nccl_comm_num=1 since you only have 1 node.")
            self.user_defined_strategy.nccl_comm_num = 1

            if self.user_defined_strategy.hierachical_allreduce:
                logging.warn(
                    "set hierachical_allreduce=False since you only have 1 node."
                )
            self.user_defined_strategy.hierachical_allreduce = False

        sync_allreduce = self.user_defined_strategy.sync_nccl_allreduce
        if sync_allreduce:
            exe_strategy.num_threads = self.user_defined_strategy.nccl_comm_num + 1
            if self.user_defined_strategy.hierachical_allreduce:
                exe_strategy.num_threads = 2 * self.user_defined_strategy.nccl_comm_num + 1
            if exe_strategy.num_threads > 4:
                logging.warn(
                    "if you use hierachical_allreduce or "
                    "with multi nccl comm, please export FLAGS_sync_nccl_allreduce = 0"
                )

        # TODO(guru4elephant): should be an independent optimizer
        sync_batch_norm = self.user_defined_strategy.sync_batch_norm
        if sync_batch_norm:
            self.user_defined_strategy.nccl_comm_num = 1
            self.user_defined_strategy.hierachical_allreduce = False
            exe_strategy.num_threads = 1
            logging.warn(
                "use sync_batch_norm will hang when set num_threads > 1, so "
                "set num_threads=1, nccl_comm_num=1, hierachical_allreduce=False."
            )

        # TODO(guru4elephant): should be an independent optimizer
        self._setup_nccl_op(startup_program, main_program)

        build_strategy.num_trainers = self.role_maker.worker_num()
        build_strategy.trainer_id = self.role_maker.worker_index()
        build_strategy.trainers_endpoints = self.role_maker.get_trainer_endpoints(
        )
        build_strategy.enable_backward_optimizer_op_deps = True

        self._compiled_program = compiler.CompiledProgram(main_program)

        self._compiled_program.with_data_parallel(
            loss_name=loss.name,
            build_strategy=build_strategy,
            exec_strategy=exe_strategy,
            share_vars_from=None)

        return self._compiled_program

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        if startup_program == None:
            startup_program = paddle.default_startup_program()
        compiled_program = self._try_to_compile(startup_program,
                                                loss.block.program, loss)
        loss.block.program.graph = compiled_program

        # just return self.optimizer_ops and self.param_grads
        return None, None
