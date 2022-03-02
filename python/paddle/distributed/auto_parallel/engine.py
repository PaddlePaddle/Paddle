# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import logging
from collections import defaultdict

import paddle
from paddle import fluid
from paddle.io import Dataset
from paddle.fluid.backward import append_backward
import paddle.fluid.core as core
from paddle.static import InputSpec
from paddle.fluid import program_guard
from paddle.fluid.framework import Operator
from paddle.fluid.framework import _current_expected_place as _get_device
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.distributed.passes import new_pass, PassContext
from paddle.distributed.utils import get_logger

from .dist_loader import NonIterableGeneratorLoader
from .dist_op import DistributedOperator
from .dist_tensor import DistributedTensor
from .dist_context import DistributedContext
from .dist_context import get_default_distributed_context
from .dist_context import set_default_distributed_context
from .process_group import get_all_process_groups
from .process_group import get_process_group
from .process_group import get_world_process_group
from .process_group import _g_process_group_map, ProcessGroup
from .completion import Completer
from .partitioner import Partitioner
from .reshard import reshard, HAS_SENT, HAS_RECV, HAS_ALLGATHER
from .cluster import Cluster
from .mapper import mapping
from .planner import Planner
from .utils import make_data_unshard
from .utils import set_grad_var_shape
from .utils import print_program_with_dist_attr
from .utils import SerialProgramInfo

paddle.enable_static()


def to_list(value):
    if value is None:
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class Engine:
    def __init__(self, model=None, data_spec=None, cluster=None, strategy=None):
        self.model = model
        self.data_spec = data_spec
        self.cluster = cluster
        self.strategy = strategy
        self._executor = None
        self._orig_main_prog = fluid.default_main_program()
        self._orig_startup_prog = fluid.default_startup_program()
        self._serial_main_progs = {}
        self._serial_startup_progs = {}
        self._dist_main_progs = defaultdict(dict)
        self._dist_startup_progs = defaultdict(dict)
        self._orig_dist_context = get_default_distributed_context()
        self._dist_contexts = {}
        self._pass_contexts = {}
        self._cur_rank = paddle.distributed.get_rank()
        self._logger = get_logger(logging.INFO)

    def prepare(self,
                optimizer=None,
                loss=None,
                metrics=None,
                mode="train",
                all_ranks=False):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.mode = mode
        self._build()
        self._plan()
        if not all_ranks:
            self._parallel(self._cur_rank)
        else:
            world_process_group = get_world_process_group()
            all_ranks = world_process_group.ranks
            for rank in all_ranks:
                self._parallel(rank)
        place = _get_device()
        if isinstance(place, fluid.CUDAPlace):
            self._place = fluid.CUDAPlace(ParallelEnv().dev_id)
        if self._executor is None:
            self._executor = fluid.Executor(place)

    def _build(self):
        serial_main_prog = self._serial_main_progs.get(self.mode, None)
        if serial_main_prog is not None:
            return

        serial_main_prog = self._orig_main_prog.clone()
        serial_startup_prog = self._orig_startup_prog.clone()
        with fluid.program_guard(serial_main_prog, serial_startup_prog):
            inputs_spec = self.data_spec[0]
            labels_spec = self.data_spec[1]
            inputs = [s._create_feed_layer() for s in to_list(inputs_spec)]
            labels = [s._create_feed_layer() for s in to_list(labels_spec)]
            self._input_vars = inputs
            self._label_vars = labels
            feed_list = self._input_vars + self._label_vars
            outputs = to_list(self.model(*inputs))
            if self.mode != "predict" and self.loss:
                loss = self.loss(*(outputs + labels))
                self._loss_var = loss

        self._serial_main_progs[self.mode] = serial_main_prog
        self._serial_startup_progs[self.mode] = serial_startup_prog
        self._dist_contexts[self.mode] = DistributedContext(
            serial_main_prog, serial_startup_prog,
            self._dist_main_progs[self.mode],
            self._dist_startup_progs[self.mode])
        self._pass_contexts[self.mode] = PassContext()

    def _plan(self):
        # Complete the distributed annotation
        serial_main_prog = self._serial_main_progs[self.mode]
        self._completer = Completer(self._dist_contexts[self.mode])
        self._completer.complete_forward_annotation(serial_main_prog)
        # TODO: add auto planner process
        # parse forward sub block
        self._dist_contexts[self.mode].block_state.parse_forward_blocks(
            serial_main_prog)

    def _parallel(self, rank):
        serial_main_program = self._serial_main_progs[self.mode]
        serial_startup_program = self._serial_startup_progs[self.mode]
        dist_context = self._dist_contexts[self.mode]
        if self.mode != "predict" and self.loss:
            # Generate backward
            serial_loss = self._loss_var
            params_grads = self._generate_backward(
                serial_main_program, serial_startup_program, serial_loss)
            # Apply pre optimization passes
            self._apply_pre_optimization(serial_main_program,
                                         serial_startup_program, serial_loss,
                                         params_grads)
            # Do logical partition
            partitioner = Partitioner(dist_context, rank)
            dist_main_prog, dist_startup_prog, dist_params_grads = partitioner.partition(
                serial_main_program, serial_startup_program, params_grads)
            # Generate optimizer
            self._generate_optimizer(dist_main_prog, dist_startup_prog,
                                     dist_params_grads)
            # Do reshard process
            set_grad_var_shape(dist_main_prog, dist_context)
            make_data_unshard(dist_main_prog, dist_startup_prog, dist_context)
            reshard(dist_main_prog, dist_startup_prog, rank, dist_context,
                    dist_params_grads)
            # Apply post optimization passes
            self._apply_post_optimization(dist_main_prog, dist_startup_prog,
                                          rank, dist_params_grads)
        self._dist_main_progs[self.mode][rank] = dist_main_prog
        self._dist_startup_progs[self.mode][rank] = dist_startup_prog

    def _generate_backward(self, main_program, startup_program, loss):
        with program_guard(main_program, startup_program):
            params_grads = append_backward(
                loss,
                distop_context=self._dist_contexts[self.mode].dist_op_context)
        self._completer.complete_backward_annotation(main_program)
        self._dist_contexts[self.mode].block_state.parse_backward_blocks(
            main_program)
        return params_grads

    def _generate_optimizer(self, main_program, startup_program, params_grads):
        with program_guard(main_program, startup_program):
            optimizer_ops = copy.deepcopy(self.optimizer).apply_gradients(
                params_grads)
        self._completer.complete_update_annotation(main_program)
        return optimizer_ops

    def _apply_pre_optimization(self, main_program, startup_program, loss,
                                params_grads):
        # apply amp pass
        if self.strategy.amp:
            config = copy.deepcopy(self.strategy.amp_configs)
            config["dist_context"] = self._dist_contexts[self.mode]
            config["params_grads"] = params_grads
            config["loss"] = loss
            auto_parallel_amp_pass = new_pass("auto_parallel_amp", config)
            auto_parallel_amp_pass.apply([main_program], [startup_program],
                                         self._pass_contexts[self.mode])

        # apply recompute pass
        if self.strategy.recompute:
            config = copy.deepcopy(self.strategy.recompute_configs)
            config["dist_context"] = self._dist_contexts[self.mode]
            config["no_grad_set"] = None
            config["loss"] = loss
            auto_parallel_recompute_pass = new_pass("auto_parallel_recompute",
                                                    config)
            auto_parallel_recompute_pass.apply([main_program],
                                               [startup_program],
                                               self._pass_contexts[self.mode])

    def _apply_post_optimization(self, main_program, startup_program, rank,
                                 params_grads):
        if self.strategy.sharding:
            config = copy.deepcopy(self.strategy.sharding_configs)
            config["dist_context"] = self._dist_contexts[self.mode]
            config["params_grads"] = params_grads
            config["global_rank"] = rank
            auto_parallel_sharding_pass = new_pass("auto_parallel_sharding",
                                                   config)
            auto_parallel_sharding_pass.apply([main_program],
                                              [startup_program],
                                              self._pass_contexts[self.mode])

        if self.strategy.gradient_merge:
            config = copy.deepcopy(self.strategy.gradient_merge_configs)
            config["dist_context"] = self._dist_contexts[self.mode]
            config["params_grads"] = params_grads
            auto_parallel_gradient_merge_pass = new_pass(
                "auto_parallel_gradient_merge_pass", config)
            auto_parallel_gradient_merge_pass.apply(
                [main_program], [startup_program],
                self._pass_contexts[self.mode])

    def fit(self, train_data, batch_size=1, epochs=1, steps_per_epoch=1000):
        assert isinstance(train_data, Dataset)
        assert steps_per_epoch is not None
        train_dataloader = self._create_dataloader(train_data, batch_size,
                                                   epochs, steps_per_epoch)
        self._init_communication()
        dist_startup_prog = self._dist_startup_progs["train"][self._cur_rank]
        self._executor.run(dist_startup_prog)
        for epoch in range(epochs):
            # train_dataloader.start()
            # for step in range(steps_per_epoch):
            #     logs = self.train_step(None)
            #     self._logger.info(logs)
            # train_dataloader.reset()
            for step, data in enumerate(train_dataloader):
                logs = self._train_step(data)
                train_logs = {
                    "train_" + name: val
                    for name, val in logs.items()
                }
                self._logger.info(logs)

    def _train_step(self, data):
        logs = {}
        dist_main_prog = self._dist_main_progs["train"][self._cur_rank]
        if self._loss_var.name not in dist_main_prog.global_block().vars:
            loss = self._executor.run(dist_main_prog)
            logs["loss"] = None
        else:
            fetch_list = self._loss_var
            loss = self._executor.run(dist_main_prog, fetch_list=fetch_list)
            logs["loss"] = loss
        return logs

    def _create_dataloader(self, dataset, batch_size, epochs, steps_per_epoch):
        feed_list = self._input_vars + self._label_vars
        dist_main_prog = self._dist_main_progs[self.mode][self._cur_rank]
        dist_startup_prog = self._dist_startup_progs[self.mode][self._cur_rank]
        dist_context = self._dist_contexts[self.mode]
        dist_main_block = dist_main_prog.global_block()
        op_size = len(dist_main_block.ops)
        places = paddle.static.cuda_places()
        with fluid.program_guard(dist_main_prog, dist_startup_prog):
            dataloader = NonIterableGeneratorLoader(
                dataset, feed_list, places, batch_size, epochs, steps_per_epoch)
        new_op_size = len(dist_main_block.ops)
        for idx in range(new_op_size - 1, op_size - 1, -1):
            op = dist_main_block.ops[new_op_size - 1]
            new_op_desc = dist_main_block.desc._prepend_op()
            new_op_desc.copy_from(op.desc)
            new_op = Operator(
                dist_main_block, new_op_desc, type=new_op_desc.type())
            dist_main_block.ops.insert(0, new_op)
            dist_op = DistributedOperator(new_op)
            dist_context.add_dist_op_for_program(dist_op)
        for _ in range(new_op_size - op_size):
            dist_main_block._remove_op(new_op_size, sync=False)
        dist_main_block._sync_with_cpp()
        return dataloader

    def _init_communication(self):
        # Traverse different rank programs and traverse each op of them,
        # instantiate communication by process_mapping.
        all_process_groups = get_all_process_groups()
        for process_group in all_process_groups:
            if self._cur_rank not in process_group.ranks:
                continue
            process_group.instantiate()

    # def save(self, path, training=True):
    #     pass

    # def load(self, path, strict=True, load_optimizer=True):
    #     pass
