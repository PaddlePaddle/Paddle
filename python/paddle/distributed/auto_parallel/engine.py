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
import paddle.distributed.auto_parallel as auto

from paddle import fluid
from paddle.io import Dataset
from paddle.metric import Metric
from paddle.static import InputSpec
from paddle.fluid import core
from paddle.fluid import program_guard
from paddle.fluid.layers.utils import flatten
from paddle.fluid.executor import global_scope
from paddle.fluid.backward import append_backward
from paddle.fluid.framework import Operator
from paddle.fluid.framework import _current_expected_place as _get_device
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.distributed import fleet
from paddle.distributed.utils import get_logger
from paddle.distributed.passes import new_pass, PassContext

from .cluster import Cluster, get_default_cluster
from .planner_v2 import Planner
from .parallelizer_v2 import Parallelizer
from .dist_op import DistributedOperator
from .dist_saver import DistributedSaver
from .dist_loader import NonIterableGeneratorLoader
from .utils import make_data_unshard, set_grad_var_shape
from .utils import print_program_with_dist_attr, to_list
from .process_group import get_all_process_groups, get_world_process_group
from .dist_context import DistributedContext, get_default_distributed_context

paddle.enable_static()


class Engine:
    def __init__(self,
                 model=None,
                 inputs_spec=None,
                 labels_spec=None,
                 cluster=None,
                 strategy=None):
        self.model = model
        self.inputs_spec = self._validate_spec(inputs_spec)
        self.labels_spec = self._validate_spec(labels_spec)
        self.cluster = cluster
        if self.cluster is None:
            self.cluster = get_default_cluster()
        self.strategy = strategy
        if self.strategy is None:
            self.strategy = fleet.DistributedStrategy()

        self._executor = None
        self._cur_rank = paddle.distributed.get_rank()
        self._nranks = paddle.distributed.get_world_size()
        self._saver = DistributedSaver()
        self._logger = get_logger(logging.INFO)

        self._default_strategy = None
        self._orig_main_prog = fluid.default_main_program()
        self._orig_startup_prog = fluid.default_startup_program()
        self._orig_dist_context = get_default_distributed_context()
        self._serial_main_progs = {}
        self._serial_startup_progs = {}
        self._dist_main_progs = defaultdict(dict)  # dist main programs
        self._dist_startup_progs = defaultdict(dict)  # dist startup programs
        self._dist_contexts = {}
        self._feed_vars = {}
        self._fetch_vars = {}

    def prepare(self,
                optimizer=None,
                loss=None,
                metrics=None,
                mode='train',
                all_ranks=False):
        self._optimizer = optimizer
        # TODO: check loss type
        self._loss = loss
        self._metrics = to_list(metrics)
        self._mode = mode
        # Build forward program
        self._build(mode)
        # Do the planning process
        planner = Planner(mode, self._dist_contexts[mode])
        planner.plan()
        # Parallelize program based on the planner's results
        # For now, the completer has to be passed to the planner,
        # because we may use it to complete the annotation of the backwarkward and update.
        parallelizer = Parallelizer(mode, planner.completer,
                                    self._dist_contexts[mode])
        if not all_ranks:
            parallelizer.parallel(self._cur_rank)
        else:
            parallelizer.parallel_all()
        # Get the distributed main programs and startup programs
        self._dist_main_progs[mode] = self._dist_contexts[
            mode].dist_main_programs
        self._dist_startup_progs[mode] = self._dist_contexts[
            mode].dist_startup_programs
        # Init comm and startup program
        self._initialize(mode)

    def _build(self, mode):
        serial_main_prog = self._serial_main_progs.get(mode, None)
        if serial_main_prog is not None:
            return

        losses = []
        metrics = []
        serial_main_prog = self._orig_main_prog.clone()
        serial_startup_prog = self._orig_startup_prog.clone()
        with fluid.program_guard(serial_main_prog, serial_startup_prog):
            inputs_spec = self.inputs_spec
            labels_spec = self.labels_spec if self.labels_spec else []
            inputs = [s._create_feed_layer() for s in inputs_spec]
            labels = [s._create_feed_layer() for s in labels_spec]
            outputs = to_list(self.model(*inputs))
            if mode != "predict" and self._loss:
                losses = to_list(self._loss(*(outputs + labels)))

        default_ctx = get_default_distributed_context()
        if not default_ctx.has_annotation or self._default_strategy:
            inputs = [self._set_data_parallel(var) for var in inputs]
            labels = [self._set_data_parallel(var) for var in labels]

        self._feed_vars[mode] = {"inputs": inputs, "labels": labels}

        self._fetch_vars[mode] = {
            "outputs": flatten(outputs),
            "loss": losses,
            "metrics": metrics
        }

        self._serial_main_progs[mode] = serial_main_prog
        self._serial_startup_progs[mode] = serial_startup_prog
        self._dist_contexts[mode] = DistributedContext(
            self._serial_main_progs[mode], self._serial_startup_progs[mode],
            self._optimizer, losses, self._feed_vars[mode],
            self._fetch_vars[mode], self.cluster, self.strategy)

    def _initialize(self, mode):
        if self._nranks > 1:
            # Traverse different rank programs and traverse each op of them,
            # instantiate communication by process_mapping.
            all_process_groups = get_all_process_groups()
            for process_group in all_process_groups:
                if self._cur_rank not in process_group.ranks:
                    continue
                process_group.instantiate()

        # initialize
        self._place = _get_device()
        if isinstance(self._place, fluid.CUDAPlace):
            self._place = fluid.CUDAPlace(ParallelEnv().dev_id)
        if self._executor is None:
            self._executor = paddle.static.Executor(self._place)
            uninitialized = []
            dist_startup_prog = self._dist_startup_progs[mode][self._cur_rank]
            for var in dist_startup_prog.list_vars():
                scope_var = global_scope().find_var(var.name)
                if scope_var and scope_var.get_tensor()._is_initialized():
                    continue
                uninitialized.append(var)
            if uninitialized:
                prune_startup_prog = dist_startup_prog._prune(uninitialized)
                self._executor.run(prune_startup_prog)

    def fit(self,
            train_data,
            batch_size=1,
            epochs=1,
            steps_per_epoch=None,
            use_program_cache=False,
            return_numpy=True,
            sample_generator=True):
        # TODO: callbacks
        # TODO: evaluate after training
        self.mode = 'train'
        assert self.mode in self._dist_main_progs, "train model is not ready, please call `engine.prepare(mode='train')` first."
        train_dataloader = self._create_dataloader(
            train_data, batch_size, epochs, steps_per_epoch, sample_generator)

        outputs = []
        for epoch in range(epochs):
            for step, data in enumerate(train_dataloader):
                logs, loss = self._train_step(data, use_program_cache,
                                              return_numpy)
                outputs.append(loss)
                train_logs = {
                    "train_" + name: val
                    for name, val in logs.items()
                }
                self._logger.info(train_logs)
        return outputs

    def evaluate(self,
                 eval_data,
                 batch_size=1,
                 use_program_cache=False,
                 return_numpy=True,
                 sample_generator=True):
        self.mode = 'eval'
        assert self.mode in self._dist_main_progs, "eval model is not ready, please call `engine.prepare(mode='eval')` first."
        eval_dataloader = self._create_dataloader(
            eval_data, batch_size, sample_generator=sample_generator)

        outputs = []
        for step, data in enumerate(eval_dataloader):
            logs, outs = self._eval_step(data, use_program_cache, return_numpy)
            outputs.append(outs)
            predict_logs = {"eval_" + name: val for name, val in logs.items()}
            self._logger.info(predict_logs)
        return outputs

    def predict(self,
                test_data,
                batch_size=1,
                use_program_cache=False,
                return_numpy=True,
                sample_generator=True):
        self.mode = 'predict'
        assert self.mode in self._dist_main_progs, "predict model is not ready, please call `engine.prepare(mode='predict')` first."
        test_dataloader = self._create_dataloader(
            test_data, batch_size, sample_generator=sample_generator)

        outputs = []
        for step, data in enumerate(test_dataloader):
            logs, outs = self._predict_step(data, use_program_cache,
                                            return_numpy)
            outputs.append(outs)
            predict_logs = {
                "predict_" + name: val
                for name, val in logs.items()
            }
            self._logger.info(predict_logs)
        return outputs

    def _train_step(self, data, use_program_cache=False, return_numpy=True):
        logs = {}
        dist_main_prog = self._dist_main_progs[self.mode][self._cur_rank]
        print_program_with_dist_attr(dist_main_prog,
                                     self._dist_contexts[self.mode])
        fetch_var = self._fetch_vars[self.mode]["loss"][0]
        if fetch_var.name not in dist_main_prog.global_block().vars:
            loss = self._executor.run(dist_main_prog,
                                      use_program_cache=use_program_cache)
            logs["loss"] = None
        else:
            loss = self._executor.run(dist_main_prog,
                                      fetch_list=to_list(fetch_var),
                                      use_program_cache=use_program_cache,
                                      return_numpy=return_numpy)
            logs["loss"] = loss
        return logs, loss

    def _eval_step(self, data, use_program_cache=False, return_numpy=True):
        logs = {}
        dist_main_prog = self._dist_main_progs[self.mode][self._cur_rank]
        fetch_var = self._fetch_vars[self.mode]["loss"][0]

        if fetch_var.name not in dist_main_prog.global_block().vars:
            outs = self._executor.run(dist_main_prog,
                                      use_program_cache=use_program_cache)
            logs["loss"] = outs
        else:
            outs = self._executor.run(dist_main_prog,
                                      fetch_list=fetch_var,
                                      use_program_cache=use_program_cache,
                                      return_numpy=return_numpy)
            logs["loss"] = outs
        return logs, outs

    def _predict_step(self, data, use_program_cache=False, return_numpy=True):
        logs = {}
        dist_main_prog = self._dist_main_progs[self.mode][self._cur_rank]
        fetch_var = []
        for var in self._fetch_vars[self.mode]["outputs"]:
            if var.name in dist_main_prog.global_block().vars:
                fetch_var.append(var)

        if fetch_var is []:
            outs = self._executor.run(dist_main_prog,
                                      use_program_cache=use_program_cache)
            logs["pred"] = outs
        else:
            outs = self._executor.run(dist_main_prog,
                                      fetch_list=fetch_var,
                                      use_program_cache=use_program_cache,
                                      return_numpy=return_numpy)
            logs["pred"] = outs
        return logs, outs

    def _create_dataloader(self,
                           dataset,
                           batch_size,
                           epochs=1,
                           steps_per_epoch=None,
                           sample_generator=True):
        feed_list = self._feed_vars[self.mode]["inputs"] + self._feed_vars[
            self.mode]["labels"]
        dist_main_prog = self._dist_main_progs[self.mode][self._cur_rank]
        dist_startup_prog = self._dist_startup_progs[self.mode][self._cur_rank]
        dist_context = self._dist_contexts[self.mode]
        dist_main_block = dist_main_prog.global_block()
        serial_main_prog = self._serial_main_progs[self.mode]
        serial_main_block = serial_main_prog.global_block()
        op_size = len(dist_main_block.ops)
        if dist_main_block.ops[0].type == 'create_py_reader':
            op_size -= 3
            for _ in range(3):
                dist_main_block._remove_op(0, sync=False)
        places = paddle.static.cuda_places()
        with fluid.program_guard(dist_main_prog, dist_startup_prog):
            dataloader = NonIterableGeneratorLoader(
                dataset,
                feed_list,
                places,
                batch_size,
                epochs,
                steps_per_epoch,
                sample_generator=sample_generator)
        new_op_size = len(dist_main_block.ops)
        for _ in range(new_op_size - 1, op_size - 1, -1):
            op = dist_main_block.ops[new_op_size - 1]
            new_op_desc = dist_main_block.desc._prepend_op()
            new_op_desc.copy_from(op.desc)
            new_op = Operator(
                dist_main_block, new_op_desc, type=new_op_desc.type())
            dist_main_block.ops.insert(0, new_op)
            for in_name in new_op.input_arg_names:
                if "lod_tensor_blocking_queue" in in_name:
                    continue
                if in_name not in dist_main_block.vars:
                    in_var = serial_main_block._var_recursive(in_name)
                    dist_main_block._clone_variable(in_var, in_var.persistable)
            for out_name in new_op.output_arg_names:
                if out_name not in dist_main_block.vars:
                    out_var = serial_main_block._var_recursive(out_name)
                    dist_main_block._clone_variable(out_var,
                                                    out_var.persistable)
            dist_op = DistributedOperator(new_op)
            dist_context.add_dist_op_for_program(dist_op)
        for _ in range(new_op_size - op_size):
            dist_main_block._remove_op(new_op_size, sync=False)
        dist_main_block._sync_with_cpp()
        return dataloader

    def _validate_spec(self, specs):
        specs = to_list(specs)
        if specs is not None:
            for i, spec in enumerate(specs):
                assert isinstance(spec, InputSpec)
                if spec.name is None:
                    raise ValueError(
                        "Requires Input[{}].name != None, but receive `None` with {}."
                        .format(i, spec))
        return specs

    def _set_data_parallel(self, var):
        if self._nranks == 1:
            self._default_strategy = 'serial'
            auto.shard_tensor(
                var,
                dist_attr={
                    "process_mesh": [0],
                    "dims_mapping": [-1 for _ in range(len(var.shape))]
                })
        else:
            self._default_strategy = 'dp'
            auto.shard_tensor(
                var,
                dist_attr={
                    "process_mesh": list(range(self._nranks)),
                    "dims_mapping":
                    [0] + [-1 for _ in range(len(var.shape) - 1)]
                })

        return var

    def save(self, path, training=True, mode=None):
        if not mode:
            mode = self.mode

        if training:
            assert 'train' in self._serial_main_progs, "training model is not ready, please call `engine.prepare(mode='train')` first."
            serial_program = self._serial_main_progs["train"]
            dist_main_prog = self._dist_main_progs["train"][self._cur_rank]
            dist_context = self._dist_contexts["train"]
            self._saver.save(
                path,
                serial_program=serial_program,
                dist_main_program=dist_main_prog,
                dist_context=dist_context)
        else:
            assert mode, "Please set the 'mode' you want to save."
            feed_vars = self._feed_vars[mode]['inputs']
            fetch_vars = self._fetch_vars[mode]['outputs']
            dist_main_prog = self._dist_main_progs[mode][self._cur_rank]
            self._saver.save_inference_model(
                path,
                feed_vars,
                fetch_vars,
                self._executor,
                program=dist_main_prog)

    def load(self, path, strict=True, load_optimizer=True, mode=None):
        if not mode:
            mode = self.mode
        assert mode, "Please set the 'mode' you want to load."

        dist_main_prog = self._dist_main_progs[mode][self._cur_rank]
        dist_context = self._dist_contexts[mode]
        self._saver.load(path, dist_main_prog, dist_context, strict,
                         load_optimizer)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        self._mode = mode

    @property
    def metrics(self):
        return self._metrics

    @property
    def main_program(self):
        return self._dist_main_progs[self.mode][self._cur_rank]

    @property
    def startup_program(self):
        return self._dist_startup_progs[self.mode][self._cur_rank]

    @property
    def dist_context(self):
        return self._dist_contexts[self.mode]

    @property
    def serial_main_program(self):
        return self._serial_main_progs[self.mode]

    @property
    def serial_startup_program(self):
        return self._serial_startup_progs[self.mode]
