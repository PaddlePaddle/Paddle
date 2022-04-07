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

import random
import numpy as np
import copy
import logging
from collections import defaultdict

import paddle
from paddle import fluid
from paddle.io import Dataset
from paddle.metric import Metric
from paddle.static import InputSpec
from paddle.fluid import core
from paddle.fluid import program_guard
from paddle.fluid.backward import append_backward
from paddle.fluid.framework import Operator
from paddle.fluid.framework import _current_expected_place as _get_device
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.distributed.passes import new_pass, PassContext
from paddle.distributed.utils import get_logger

from .mapper import mapping
from .cluster import Cluster
from .reshard import reshard
from .planner import Planner
from .completion import Completer
from .partitioner import Partitioner
from .dist_op import DistributedOperator
from .dist_saver import DistributedSaver
from .dist_loader import NonIterableGeneratorLoader
from .utils import make_data_unshard, set_grad_var_shape
from .utils import print_program_with_dist_attr, to_list
from .process_group import get_all_process_groups, get_world_process_group
from .dist_context import DistributedContext, get_default_distributed_context
from ...hapi.callbacks import config_callbacks, ModelCheckpoint
from ...hapi.model_summary import summary

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
        self.strategy = strategy

        self._executor = None
        self._orig_main_prog = fluid.default_main_program()
        self._orig_startup_prog = fluid.default_startup_program()
        self._orig_dist_context = get_default_distributed_context()
        self._serial_main_progs = {}
        self._serial_startup_progs = {}
        self._dist_main_progs = defaultdict(dict)  # dist main programs
        self._dist_startup_progs = defaultdict(dict)  # dist startup programs
        self._dist_contexts = {}
        self._pass_contexts = {}
        self._cur_rank = paddle.distributed.get_rank()
        self._logger = get_logger(logging.INFO)
        self._saver = DistributedSaver()
        self._feed_vars = {}
        self._fetch_vars = {}

    def prepare(self,
                optimizer=None,
                loss=None,
                metrics=None,
                mode='train',
                all_ranks=False):
        self._optimizer = optimizer
        self._loss = loss
        if loss and not isinstance(loss,
                                   paddle.nn.Layer) and not callable(loss):
            raise TypeError(
                "'loss' must be sub classes of `paddle.nn.Layer` or any callable function."
            )

        metrics = metrics or []
        for metric in to_list(metrics):
            assert isinstance(metric, Metric), \
                "{} is not sub class of Metric".format(
                    metric.__class__.__name__)
        self._metrics = to_list(metrics)

        self._mode = mode
        modes = ['train', 'eval', 'predict']
        for m in modes:
            self._build(m)  # build forward program
            self._plan(m)  # completion & planner
            self._parallel(m, all_ranks)  # parallel
            self._initialize(m)  # init comm and startup program

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
                assert len(losses) == 1
            if mode != 'predict' and self._metrics:
                for metric in self._metrics:
                    metrics.append(to_list(metric.compute(*(outputs + labels))))

        self._feed_vars[mode] = {"input": inputs, "label": labels}

        self._fetch_vars[mode] = {
            "output": outputs,
            "loss": losses,
            "metric": metrics
        }

        self._serial_main_progs[mode] = serial_main_prog
        self._serial_startup_progs[mode] = serial_startup_prog
        self._dist_contexts[mode] = DistributedContext(
            serial_main_prog, serial_startup_prog, self._dist_main_progs[mode],
            self._dist_startup_progs[mode])
        self._pass_contexts[mode] = PassContext()

    def _plan(self, mode):
        # Complete the distributed annotation
        serial_main_prog = self._serial_main_progs[mode]

        # NOTE: To adapt `High-order differential`. There are backward ops in forward completition,
        # and it need dependency of backward-forward ops in forward completition.
        defualt_dist_context = paddle.distributed.auto_parallel.dist_context.get_default_distributed_context(
        )
        self._dist_contexts[
            mode]._dist_op_context = defualt_dist_context.dist_op_context

        self._completer = Completer(self._dist_contexts[mode])
        self._completer.complete_forward_annotation(serial_main_prog)
        # TODO: add auto planner process
        # parse forward sub block
        self._dist_contexts[mode].block_state.parse_forward_blocks(
            serial_main_prog)

    def _parallel(self, mode, all_ranks=False):
        if not all_ranks:
            self._parallel_program(mode, self._cur_rank)
        else:
            world_process_group = get_world_process_group()
            for rank in world_process_group.ranks:
                self._parallel_program(mode, rank)

    def _initialize(self, mode):
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
        dist_startup_prog = self._dist_startup_progs[mode][self._cur_rank]
        self._executor.run(dist_startup_prog)

    def _parallel_program(self, mode, rank):
        serial_main_program = self._serial_main_progs[mode]
        serial_startup_program = self._serial_startup_progs[mode]
        dist_context = self._dist_contexts[mode]
        pass_context = self._pass_contexts[mode]
        if mode == "train" and self._optimizer:
            # Generate backward
            serial_loss = self._fetch_vars[mode]["loss"][0]
            params_grads = self._generate_backward(serial_main_program,
                                                   serial_startup_program,
                                                   serial_loss, dist_context)
            # Apply pre optimization passes
            self._apply_pre_optimization(
                serial_main_program, serial_startup_program, serial_loss,
                params_grads, dist_context, pass_context)
            # Do logical partition
            partitioner = Partitioner(dist_context, rank)
            dist_main_prog, dist_startup_prog, dist_params_grads = partitioner.partition(
                serial_main_program, serial_startup_program, params_grads)
            # Generate optimizer
            self._generate_optimizer(dist_main_prog, dist_startup_prog,
                                     dist_params_grads)
            # Do reshard process
            # NOTE: `set_grad_var_shape` has not adapted to `High-order differential`
            # set_grad_var_shape(dist_main_prog, dist_context)
            make_data_unshard(dist_main_prog, dist_startup_prog, dist_context)
            reshard(dist_main_prog, dist_startup_prog, rank, dist_context,
                    dist_params_grads)
            # Apply post optimization passes
            self._apply_post_optimization(dist_main_prog, dist_startup_prog,
                                          rank, dist_params_grads, dist_context,
                                          pass_context)
        else:
            # Do logical partition
            partitioner = Partitioner(dist_context, rank)
            dist_main_prog, dist_startup_prog, dist_params_grads = partitioner.partition(
                serial_main_program, serial_startup_program, [])
            # Do reshard process
            # TODO: support micro batch size
            make_data_unshard(dist_main_prog, dist_startup_prog, dist_context)
            reshard(dist_main_prog, dist_startup_prog, rank, dist_context, [],
                    1)

        # clone program for test
        if mode != 'train':
            dist_main_prog = dist_main_prog.clone(for_test=True)
            dist_startup_prog = dist_startup_prog.clone(for_test=True)

        self._dist_main_progs[mode][rank] = dist_main_prog
        self._dist_startup_progs[mode][rank] = dist_startup_prog

    def _generate_backward(self, main_program, startup_program, loss,
                           dist_context):
        with program_guard(main_program, startup_program):
            params_grads = append_backward(
                loss, distop_context=dist_context.dist_op_context)
        self._completer.complete_backward_annotation(main_program)
        dist_context.block_state.parse_backward_blocks(main_program)
        return params_grads

    def _generate_optimizer(self, main_program, startup_program, params_grads):
        with program_guard(main_program, startup_program):
            optimizer_ops = copy.deepcopy(self._optimizer).apply_gradients(
                params_grads)
        self._completer.complete_update_annotation(main_program)
        return optimizer_ops

    def _apply_pre_optimization(self, main_program, startup_program, loss,
                                params_grads, dist_context, pass_context):
        # apply amp pass
        if self.strategy.amp:
            config = copy.deepcopy(self.strategy.amp_configs)
            config["dist_context"] = dist_context
            config["params_grads"] = params_grads
            config["loss"] = loss
            config["inputs"] = self._feed_vars[self._mode][
                'input'] + self._feed_vars[self._mode]['label']
            auto_parallel_amp_pass = new_pass("auto_parallel_amp", config)
            auto_parallel_amp_pass.apply([main_program], [startup_program],
                                         pass_context)

        # apply recompute pass
        if self.strategy.recompute:
            config = copy.deepcopy(self.strategy.recompute_configs)
            config["dist_context"] = dist_context
            config["no_grad_set"] = None
            config["loss"] = loss
            auto_parallel_recompute_pass = new_pass("auto_parallel_recompute",
                                                    config)
            auto_parallel_recompute_pass.apply([main_program],
                                               [startup_program], pass_context)

    def _apply_post_optimization(self, main_program, startup_program, rank,
                                 params_grads, dist_context, pass_context):
        if self.strategy.sharding:
            config = copy.deepcopy(self.strategy.sharding_configs)
            config["dist_context"] = dist_context
            config["params_grads"] = params_grads
            config["global_rank"] = rank
            auto_parallel_sharding_pass = new_pass("auto_parallel_sharding",
                                                   config)
            auto_parallel_sharding_pass.apply([main_program],
                                              [startup_program], pass_context)

        if self.strategy.gradient_merge:
            config = copy.deepcopy(self.strategy.gradient_merge_configs)
            config["dist_context"] = dist_context
            config["params_grads"] = params_grads
            auto_parallel_gradient_merge_pass = new_pass(
                "auto_parallel_gradient_merge_pass", config)
            auto_parallel_gradient_merge_pass.apply(
                [main_program], [startup_program], pass_context)

    def fit(self,
            train_data,
            eval_data=None,
            batch_size=1,
            epochs=1,
            eval_freq=1,
            log_freq=10,
            save_freq=1,
            save_dir=None,
            callbacks=None,
            verbose=2,
            steps_per_epoch=None,
            sample_generator=True):
        # TODO: callbacks
        # TODO: evaluate after training
        self._mode = 'train'
        # assert isinstance(train_data, Dataset)
        train_dataloader = self._create_dataloader(
            train_data, batch_size, epochs, steps_per_epoch, sample_generator)

        cbks = config_callbacks(
            callbacks,
            model=self,
            epochs=epochs,
            steps=train_dataloader.steps,
            log_freq=log_freq,
            save_freq=save_freq,
            save_dir=save_dir,
            verbose=verbose,
            metrics=self.metrics_name())

        cbks.on_begin('train')
        outputs = []
        for epoch in range(epochs):
            cbks.on_epoch_begin(epoch)
            for step, data in enumerate(train_dataloader):
                logs, loss = self._train_step(data)
                outputs.append(loss)
                train_logs = {
                    "train_" + name: val
                    for name, val in logs.items()
                }
                self._logger.info(train_logs)
        return outputs

    def predict(self,
                test_data,
                batch_size=1,
                use_program_cache=False,
                return_numpy=True):
        self._mode = 'predict'
        # TODO: need check dataset
        test_dataloader = self._create_dataloader(test_data, batch_size)

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

    def _train_step(self, data):
        logs = {}
        dist_main_prog = self._dist_main_progs[self._mode][self._cur_rank]
        loss = self._fetch_vars[self._mode]["loss"][0]
        fetch_var = self._fetch_vars[self._mode]["loss"] + self._fetch_vars[
            self._mode]["output"]
        if loss.name not in dist_main_prog.global_block().vars:
            res = self._executor.run(dist_main_prog)
            logs["loss"] = None
        else:
            res = self._executor.run(dist_main_prog, fetch_list=fetch_var)
            # for idx, var in enumerate(fetch_var):
            #     logs[var.name] = res[idx]
            logs[fetch_var[0].name] = res[0]
        return logs, res

    def _predict_step(self, data, use_program_cache=False, return_numpy=True):
        logs = {}
        dist_main_prog = self._dist_main_progs[self._mode][self._cur_rank]
        fetch_var = self._fetch_vars[self._mode]["output"]
        if fetch_var[0].name not in dist_main_prog.global_block().vars:
            res = self._executor.run(dist_main_prog,
                                     use_program_cache=use_program_cache)
            logs["pred"] = None
        else:
            res = self._executor.run(dist_main_prog,
                                     fetch_list=fetch_var,
                                     use_program_cache=use_program_cache,
                                     return_numpy=return_numpy)
            for idx, var in enumerate(fetch_var):
                logs[var.name] = res[idx]
        return logs, res

    def _create_dataloader(self,
                           dataset,
                           batch_size,
                           epochs=1,
                           steps_per_epoch=None,
                           sample_generator=True):
        feed_list = self._feed_vars[self._mode]["input"] + self._feed_vars[
            self._mode]["label"]
        dist_main_prog = self._dist_main_progs[self._mode][self._cur_rank]
        dist_startup_prog = self._dist_startup_progs[self._mode][self._cur_rank]
        dist_context = self._dist_contexts[self._mode]
        dist_main_block = dist_main_prog.global_block()
        serial_main_prog = self._serial_main_progs[self._mode]
        serial_main_block = serial_main_prog.global_block()
        op_size = len(dist_main_block.ops)
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
                if in_name == "lod_tensor_blocking_queue_0":
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

    def save(self, path, training=True, mode=None):
        if not mode:
            mode = self._mode

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
            feed_vars = self._feed_vars[mode]['input']
            fetch_vars = self._fetch_vars[mode]['output']
            dist_main_prog = self._dist_main_progs[mode][self._cur_rank]
            self._saver.save_inference_model(
                path,
                feed_vars,
                fetch_vars,
                self._executor,
                program=dist_main_prog)

    def load(self, path, strict=True, load_optimizer=True, mode=None):
        if not mode:
            mode = self._mode
        assert mode, "Please set the 'mode' you want to load."

        dist_main_prog = self._dist_main_progs[mode][self._cur_rank]
        dist_context = self._dist_contexts[mode]
        self._saver.load(path, dist_main_prog, dist_context, strict,
                         load_optimizer)

    def metrics_name(self):
        metrics_name = []
        for metric in self._metrics:
            metrics_name.append(metric.name)
        return metrics_name

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
        return self._dist_main_progs[self._mode][self._cur_rank]

    @property
    def startup_program(self):
        return self._dist_startup_progs[self._mode][self._cur_rank]

    @property
    def dist_context(self):
        return self._dist_contexts[self._mode]

    @property
    def serial_main_program(self):
        return self._serial_main_progs[self._mode]

    @property
    def serial_startup_program(self):
        return self._serial_startup_progs[self._mode]
