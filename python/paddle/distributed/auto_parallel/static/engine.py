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
import json
import logging
import numbers
import os
import random

import numpy as np

import paddle
import paddle.distributed.auto_parallel.static.utils as auto_utils
from paddle import static, utils
from paddle.distributed import fleet
from paddle.fluid.executor import _to_name_str
from paddle.framework import IrGraph
from paddle.framework import _current_expected_place as _get_device
from paddle.framework import core, in_dynamic_mode
from paddle.metric import Metric
from paddle.static import InputSpec, Operator, Variable, global_scope

from ...utils.log_utils import get_logger
from ..interface import CollectionNames, fetch, get_collection
from ..strategy import Strategy
from .callbacks import config_callbacks
from .cluster import Cluster, get_default_cluster
from .converter import Converter
from .cost.estimate_cost import get_cost_from_engine
from .dist_context import DistributedContext, get_default_distributed_context
from .dist_loader import (
    DistributedDataLoader,
    DistributedDataLoaderFromGenerator,
)
from .dist_op import DistributedOperator
from .dist_saver import DistributedSaver
from .helper import ProgramHelper
from .parallelizer_v2 import Parallelizer
from .planner_v2 import Planner
from .process_group import get_all_process_groups, new_process_group


class Engine:
    """
    An Engine object can provide the full power of auto parallel to users.
    With the help of it, users can easily obtain the abilities of the
    distributed training and inference. It also support the dynamic graph and
    static graph at the same time.

    Args:
        model (paddle.nn.Layer, optional): The model is an instance of
            paddle.nn.Layer.
        loss (Loss|Callable|None, optional): The loss can be a `paddle.nn.Layer`
            instance or any callable function taken the predicted values and
            ground truth values as input. It can be None when there is no loss.
            Default: None.
        optimizer (Optimizer|None, optional): The optimizer need to be set in training
            and should be None in eval and predict mode. Default: None.
        metrics (Metric|list[Metric]|None, optional): If metrics is set, all
            metrics will be calculated and output in train/eval mode. Default: None.
        cluster (Cluster|None, optional): The cluster represents the topology information
            about the used physical devices. Default: None. (Unused for now)
        strategy (Strategy|None, optional): The strategy is used to configure the
        parallelization and optimization behaviors. Default: None.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.vision.transforms as T
            from paddle.distributed.fleet import auto
            from paddle.vision.datasets import MNIST

            transform = T.Compose([
                T.Transpose(),
                T.Normalize([127.5], [127.5])
            ])
            train_dataset = MNIST(mode='train', transform=transform)
            valid_dataset = MNIST(mode='test', transform=transform)

            model = paddle.vision.models.LeNet()
            loss = paddle.nn.CrossEntropyLoss()
            optimizer = paddle.optimizer.Adam(
                learning_rate=0.001, parameters=model.parameters())
            metrics = paddle.metric.Accuracy(topk=(1, 2))

            engine = auto.Engine(model, loss, optimizer, metrics)
            # fit
            engine.fit(train_dataset,
                       epochs=2,
                       batch_size=64)
            # evaluate
            engine.evaluate(valid_dataset,
                            batch_size=64)
            # predict
            engine.predict(valid_dataset,
                           batch_size=64)
            # save
            engine.save("./my_model")
            # load
            engine.load("./my_model")

    """

    def __init__(
        self,
        model=None,
        loss=None,
        optimizer=None,
        metrics=None,
        cluster=None,
        strategy=None,
    ):

        if (
            model
            and not isinstance(model, paddle.nn.Layer)
            and not callable(model)
        ):
            raise TypeError(
                "'model must be sub classes of `paddle.nn.Layer` or any callable function."
            )
        self._model = model

        if (
            loss
            and not isinstance(loss, (paddle.nn.Layer, Variable))
            and not callable(loss)
        ):
            raise TypeError(
                "'loss' must be sub classes of `paddle.nn.Layer` or any callable function or a Variable."
            )
        self._loss = loss

        if optimizer and not isinstance(
            optimizer,
            (paddle.optimizer.Optimizer, paddle.static.Optimizer),
        ):
            raise TypeError(
                "'optimizer' must be object of class `paddle.optimizer.Optimizer`"
                " or `paddle.static.Optimizer`."
            )
        self._optimizer = auto_utils.validate_opt(optimizer)

        metrics = metrics or []
        for metric in auto_utils.to_list(metrics):
            if metric and not isinstance(metric, Metric):
                raise TypeError(
                    "{} is not sub class of Metric".format(
                        metric.__class__.__name__
                    )
                )
        self._metrics = auto_utils.to_list(metrics)

        if cluster and not isinstance(cluster, Cluster):
            raise TypeError(
                "'cluster' must be the object or class `paddle.distributed.auto_parallel.Cluster`"
            )
        self._cluster = cluster or get_default_cluster()

        if strategy and not isinstance(strategy, Strategy):
            raise TypeError(
                "'strategy' must be object of class `paddle.distributed.auto_parallel.Strategy`"
            )
        self._strategy = strategy or Strategy()

        self._logger = get_logger(logging.INFO)

        self._json_config = None
        if cluster:
            self._cluster = cluster
        else:
            if os.getenv("PADDLE_AUTO_PARALLEL_CONFIG"):
                try:
                    path = os.getenv("PADDLE_AUTO_PARALLEL_CONFIG")
                    with open(path, "r") as f:
                        self._json_config = json.load(f)
                except Exception as e:
                    self._logger.info(
                        "Load json failed, please check json file, engine will run default config."
                    )
                    self._json_config = None
            self._cluster = get_default_cluster(self._json_config)

        if os.getenv("POD_NAME"):
            self._logger.info(
                "Distribute training by paddle.distributed.launch"
            )
            fleet.init(is_collective=True)

        # for compute cost
        # TODO: remove _fwd_main_progs and _orig_optimizer
        self._fwd_dist_contexts = {}
        self._fwd_main_progs = {}
        self._orig_optimizer = copy.deepcopy(self._optimizer)

        self._executor = None
        self._cur_rank = paddle.distributed.get_rank()
        self._nranks = paddle.distributed.get_world_size()
        self._saver = DistributedSaver()

        self._orig_main_prog = static.default_main_program()
        self._orig_startup_prog = static.default_startup_program()
        self._orig_dist_context = get_default_distributed_context()
        self._dist_contexts = {}
        self._planners = {}
        self._has_prepared = {"train": False, "eval": False, "predict": False}
        self._has_prepared_reader = {
            "train": False,
            "eval": False,
            "predict": False,
        }
        self._inputs_spec = []
        self._labels_spec = []
        self._inputs = []
        self._labels = []
        self._losses = []

        self._mode = None
        self._skip_build = False
        self._outside_dataloader = False
        self._planned_mode = None
        self._dygraph_mode = False
        self._tuning = self._strategy.tuning
        self._acc_steps = 1
        if self._strategy.gradient_merge.enable:
            self._acc_steps = self._strategy.gradient_merge.k_steps
        elif self._strategy.pipeline.enable:
            self._acc_steps = self._strategy.pipeline.accumulate_steps

        self.history = None

        paddle.framework.set_flags({'FLAGS_new_executor_sequential_run': 1})

    def _prepare_data_spec(self, data, split, batch_size):
        inputs_spec = []
        labels_spec = []
        if isinstance(data, paddle.io.IterableDataset):
            if split is None:
                inputs, labels = next(iter(data))
            else:
                sample = next(iter(data))
                inputs = sample[:split]
                labels = sample[split:]
        elif isinstance(data, paddle.io.Dataset):
            if split is None:
                inputs, labels = data[0]
            else:
                sample = data[0]
                inputs = sample[:split]
                labels = sample[split:]
        else:
            raise TypeError(
                "Data should be a Dataset or IterableDataset, but received {}.".format(
                    type(data).__name__
                )
            )
        inputs = auto_utils.to_list(inputs)
        labels = auto_utils.to_list(labels)

        num_shards = self._strategy.dataset.num_shards

        def _adjust_item_spec(num_shards, spec):
            if num_shards > 1 and len(spec.shape) > 1:
                spec.shape[0] = spec.shape[0] * num_shards

        def _infer_item_spec(item, name, batch_size, specs):
            if isinstance(item, np.ndarray):
                spec = InputSpec.from_numpy(item, name)
                if batch_size is None:
                    _adjust_item_spec(num_shards, spec)
                    specs.append(spec)
                else:
                    specs.append(spec.batch(batch_size))
            elif isinstance(item, (Variable, core.eager.Tensor)):
                spec = InputSpec.from_tensor(item, name)
                _adjust_item_spec(num_shards, spec)
                if batch_size is None:
                    specs.append(spec)
                else:
                    specs.append(spec.batch(batch_size))
            elif isinstance(item, numbers.Number):
                specs.append(InputSpec([batch_size], type(item), name))
            else:
                raise TypeError(
                    "The sample's dtype returned of dataset should be number, np.ndarray or Tensor, but got {}".format(
                        type(item).__name__
                    )
                )

        if inputs is not None:
            for i, item in enumerate(inputs):
                assert item is not None, "Receive None input."
                name = "input" + str(i)
                _infer_item_spec(item, name, batch_size, inputs_spec)
        if labels is not None:
            for i, item in enumerate(labels):
                assert item is not None, "Receive None input."
                name = "label" + str(i)
                _infer_item_spec(item, name, batch_size, labels_spec)

        inputs_spec = self._validate_spec(inputs_spec)
        labels_spec = self._validate_spec(labels_spec)
        return inputs_spec, labels_spec

    def _prepare_data_tensor(self, inputs_spec, labels_spec, inputs, labels):
        if in_dynamic_mode() or self._dygraph_mode:
            raise ValueError("Only support static graph mode.")

        if inputs_spec:
            assert isinstance(
                inputs_spec, list
            ), "inputs should be list, but received {}".format(
                type(inputs_spec)
            )
            assert isinstance(
                inputs, list
            ), f"inputs should be list, but received {type(inputs)}"
            assert len(inputs_spec) == len(
                inputs
            ), "the number of `inputs_spec` should be equal to `inputs`'s."
            for input_spec, input in zip(inputs_spec, inputs):
                if input_spec.shape != input.shape:
                    input.desc.set_shape(input_spec.shape)
        if labels_spec:
            assert isinstance(
                labels_spec, list
            ), "labels should be list, but received {}".format(
                type(labels_spec)
            )
            assert isinstance(
                labels, list
            ), f"labels should be list, but received {type(labels)}"
            assert len(labels_spec) == len(
                labels
            ), "the number of `labels_spec` should be equal to `labels`'s."
            for label_spec, label in zip(labels_spec, labels):
                if label_spec.shape != label.shape:
                    label.desc.set_shape(label_spec.shape)

        return inputs, labels

    def _prepare_reader(self, feed_list=[]):
        dist_context = self._dist_contexts[self._mode]
        dist_main_prog = dist_context.dist_main_programs[self._cur_rank]
        dist_main_block = dist_main_prog.global_block()

        # NOTE: this list may be changed if Paddle changes the existing rules.
        related_reader_ops = [
            "create_py_reader",
            "create_double_buffer_reader",
            "read",
        ]
        # remove the first three ops if multiple run fit/evaluate/predict
        if dist_main_block.ops[0].type == 'create_py_reader':
            for i in range(len(related_reader_ops)):
                if dist_main_block.ops[0].type in related_reader_ops:
                    dist_main_block._remove_op(0, sync=False)
        dist_main_block._sync_with_cpp()
        # Step 1: find the reader ops
        reader_op_indices = []
        for idx, op in enumerate(dist_main_block.ops):
            if op.type in related_reader_ops:
                reader_op_indices.append(idx)
        # Step 2: insert the new reader ops to cpp
        # record the read ops' desc to insert to program of forward task_node
        read_ops_desc = []
        new_reader_ops = []
        for idx in reversed(reader_op_indices):
            new_op_desc = dist_main_block.desc._prepend_op()
            new_op_desc.copy_from(dist_main_block.ops[idx].desc)
            read_ops_desc.append(new_op_desc)
            new_op = Operator(
                dist_main_block, new_op_desc, type=new_op_desc.type()
            )
            new_reader_ops.append(new_op)
            dist_op = DistributedOperator(new_op)
            dist_context.add_dist_op_for_program(dist_op)
        # Step 3: insert the new reader ops to python
        for new_op in new_reader_ops:
            dist_main_block.ops.insert(0, new_op)
        for i in range(len(reader_op_indices)):
            reader_op_indices[i] += len(reader_op_indices)
        # Step 4: remove the old reader ops from python and cpp
        for idx in reversed(reader_op_indices):
            op = dist_main_block.ops.pop(idx)
            dist_main_block.desc._remove_op(idx, idx + 1)
        dist_main_block._sync_with_cpp()
        self._has_prepared_reader[self._mode] = True

        # Insert read op to forward TaskNode if 1F1B pass is setted
        if self.main_program._pipeline_opt:
            assert "tasks" in self.main_program._pipeline_opt["fleet_opt"]
            fleet_opt = self.main_program._pipeline_opt["fleet_opt"]
            fwd_task = None
            if self._strategy.pipeline.schedule_mode == "1F1B":
                fwd_task = fleet_opt["tasks"][1]
            elif self._strategy.pipeline.schedule_mode == "stream":
                fwd_task = fleet_opt["tasks"][0]
            assert fwd_task is not None
            fwd_prog = fwd_task.get_program()
            fwd_block = fwd_prog.global_block()

            for var in feed_list:
                if var.name not in fwd_block.vars:
                    fwd_block._clone_variable(var)

            for op_desc in read_ops_desc:
                new_op_desc = fwd_block.desc._prepend_op()
                new_op_desc.copy_from(op_desc)
                new_op = Operator(
                    fwd_block, new_op_desc, type=new_op_desc.type()
                )
                fwd_block.ops.insert(0, new_op)

            fwd_block._sync_with_cpp()
            fwd_task.set_program(fwd_prog)

    def _prepare_feed(self, data, user_feeds, mode):
        feeds = {}
        if data is not None:
            if isinstance(data, (list, tuple)):
                if len(data) == 1 and isinstance(data[0], dict):
                    for name, value in data[0].items():
                        feeds[name] = value
                else:
                    raise ValueError(f"Unsupported data {data}")
            elif isinstance(data, dict):
                for name, value in data.items():
                    feeds[name] = value
            else:
                raise ValueError(f"Unsupported data {data}")
        if user_feeds is not None:
            assert isinstance(
                user_feeds, dict
            ), "user_feeds must be a dict, but receive {}".format(
                type(user_feeds).__name__
            )
            for name, data in user_feeds.items():
                feeds[name] = data
        return feeds

    def _prepare_fetch(self, user_fetches, mode):
        if user_fetches is not None:
            assert isinstance(
                user_fetches, list
            ), "user_fetches must be a list, but receive {}".format(
                type(user_fetches).__name__
            )
        fetch_names = []
        fetch_indices = []

        def _process_fetch_group(group_name, var_list):
            group_indices = []
            for var in var_list:
                # Remove duplicate var_names
                if self._is_local_var(var):
                    var_name = _to_name_str(var)
                    if var_name not in fetch_names:
                        fetch_names.append(var_name)
                    group_indices.append(fetch_names.index(var_name))
            if not group_indices:
                fetch_names.append([])
            fetch_indices.append(group_indices)

        dist_context = self._dist_contexts[mode]
        fetch_vars = dist_context.serial_fetch_vars
        if mode != "predict":
            _process_fetch_group("loss", fetch_vars["loss"])
        if mode != "predict":
            metrics = fetch_vars["metrics"]
            for i, var_list in enumerate(metrics):
                _process_fetch_group("metrics_" + str(i), var_list)
        if mode == "predict":
            _process_fetch_group("outputs", fetch_vars["outputs"])
        for usr_fetch in user_fetches or []:
            var_name = _to_name_str(usr_fetch)
            fetch(var_name)
        user_fetches_collection = [
            item[1] for item in get_collection(CollectionNames.FETCHES)
        ]
        var_list = user_fetches_collection or []
        _process_fetch_group("fetches", var_list)
        return fetch_names, fetch_indices

    def _prepare_logger(
        self,
        outs,
        epoch=None,
        step=None,
        lr=None,
        fetch_names=None,
        fetch_indices=None,
        mode=None,
    ):
        logs = {}
        if epoch is not None:
            logs["epoch"] = epoch
        if step is not None:
            logs["step"] = step + 1
        if lr is not None:
            logs["lr"] = lr
        group_idx = 0
        if mode != "predict":
            # logging loss
            loss_indices = fetch_indices[group_idx]
            assert len(loss_indices) <= 1
            for idx in loss_indices:
                logs["loss"] = outs[idx]
            group_idx += 1
            # logging metrics
            dist_context = self._dist_contexts[mode]
            metric_vars = dist_context.serial_fetch_vars["metrics"]
            if metric_vars:
                for metric in self._metrics:
                    metrics_indices = fetch_indices[group_idx]
                    metric_out = []
                    for idx in metrics_indices:
                        metric_out.append(outs[idx])
                    if metric_out:
                        metric.update(*metric_out)
                        results = metric.accumulate()
                        for i, res in enumerate(auto_utils.to_list(results)):
                            logs[metric.name()[i]] = res
                    group_idx += 1
        # logging outputs
        elif mode == "predict":
            outputs_indices = fetch_indices[group_idx]
            logs_out = {}
            for idx in outputs_indices:
                logs_out["out%d" % (idx)] = outs[idx]
            logs["outputs"] = logs_out
            group_idx += 1
        # logging user fetches
        collect_fetches = get_collection(CollectionNames.FETCHES)
        logs_fetch = {}
        for name, var_name in collect_fetches:
            if var_name in fetch_names:
                idx = fetch_names.index(var_name)
                logs_fetch[name or var_name] = outs[idx]
        logs["fetches"] = logs_fetch
        return logs

    def _prepare_program(self, mode, init_parameters=True):
        # Do the build process
        self._build(mode)
        # Do the planning process
        self._plan(mode)
        # Do the parallel process
        self._parallel(mode)
        # Init comm
        self._init_comm()
        if init_parameters:
            # startup program
            self._initialize(mode)
        self._has_prepared[mode] = True

    def _build(self, mode):
        if in_dynamic_mode() or self._dygraph_mode:
            paddle.disable_static()
            self._dygraph_mode = True
            self._logger.info("Building model with 'to_static' method.")

            self.program_helper = ProgramHelper(
                self._model,
                self._loss,
                self._metrics,
                self._inputs_spec,
                self._labels_spec,
            )
            # build forward main program
            with utils.unique_name.guard():
                self.program_helper.build_program(mode)

            self.concrete_program = self.program_helper.concrete_program
            serial_main_prog = self.program_helper.main_program
            serial_startup_prog = self.program_helper.startup_program

            self._inputs = self.program_helper.input_vars
            self._labels = self.program_helper.label_vars
            outputs = self.program_helper.output_vars
            self._losses = self.program_helper.loss_vars
            metrics = self.program_helper.metric_vars

            paddle.enable_static()
        else:
            # build program in static mode
            dist_context = self._dist_contexts.get(mode, None)
            if dist_context is not None:
                return

            outputs = []
            metrics = []
            self._losses = []
            serial_main_prog = self._orig_main_prog.clone()
            serial_startup_prog = self._orig_startup_prog.clone()
            if not self._skip_build:
                with static.program_guard(
                    serial_main_prog, serial_startup_prog
                ), utils.unique_name.guard():
                    self._inputs = [
                        s._create_feed_layer() for s in self._inputs_spec
                    ]
                    self._labels = [
                        s._create_feed_layer() for s in self._labels_spec
                    ]

                    outputs = auto_utils.to_list(self._model(*self._inputs))

                    if mode != "predict" and self._loss:
                        assert isinstance(
                            self._loss, paddle.nn.Layer
                        ) or callable(
                            self._loss
                        ), "the type of `loss` of the Engine arguments should be sub classes of `paddle.nn.Layer` or any callable function."
                        self._losses = auto_utils.to_list(
                            self._loss(*(outputs + self._labels))
                        )

                    if mode != "predict" and (outputs or self._labels):
                        for metric in self._metrics:
                            metrics.append(
                                auto_utils.to_list(
                                    metric.compute(*(outputs + self._labels))
                                )
                            )
            elif mode == "train":
                assert isinstance(
                    self._loss, Variable
                ), "the type of `loss` of the Engine arguments should be Variable."
                self._losses = auto_utils.to_list(self._loss)

        default_ctx = get_default_distributed_context()
        if not default_ctx.has_annotation:
            # We build the world process group because the data parallel
            # needs all ranks by default.
            new_process_group(list(range(self._nranks)))
            default_ctx.data_parallel = True
            self._inputs = [
                auto_utils.set_data_parallel(var) for var in self._inputs
            ]
            self._labels = [
                auto_utils.set_data_parallel(var) for var in self._labels
            ]

        feed_vars = {"inputs": self._inputs, "labels": self._labels}

        fetch_vars = {
            "outputs": paddle.utils.flatten(outputs),
            "loss": self._losses,
            "metrics": metrics,
        }

        if mode != "train":
            serial_main_prog = serial_main_prog.clone(for_test=True)

        auto_utils.set_recompute_segments(
            self._model, self._losses, self._strategy, serial_main_prog
        )
        self._dist_contexts[mode] = DistributedContext(
            serial_main_prog,
            serial_startup_prog,
            self._optimizer,
            self._losses,
            feed_vars,
            fetch_vars,
            self._cluster,
            self._strategy,
            self._json_config,
        )
        self._fwd_dist_contexts[mode] = DistributedContext(
            serial_main_prog,
            serial_startup_prog,
            self._optimizer,
            self._losses,
            feed_vars,
            fetch_vars,
            self._cluster,
            self._strategy,
            self._json_config,
        )
        self._dist_contexts[mode].gradient_scale = self._strategy.gradient_scale
        self._fwd_main_progs[mode] = serial_main_prog.clone()

    def _optimization_tuning(self, mode, dataset, batch_size):
        if not self._tuning.enable:
            raise ValueError("Please set `tuning.enable=True`.")

        assert mode == "train"
        # Do the build process
        self._build(mode)
        # Do the planning process
        self._plan(mode)

        dataset.dp_world_size = self._dp_world_sizes
        dataset.dp_rank = self._dp_ranks

        from .tuner.optimization_tuner import OptimizationTuner

        self._optimization_tuner = OptimizationTuner(
            self._dist_contexts[mode],
            dataset,
            self._inputs_spec,
            self._labels_spec,
            batch_size=batch_size,
            rank=self._cur_rank,
        )

        self._optimization_tuner.tune()

        if self._tuning.run_after_tuning:
            # update the strategy
            self._dist_contexts[
                mode
            ]._strategy = self._optimization_tuner.get_best_config()

    def _plan(self, mode):
        if self._planned_mode is None:
            self._planned_mode = mode
        else:
            self._init_dist_context(mode)

        self._planners[mode] = Planner(mode, self._dist_contexts[mode])
        self._planners[mode].plan()

        # infer data parallel info
        inputs_var = self._dist_contexts[mode].serial_feed_vars["inputs"]
        labels_var = self._dist_contexts[mode].serial_feed_vars["labels"]
        block = self._dist_contexts[mode].serial_main_program.global_block()
        # TODO: check this feed_list
        feed_list = []
        for var in inputs_var + labels_var:
            if var.name in block.vars:
                feed_list.append(block.vars[var.name])

        self._dp_world_sizes = []
        self._dp_ranks = []
        for feed_var in feed_list:
            dp_world_size, dp_rank = auto_utils.get_input_split_info(
                self._cur_rank, feed_var, self._dist_contexts[mode]
            )
            self._dp_world_sizes.append(dp_world_size)
            self._dp_ranks.append(dp_rank)

    def _parallel(self, mode, all_ranks=False):
        # Parallelize program based on the planner's results
        # For now, the completer has to be passed to the planner,
        # because we may use it to complete the annotation of the backward and update.
        parallelizer = Parallelizer(
            mode,
            self._planners[mode].completer,
            self._dist_contexts[mode],
        )
        if not all_ranks:
            parallelizer.parallel(self._cur_rank)
        else:
            parallelizer.parallel_all()

    def _init_dist_context(self, mode):
        # Init dist_context['mode'] with the first planned dist_context
        # to guarantee that train/eval/predict mode have same parallel strategy
        dist_context = self._dist_contexts[mode]
        origin_main_prog = dist_context._original_serial_main_program
        ref_mode = self._planned_mode
        ref_dist_context = self._dist_contexts[ref_mode]
        ref_origin_main_prog = ref_dist_context._original_serial_main_program
        ref_blocks = ref_origin_main_prog.blocks
        for ib, block in enumerate(origin_main_prog.blocks):
            for iop, op in enumerate(block.ops):
                ref_op = ref_blocks[ib].ops[iop]
                assert (
                    op.type == ref_op.type
                ), "'{}' mode op '{}' is different with '{}' op '{}'. ".format(
                    mode, op.type, ref_mode, ref_op.type
                )
                ref_op_dist_attr = (
                    ref_dist_context.get_op_dist_attr_for_program(ref_op)
                )
                dist_context.set_op_dist_attr_for_program(op, ref_op_dist_attr)

    def _init_comm(self):
        if self._nranks > 1:
            # Traverse different rank programs and traverse each op of them,
            # instantiate communication by process_mapping.
            all_process_groups = get_all_process_groups()

            if self._strategy.auto_mode == "full_random":
                auto_utils.initialize_pg_in_full_mode(
                    all_process_groups, self._cur_rank
                )
            else:
                for process_group in all_process_groups:
                    process_group.instantiate()

    def _initialize(self, mode):
        self._place = _get_device()
        if isinstance(self._place, paddle.framework.CUDAPlace):
            self._place = paddle.framework.CUDAPlace(
                paddle.distributed.ParallelEnv().dev_id
            )

        if self._strategy.seed:
            paddle.seed(self._strategy.seed + self._dp_ranks[0])
            np.random.seed(self._strategy.seed + self._dp_ranks[0])
            random.seed(self._strategy.seed + self._dp_ranks[0])

        dist_context = self._dist_contexts[mode]
        if self._dygraph_mode:
            dist_main_program = dist_context.dist_main_programs[self._cur_rank]
            self.program_helper.init(
                dist_main_program, self._place, dist_context
            )

        if self._executor is None:
            self._executor = paddle.static.Executor(self._place)
            uninitialized = []
            dist_startup_prog = dist_context.dist_startup_programs[
                self._cur_rank
            ]
            for var in dist_startup_prog.list_vars():
                scope_var = global_scope().find_var(var.name)
                if scope_var and scope_var.get_tensor()._is_initialized():
                    continue
                uninitialized.append(var)
            if uninitialized:
                prune_startup_prog = dist_startup_prog._prune(uninitialized)
                self._executor.run(prune_startup_prog)

            if hasattr(self, "_state_dict") and hasattr(self, "_dist_attr"):
                self._set_state_dict(
                    mode, self._strict, self._state_dict, self._dist_attr
                )

        if self._strategy.reinit:
            self._logger.info("NOTE: parameters will be re-initialized.")
            dist_startup_prog = dist_context.dist_startup_programs[
                self._cur_rank
            ]
            self._executor.run(dist_startup_prog)

    def fit(
        self,
        train_data,
        train_sample_split=None,
        batch_size=1,
        epochs=1,
        steps_per_epoch=None,
        log_freq=10,
        save_dir=None,
        save_freq=1,
        valid_data=None,
        valid_sample_split=None,
        valid_freq=1,
        valid_steps=None,
        collate_fn=None,
        callbacks=None,
        verbose=2,
    ):
        """
        Trains the model for a fixed number of epochs. If `valid_data` is set,
        evaluation will be done at the end of each epoch.

        Args:
            train_data (Dataset): An instance of paddle paddle.io.Dataset. Default: None.
            train_sample_split (int, optional): Each sample of the train dataset is assumed
                to be a (input, label) pair by default and has two items. If each sample has
                more than two items, train_sample_split specifies how to split these items into
                input and label. The items before it are input and the left are label. Default: None.
            batch_size (int, optional): The batch size of train_data and valid_data if provided.
                The user's data will be used directly without batching if set to None. Default: 1.
            epochs (int, optional): The number of epochs to train the model. Default: 1.
            steps_per_epoch (int, optional): The total number of steps (batches of samples)
                is executed in one epoch before stating the next one. If None, it is equal to
                the number samples in your dataset divided by the batch size. Default: None.
            valid_data (Dataset, optional): An instance of paddle paddle.io.Dataset used for
                evaluation at the end of epoch. No evaluation will be done if set to None.
                Default: None. (Unsupported for now)
            valid_freq (int, optional): Only relevant if valid_data is provided. This specifies
                how many training epochs before a new evaluation is performed. Default: 1.
            valid_sample_split (int, optional): Only relevant if valid_data is provided.
                Each sample of the valid dataset is assumed to be a (input, label) pair
                by default and has two items. If each sample has more than two items,
                valid_sample_split specifies how to split these items into input and label.
                The items before it are input and the left are label. Default: None.
            valid_steps (int, optional): Only relevant if valid_data is provided.
                It is the total number of steps (batches of samples) to draw before
                stopping validation at the end of every epoch. If None, validation will run until the
                `valid_data` dataset is exhausted. The validation will start from the
                beginning of the dataset at each epoch. Default: None.
            collate_fn(callable, optional): function to generate mini-batch data by merging
                the sample list, None for only stack each fields of sample in axis
                0. Default None.
            callbacks (Callback|None, optional): A list of `Callback` instances to apply
                during training. Default: None. (Unused for now)

        Returns:
            None

        Examples:

            .. code-block:: python

                import paddle
                import paddle.vision.transforms as T
                from paddle.distributed.fleet import auto
                from paddle.vision.datasets import MNIST

                transform = T.Compose([
                    T.Transpose(),
                    T.Normalize([127.5], [127.5])
                ])
                train_dataset = MNIST(mode='train', transform=transform)

                model = paddle.vision.models.LeNet()
                loss = paddle.nn.CrossEntropyLoss()
                optimizer = paddle.optimizer.Adam(
                    learning_rate=0.001, parameters=model.parameters())
                metrics = paddle.metric.Accuracy(topk=(1, 2))

                engine = auto.Engine(model, loss, optimizer, metrics)
                engine.fit(train_dataset,
                           epochs=2,
                           batch_size=64)
        """
        self._mode = 'train'
        self._inputs_spec, self._labels_spec = self._prepare_data_spec(
            train_data, train_sample_split, batch_size
        )
        micro_batch_size = self._validate_batch_size(batch_size)
        if not self._has_prepared[self._mode]:
            self._prepare_program(self._mode)
        else:
            self._switch_mode(self._mode)

        train_dataloader = self._prepare_dataloader_from_generator(
            dataset=train_data,
            capacity=70,
            iterable=False,
            batch_size=micro_batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            collate_fn=collate_fn,
        )

        fetch_names, fetch_indices = self._prepare_fetch(None, mode=self._mode)

        cbks = config_callbacks(
            callbacks,
            engine=self,
            batch_size=micro_batch_size,
            epochs=epochs,
            steps=train_dataloader._steps,
            log_freq=log_freq,
            save_freq=save_freq,
            save_dir=save_dir,
            verbose=verbose,
            metrics=self._metrics_name(),
            acc_step=self._acc_steps,
        )

        cbks.on_begin('train')
        for epoch in range(epochs):
            logs = {}
            cbks.on_epoch_begin(epoch)
            for step, _ in enumerate(train_dataloader):
                cbks.on_batch_begin('train', step, logs)
                try:
                    outs = self._executor.run(
                        self.main_program,
                        fetch_list=fetch_names,
                        use_program_cache=self._strategy.use_cache,
                        return_numpy=self._strategy.return_numpy,
                    )
                except core.EOFException:
                    break
                lr = auto_utils.get_lr(self.optimizer)
                logs = self._prepare_logger(
                    outs,
                    epoch,
                    step,
                    lr,
                    fetch_names,
                    fetch_indices,
                    self._mode,
                )
                cbks.on_batch_end('train', step, logs)

            if valid_data and (epoch + 1) % valid_freq == 0:
                val_logs = self.evaluate(
                    valid_data,
                    valid_sample_split,
                    batch_size,
                    valid_steps,
                    log_freq,
                    collate_fn,
                    callbacks,
                    verbose,
                )
                val_logs = {
                    "val_" + name: val for name, val in val_logs.items()
                }
                logs.update(val_logs)
                self._switch_mode("train")
            else:
                self._reset_metrics()

            cbks.on_epoch_end(epoch, logs)

        cbks.on_end('train', logs)
        return self.history

    def evaluate(
        self,
        valid_data,
        valid_sample_split=None,
        batch_size=1,
        steps=None,
        log_freq=10,
        collate_fn=None,
        callbacks=None,
        verbose=2,
    ):
        """
        Evaluate the loss and metrics of the model on evaluation data.

        Args:
            valid_data (Dataset): An instance of paddle paddle.io.Dataset. Default: None.
            valid_sample_split (int, optional): Each sample of the eval dataset is assumed
                to be a (input, label) pair by default and has two items. If each sample has
                more than two items, valid_sample_split specifies how to split these items into
                input and label. The items before it are input and the left are label. Default: None.
            batch_size (int, optional): The batch size of valid_data. The user's data will
                be used directly without batching if set to None. Default: 1.
            steps (int, optional): It is the total number of steps (batches of samples) to draw before
                stopping evaluation. If None, evaluation will run until the `valid_data` dataset is exhausted.
                The evaluation will start from the beginning of the dataset in each run. Default: None.
            collate_fn(callable, optional): function to generate mini-batch data by merging
                the sample list, None for only stack each fields of sample in axis
                0. Default None.
            callbacks (Callback|None, optional): A list of `Callback` instances to apply
                during evaluating. Default: None. (Unused for now)

        Returns:
            None

        Examples:

            .. code-block:: python

                import paddle
                import paddle.vision.transforms as T
                from paddle.distributed.fleet import auto
                from paddle.vision.datasets import MNIST

                transform = T.Compose([
                    T.Transpose(),
                    T.Normalize([127.5], [127.5])
                ])
                valid_dataset = MNIST(mode='test', transform=transform)

                model = paddle.vision.models.LeNet()
                loss = paddle.nn.CrossEntropyLoss()
                metrics = paddle.metric.Accuracy(topk=(1, 2))

                engine = auto.Engine(model, loss, metrics=metrics)
                engine.evaluate(valid_dataset, batch_size=64)

        """
        self._mode = 'eval'
        self._inputs_spec, self._labels_spec = self._prepare_data_spec(
            valid_data, valid_sample_split, batch_size
        )
        micro_batch_size = self._validate_batch_size(batch_size)
        if not self._has_prepared[self._mode]:
            self._prepare_program(self._mode)
        else:
            self._switch_mode(self._mode)

        valid_dataloader = self._prepare_dataloader_from_generator(
            dataset=valid_data,
            capacity=70,
            iterable=False,
            batch_size=micro_batch_size,
            steps_per_epoch=steps,
            collate_fn=collate_fn,
        )

        fetch_names, fetch_indices = self._prepare_fetch(None, mode=self._mode)

        cbks = config_callbacks(
            callbacks,
            engine=self,
            batch_size=micro_batch_size,
            log_freq=log_freq,
            verbose=verbose,
            metrics=self._metrics_name(),
        )

        eval_steps = valid_dataloader._steps
        cbks.on_begin(
            'eval', {'steps': eval_steps, 'metrics': self._metrics_name()}
        )
        logs = {}
        for step, _ in enumerate(valid_dataloader):
            cbks.on_batch_begin('eval', step, logs)
            try:
                outs = self._executor.run(
                    self.main_program,
                    fetch_list=fetch_names,
                    use_program_cache=self._strategy.use_cache,
                    return_numpy=self._strategy.return_numpy,
                )
            except core.EOFException:
                break
            logs = self._prepare_logger(
                outs, None, step, None, fetch_names, fetch_indices, self._mode
            )
            cbks.on_batch_end('eval', step, logs)
        cbks.on_end('eval', logs)
        self._reset_metrics()
        return logs

    def predict(
        self,
        test_data,
        test_sample_split=None,
        batch_size=1,
        steps=None,
        collate_fn=None,
        callbacks=None,
        verbose=2,
    ):
        """
        Compute the output predictions on testing data.

        Args:
            test_data (Dataset): An instance of paddle paddle.io.Dataset. Default: None.
            test_sample_split (int, optional): Each sample of the test dataset is assumed
                to be a (input, label) pair by default and has two items. If each sample has
                more than two items, test_sample_split specifies how to split these items into
                input and label. The items before it are input and the left are label. Default: None.
            batch_size (int, optional): The batch size of test_data. The user's data will
                be used directly without batching if set to None. Default: 1.
            steps (int, optional): It is the total number of steps (batches of samples) to draw before
                stopping predict. If None, predict will run until the `test_data` dataset is exhausted.
                The predict will start from the beginning of the dataset in each run. Default: None.
            collate_fn(callable, optional): function to generate mini-batch data by merging
                the sample list, None for only stack each fields of sample in axis
                0. Default None.
            callbacks (Callback|None, optional): A list of `Callback` instances to apply
                during testing. Default: None. (Unused for now)

        Returns:
            None

        Examples:

            .. code-block:: python

                import paddle
                import paddle.vision.transforms as T
                from paddle.distributed.fleet import auto
                from paddle.vision.datasets import MNIST

                transform = T.Compose([
                    T.Transpose(),
                    T.Normalize([127.5], [127.5])
                ])
                valid_dataset = MNIST(mode='test', transform=transform)

                model = paddle.vision.models.LeNet()

                engine = auto.Engine(model)
                engine.predict(valid_dataset, batch_size=64)
        """
        self._mode = 'predict'
        self._inputs_spec, self._labels_spec = self._prepare_data_spec(
            test_data, test_sample_split, batch_size
        )
        micro_batch_size = self._validate_batch_size(batch_size)
        if not self._has_prepared[self._mode]:
            self._prepare_program(self._mode)
        else:
            self._switch_mode(self._mode)

        test_dataloader = self._prepare_dataloader_from_generator(
            dataset=test_data,
            capacity=70,
            iterable=False,
            batch_size=micro_batch_size,
            steps_per_epoch=steps,
            collate_fn=collate_fn,
        )

        fetch_names, fetch_indices = self._prepare_fetch(None, mode=self._mode)

        outputs = []
        cbks = config_callbacks(callbacks, engine=self, verbose=verbose)
        test_steps = test_dataloader._steps
        cbks.on_begin('predict', {'steps': test_steps})
        logs = {}
        for step, _ in enumerate(test_dataloader):
            cbks.on_batch_begin('predict', step, logs)
            try:
                outs = self._executor.run(
                    self.main_program,
                    fetch_list=fetch_names,
                    use_program_cache=self._strategy.use_cache,
                    return_numpy=self._strategy.return_numpy,
                )
            except core.EOFException:
                break
            logs = self._prepare_logger(
                outs, None, step, None, fetch_names, fetch_indices, self._mode
            )
            cbks.on_batch_end('predict', step, logs)
            outputs.append(list(logs["outputs"].values()))
        cbks.on_end('predict', logs)
        return outputs

    def dataloader(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=None,
        num_workers=0,
        use_buffer_reader=True,
        use_shared_memory=True,
        timeout=0,
        worker_init_fn=None,
        epochs=1,
        steps_per_epoch=None,
        sample_split=1,
        mode=None,
    ):
        if mode is not None:
            self.to_mode(mode)
        self._inputs_spec, self._labels_spec = self._prepare_data_spec(
            dataset, sample_split, batch_size
        )
        micro_batch_size = self._validate_batch_size(batch_size)
        if not self._has_prepared[self._mode]:
            self._prepare_program(self._mode)
        else:
            self._switch_mode(self._mode)

        dataloader = self._prepare_dataloader(
            dataset,
            return_list=False,
            batch_size=micro_batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
            num_workers=num_workers,
            use_buffer_reader=use_buffer_reader,
            use_shared_memory=use_shared_memory,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )
        return dataloader

    def dataloader_from_generator(
        self,
        dataset,
        capacity=70,
        use_double_buffer=True,
        iterable=True,
        use_multiprocess=False,
        drop_last=True,
        batch_size=1,
        epochs=1,
        steps_per_epoch=None,
        collate_fn=None,
        sample_split=1,
        mode=None,
    ):
        if mode is not None:
            self.to_mode(mode)
        self._inputs_spec, self._labels_spec = self._prepare_data_spec(
            dataset, sample_split, batch_size
        )
        micro_batch_size = self._validate_batch_size(batch_size)
        if not self._has_prepared[self._mode]:
            self._prepare_program(self._mode)
        else:
            self._switch_mode(self._mode)

        dataloader = self._prepare_dataloader_from_generator(
            dataset=dataset,
            capacity=capacity,
            use_double_buffer=use_double_buffer,
            iterable=iterable,
            return_list=False,
            use_multiprocess=use_multiprocess,
            drop_last=drop_last,
            batch_size=micro_batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            collate_fn=collate_fn,
        )
        return dataloader

    def prepare(
        self,
        inputs_spec=None,
        labels_spec=None,
        inputs=None,
        labels=None,
        main_program=None,
        startup_program=None,
        mode=None,
        init_parameters=True,
    ):
        if mode is not None:
            self.to_mode(mode)

        if not self._mode:
            raise ValueError(
                "Please set mode to be prepared with `prepare(mode=...)`"
            )

        if self._has_prepared[self._mode]:
            return

        inputs_spec = self._validate_spec(inputs_spec)
        labels_spec = self._validate_spec(labels_spec)
        inputs = self._validate_vars(inputs)
        labels = self._validate_vars(labels)

        self._orig_main_prog = main_program
        self._orig_startup_prog = startup_program
        if inputs or labels:
            self._skip_build = True
            inputs, labels = self._prepare_data_tensor(
                inputs_spec, labels_spec, inputs, labels
            )
            if self._orig_main_prog is None:
                self._orig_main_prog = static.default_main_program()
            if self._orig_startup_prog is None:
                self._orig_startup_prog = static.default_startup_program()
        elif inputs_spec or labels_spec:
            self._outside_dataloader = True
            if self._orig_main_prog is None:
                self._orig_main_prog = static.default_main_program()
            if self._orig_startup_prog is None:
                self._orig_startup_prog = static.default_startup_program()
        else:
            assert (
                self._inputs_spec and self._labels_spec
            ), "Please call the dataloader(...) before calling prepare(...)"

        self._inputs_spec, self._labels_spec = inputs_spec, labels_spec
        self._inputs, self._labels = inputs, labels
        if not self._has_prepared[self._mode]:
            self._prepare_program(self._mode, init_parameters)
        else:
            self._switch_mode(self._mode)

    def run(self, data=None, feed=None, fetch_list=None, mode=None):
        if mode is not None:
            self.to_mode(mode)
        feed_dict = self._prepare_feed(data, feed, self._mode)
        fetch_names, fetch_indices = self._prepare_fetch(fetch_list, self._mode)
        if (
            self._outside_dataloader
            and not self._has_prepared_reader[self._mode]
        ):
            self._prepare_reader()
        outs = self._executor.run(
            self.main_program,
            feed=feed_dict,
            fetch_list=fetch_names,
            use_program_cache=self._strategy.use_cache,
            return_numpy=self._strategy.return_numpy,
        )
        logs = self._prepare_logger(
            outs, None, None, None, fetch_names, fetch_indices, self._mode
        )
        return logs

    def _prepare_dataloader(
        self,
        dataset,
        return_list=True,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=None,
        num_workers=0,
        use_buffer_reader=True,
        use_shared_memory=True,
        timeout=0,
        worker_init_fn=None,
        epochs=1,
        steps_per_epoch=None,
    ):

        dist_context = self._dist_contexts[self._mode]
        dist_main_prog = dist_context.dist_main_programs[self._cur_rank]
        dist_startup_prog = dist_context.dist_startup_programs[self._cur_rank]
        dist_main_block = dist_main_prog.global_block()

        # NOTE: Get feed_list, then insert dataloader op with sharded var shape.
        # Cause predict_program does not contain labels var,
        # then we will add labels var from serial_program to dist_program,
        # that maintains the length of feed_list equal to the length of dataset's values.
        inputs_var = dist_context.serial_feed_vars["inputs"]
        labels_var = dist_context.serial_feed_vars["labels"]
        feed_list = []
        for var in inputs_var + labels_var:
            if var.name in dist_main_block.vars:
                feed_list.append(dist_main_block.vars[var.name])
            else:
                copy_var = dist_main_block._clone_variable(var, var.persistable)
                copy_var.desc.set_original_id(var.desc.original_id())
                feed_list.append(copy_var)

        # insert read op at the end of program
        places = paddle.static.cuda_places()
        with static.program_guard(dist_main_prog, dist_startup_prog):
            dataloader = DistributedDataLoader(
                dataset,
                feed_list=feed_list,
                places=places,
                return_list=return_list,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                collate_fn=collate_fn,
                num_workers=num_workers,
                use_buffer_reader=use_buffer_reader,
                use_shared_memory=use_shared_memory,
                timeout=timeout,
                worker_init_fn=worker_init_fn,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                split_data=self._strategy.split_data,
                data_parallel_world_size=self._dp_world_sizes,
                data_parallel_rank=self._dp_ranks,
            )

        return dataloader

    def _prepare_dataloader_from_generator(
        self,
        dataset,
        capacity=None,
        use_double_buffer=True,
        iterable=True,
        return_list=False,
        use_multiprocess=False,
        drop_last=True,
        batch_size=1,
        epochs=1,
        steps_per_epoch=None,
        collate_fn=None,
    ):

        dist_context = self._dist_contexts[self._mode]
        dist_main_prog = dist_context.dist_main_programs[self._cur_rank]
        dist_startup_prog = dist_context.dist_startup_programs[self._cur_rank]
        dist_main_block = dist_main_prog.global_block()

        # NOTE: Get feed_list, then insert dataloader op with sharded var shape.
        # Cause predict_program does not contain labels var,
        # then we will add labels var from serial_program to dist_program,
        # that maintains the length of feed_list equal to the length of dataset's values.
        inputs_var = dist_context.serial_feed_vars["inputs"]
        labels_var = dist_context.serial_feed_vars["labels"]
        feed_list = []
        for var in inputs_var + labels_var:
            if var.name in dist_main_block.vars:
                feed_list.append(dist_main_block.vars[var.name])
            else:
                copy_var = dist_main_block._clone_variable(var, var.persistable)
                copy_var.desc.set_original_id(var.desc.original_id())
                feed_list.append(copy_var)

        places = paddle.static.cuda_places()
        with static.program_guard(dist_main_prog, dist_startup_prog):
            dataloader = DistributedDataLoaderFromGenerator(
                dataset=dataset,
                feed_list=feed_list,
                capacity=capacity,
                use_double_buffer=use_double_buffer,
                iterable=iterable,
                return_list=return_list,
                use_multiprocess=use_multiprocess,
                drop_last=drop_last,
                places=places,
                batch_size=batch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                collate_fn=collate_fn,
                split_data=self._strategy.split_data,
                data_parallel_world_size=self._dp_world_sizes,
                data_parallel_rank=self._dp_ranks,
                acc_steps=1
                if not self._strategy.pipeline.enable
                else self._acc_steps,
            )
        self._prepare_reader(feed_list)
        return dataloader

    def _tune(self, tune_data, tune_sample_split=None, batch_size=1):
        self._mode = 'train'
        self._inputs_spec, self._labels_spec = self._prepare_data_spec(
            tune_data, tune_sample_split, batch_size
        )
        self._optimization_tuning(self._mode, tune_data, batch_size)

    def _validate_batch_size(self, batch_size):
        if batch_size is None:
            return None
        assert (
            batch_size % self._acc_steps == 0
        ), "Requires batch_size:[{}] to be divisible by acc_steps:[{}].".format(
            batch_size, self._acc_steps
        )
        return batch_size // self._acc_steps

    def _validate_spec(self, specs):
        specs = auto_utils.to_list(specs)
        if specs is not None:
            for i, spec in enumerate(specs):
                if not isinstance(spec, InputSpec):
                    raise TypeError(
                        "'spec' must be object of class `paddle.static.InputSpec`."
                    )
                if spec.name is None:
                    raise ValueError(
                        "Requires Input[{}].name != None, but receive `None` with {}.".format(
                            i, spec
                        )
                    )
                if self._acc_steps > 1:
                    shape = list(spec.shape)
                    assert (
                        shape[0] % self._acc_steps == 0
                    ), "Requires batch_size[{}] to be divisible by k_steps[{}].".format(
                        spec.shape[0], self._acc_steps
                    )
                    shape[0] //= self._acc_steps
                    spec.shape = shape
        return specs or []

    def _validate_vars(self, vars):
        vars = auto_utils.to_list(vars)
        if vars is not None:
            for i, var in enumerate(vars):
                if not isinstance(var, Variable):
                    raise TypeError("'var' must be a `Variable`.")
        return vars or []

    def _is_local_var(self, var):
        var_name = _to_name_str(var)
        return var_name in self.main_program.global_block().vars

    def _reset_metrics(self):
        for metric in self._metrics:
            metric.reset()

    def _metrics_name(self):
        metrics_name = ['loss'] if self._loss else []
        for m in self._metrics:
            metrics_name.extend(auto_utils.to_list(m.name()))
        return metrics_name

    def _switch_mode(self, mode):
        assert (
            mode in self._dist_contexts
        ), f"{mode} model is not ready, please call `prepare()` first."
        self.to_mode(mode)

    def to_mode(self, mode):
        assert mode in [
            "train",
            "eval",
            "predict",
        ], f"mode {mode} should be one of ['train', 'eval', 'predict']"
        self._mode = mode

    def _set_state_dict(self, mode, strict, state_dict, dist_attr):
        dist_context = self._dist_contexts[mode]
        program = dist_context.dist_main_programs[self._cur_rank]
        cur_dist_attr = auto_utils.get_dist_attr(program, dist_context)
        converter = Converter(state_dict, dist_attr, cur_dist_attr)
        state_dict = converter.convert(strict=strict)
        for name, param in program.state_dict().items():
            param_array = np.array(param)
            if name not in state_dict:
                continue
            if param_array.dtype != state_dict[name].dtype:
                self._logger.info(
                    "cast {}'s dtype from '{}' to '{}'".format(
                        name,
                        str(state_dict[name].dtype),
                        str(param_array.dtype),
                    )
                )
                state_dict[name] = state_dict[name].astype(param_array.dtype)
        program.set_state_dict(state_dict)

    def save(self, path, training=True):
        """
        Saves the model, parameters, optimizer state to path.
        If `training` is set to False, only inference model will be saved.

        Args:
            path (str): The file prefix to save model. The format
                is 'dirname/file_prefix' or 'file_prefix'. if empty str.
                A exception will be raised.
            training (bool, optional): Whether to save for training. If not, save
                for inference only. If `training` is set to True, the optimizer state
                will be saved. Otherwise, only the model and parameters are saved.
                This function will silently overwrite existing file at the target
                location. Default: True.

        Returns:
            None

        Examples:

            .. code-block:: python
                import paddle
                import paddle.vision.transforms as T
                from paddle.distributed.fleet import auto
                from paddle.vision.datasets import MNIST

                transform = T.Compose([
                    T.Transpose(),
                    T.Normalize([127.5], [127.5])
                ])
                train_dataset = MNIST(mode='train', transform=transform)

                model = paddle.vision.models.LeNet()
                loss = paddle.nn.CrossEntropyLoss()
                optimizer = paddle.optimizer.Adam(
                    learning_rate=0.001, parameters=model.parameters())
                metrics = paddle.metric.Accuracy(topk=(1, 2))

                engine = auto.Engine(model, loss, optimizer, metrics)
                engine.fit(train_dataset,
                           epochs=1,
                           batch_size=64)
                engine.save("./my_model")

        """
        if training:
            assert self._mode in self._dist_contexts
            dist_context = self._dist_contexts[self._mode]
            serial_program = dist_context.serial_main_program
            dist_main_prog = dist_context.dist_main_programs[self._cur_rank]
            self._saver.save(
                path,
                serial_program=serial_program,
                dist_main_program=dist_main_prog,
                dist_context=dist_context,
            )
        else:
            assert "predict" in self._dist_contexts
            dist_context = self._dist_contexts["predict"]
            feed_vars = dist_context.serial_feed_vars['inputs']
            fetch_vars = dist_context.serial_fetch_vars['outputs']
            dist_main_prog = dist_context.dist_main_programs[self._cur_rank]
            if self._strategy.qat.enable and self._strategy.qat.onnx_format:
                from paddle.static.quantization import QuantWeightPass

                self._logger.info("export quantized model.")
                self._logger.info(
                    f"convert config {self._strategy.qat.to_dict()}"
                )
                test_graph = IrGraph(
                    core.Graph(dist_main_prog.desc), for_test=True
                )
                quant_weight_pass = QuantWeightPass(global_scope(), self._place)
                for sub_graph in test_graph.all_sub_graphs():
                    quant_weight_pass.apply(sub_graph)
                dist_main_prog = test_graph.to_program()
            self._saver.save_inference_model(
                path,
                feed_vars,
                fetch_vars,
                self._executor,
                program=dist_main_prog,
            )

    def load(self, path, strict=True, load_optimizer=True):
        """
        Load the stored model, parameters and optimizer states.

        Args:
            path (str): The prefix of files storing the model states and
                optimizer states.
            strict (bool, optional): Whether to skip the loading of mismatch
                parameter or raise an error when mismatch happens (not found
                the parameter in file storing model states of or receives a
                mismatch shape). Default: True.
            load_optimizer (bool, optional): If True, the stored optimizer
                states is restored. Otherwise, the optimizer states is initialized
                from scratch. Default: True.

        Returns:
            None

        Examples:

            .. code-block:: python
                import paddle
                import paddle.vision.transforms as T
                from paddle.distributed.fleet import auto
                from paddle.vision.datasets import MNIST

                transform = T.Compose([
                    T.Transpose(),
                    T.Normalize([127.5], [127.5])
                ])
                train_dataset = MNIST(mode='train', transform=transform)

                model = paddle.vision.models.LeNet()
                loss = paddle.nn.CrossEntropyLoss()
                optimizer = paddle.optimizer.Adam(
                    learning_rate=0.001, parameters=model.parameters())
                metrics = paddle.metric.Accuracy(topk=(1, 2))

                engine = auto.Engine(model, loss, optimizer, metrics)
                engine.fit(train_dataset,
                           epochs=1,
                           batch_size=64)
                engine.save("./my_model")
                engine.load("./my_model")

        """
        self._strict = strict
        self._state_dict, self._dist_attr = self._saver.load(
            path, load_optimizer
        )
        return self._state_dict, self._dist_attr

    def cost(self, inputs_spec=None, labels_spec=None, mode=None):
        """
        Get and Print cost, including memory of every rank,
        max memory among all ranks, and the global cost of one step based on
        communication cost(computation cost is 0 by default).
        In the future, the flops information of every rank and global cost including
        computation cost will be added.

        Args:
            inputs_spec(InputSpec): The specification of inputs. Default: None.
            labels_spec(InputSpec): The specification of labels. Default: None.
            mode (str): The engine mode must be in ["train", "predict", "eval"]. Default: None.

        Returns:
            Return the global execution time (ms) and max memory (B).

        """
        # Check parallel mode
        if self._strategy.auto_mode == "full":
            self._logger.info(
                "The cost will be calcudated in the search process when the auto mode is full."
            )
            return

        # Check mode
        mode = mode if mode is not None else self._mode
        assert mode is not None, "Please set mode."
        if mode not in self._has_prepared:
            raise ValueError(
                "The mode {} is not in accepted modes {}".format(
                    mode, list(self._has_prepared.keys())
                )
            )
        self.to_mode(mode)

        if inputs_spec is not None and not self._has_prepared[mode]:
            self._inputs_spec = self._validate_spec(inputs_spec)
            self._labels_spec = self._validate_spec(labels_spec)
            self._build(mode)
            self._plan(mode)
        else:
            if in_dynamic_mode() or self._dygraph_mode:
                raise ValueError(
                    "Please call `prepare()` or `fit()` or  `evaluate()` or  `predict()` before calling `cost()`."
                )
            else:
                self._logger.info(
                    "The program whose cost to be estimated must be static default program. Otherwise, please call `prepare()`before calling `cost()`."
                )
                program = paddle.static.default_main_program()
                if (
                    not program.global_block().ops
                    or not program.global_block().ops
                ) and not self._has_prepared[mode]:
                    raise ValueError(
                        "Please call `prepare()` or `fit()` or  `evaluate()` or  `predict()` before calling `cost()`."
                    )

        # Estimate the exec cost and max memory
        global_cost, max_memory = get_cost_from_engine(self, mode)

        return global_cost.time, max_memory

    @property
    def main_program(self):
        dist_context = self._dist_contexts[self._mode]
        return dist_context.dist_main_programs[self._cur_rank]

    @property
    def startup_program(self):
        dist_context = self._dist_contexts[self._mode]
        return dist_context.dist_startup_programs[self._cur_rank]

    @property
    def dist_context(self):
        return self._dist_contexts[self._mode]

    @property
    def serial_main_program(self):
        dist_context = self._dist_contexts[self._mode]
        return dist_context.serial_main_program

    @property
    def serial_startup_program(self):
        dist_context = self._dist_contexts[self._mode]
        return dist_context.serial_startup_program

    @property
    def feed_vars(self):
        dist_context = self._dist_contexts[self._mode]
        return dist_context.serial_feed_vars

    @property
    def fetch_vars(self):
        dist_context = self._dist_contexts[self._mode]
        return dist_context.serial_fetch_vars

    @property
    def optimizer(self):
        dist_context = self._dist_contexts[self._mode]
        if dist_context._serial_optimizer:
            return dist_context._serial_optimizer
        return self._optimizer

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels
