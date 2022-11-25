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

import os
import copy
import logging
import random
import numbers
import numpy as np
from collections import defaultdict
import socket

import paddle
import paddle.utils as utils
<<<<<<< HEAD
import paddle.distributed.auto_parallel.utils as auto_utils

from paddle import fluid, static
=======

from paddle import fluid, static
from paddle.io import Dataset
from paddle.jit import to_static
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
from paddle.metric import Metric
from paddle.static import InputSpec
from paddle.fluid import core
from paddle.fluid import Variable
from paddle.fluid.layers.utils import flatten
from paddle.fluid.executor import global_scope, _to_name_str
<<<<<<< HEAD
from paddle.fluid.framework import Operator, _non_static_mode
from paddle.fluid.framework import _current_expected_place as _get_device
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.distributed import fleet

from .callbacks import config_callbacks
from .converter import Converter
from .helper import ProgramHelper
=======
from paddle.fluid.backward import append_backward
from paddle.fluid.framework import Operator, Parameter, _non_static_mode
from paddle.fluid.framework import _current_expected_place as _get_device
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.distributed import fleet
from paddle.distributed.utils import get_logger
from paddle.distributed.passes import new_pass, PassContext

from .hepler import ProgramHelper
from ..collective import _get_global_env
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
from .cluster import Cluster, get_default_cluster
from .planner_v2 import Planner
from .parallelizer_v2 import Parallelizer
from .dist_op import DistributedOperator
from .dist_saver import DistributedSaver
<<<<<<< HEAD
from .dist_loader import (
    DistributedDataLoaderFromGenerator,
    DistributedDataLoader,
)
from .strategy import Strategy
from .process_group import new_process_group, get_all_process_groups
=======
from .dist_loader import NonIterableGeneratorLoader
from .utils import make_data_unshard, set_grad_var_shape
from .utils import print_program_with_dist_attr, to_list
from .process_group import new_process_group, get_all_process_groups, get_world_process_group
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
from .dist_context import DistributedContext, get_default_distributed_context
from .interface import CollectionNames, get_collection
from .cost.estimate_cost import get_cost_from_engine

from ..utils.log_utils import get_logger


class Engine:
<<<<<<< HEAD
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
            (paddle.optimizer.Optimizer, paddle.fluid.optimizer.Optimizer),
        ):
            raise TypeError(
                "'optimizer' must be object of class `paddle.optimizer.Optimizer`"
                " or `paddle.fluid.optimizer.Optimizer`."
            )
        self._optimizer = auto_utils.validate_opt(optimizer)
        self._orig_optimizer = copy.deepcopy(self._optimizer)

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
        if os.getenv("POD_NAME"):
            self._logger.info(
                "Distribute training by paddle.distributed.launch"
            )
            fleet.init(is_collective=True)
=======

    def __init__(self,
                 model=None,
                 inputs_spec=None,
                 labels_spec=None,
                 cluster=None,
                 strategy=None,
                 user_tuning_config=None):
        self.model = model
        self.inputs_spec = self._validate_spec(inputs_spec)
        self.labels_spec = self._validate_spec(labels_spec)
        self.cluster = cluster
        if self.cluster is None:
            self.cluster = get_default_cluster()
        self.strategy = strategy
        if self.strategy is None:
            self.strategy = fleet.DistributedStrategy()
        self._user_tuning_config = user_tuning_config
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

        self._executor = None
        self._cur_rank = paddle.distributed.get_rank()
        self._nranks = paddle.distributed.get_world_size()
        self._saver = DistributedSaver()

        self._orig_main_prog = static.default_main_program()
        self._orig_startup_prog = static.default_startup_program()
        self._orig_dist_context = get_default_distributed_context()
        self._dist_contexts = {}
<<<<<<< HEAD
        self._fwd_main_progs = {}
        self._fwd_dist_contexts = {}
=======
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self._serial_main_progs = {}
        self._serial_startup_progs = {}
        self._dist_main_progs = defaultdict(dict)  # dist main programs
        self._dist_startup_progs = defaultdict(dict)  # dist startup programs
        self._feed_vars = {}
        self._fetch_vars = {}
        self._planners = {}
<<<<<<< HEAD
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

        self.history = None

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
                "Data should be a Dataset or IterableDatset, but received {}.".format(
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
            elif isinstance(item, (Variable, core.VarBase, core.eager.Tensor)):
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
        if _non_static_mode() or self._dygraph_mode:
            raise ValueError("Only support static graph mode.")

        if inputs_spec:
            assert isinstance(
                inputs_spec, list
            ), "inputs should be list, but received {}".format(
                type(inputs_spec)
            )
            assert isinstance(
                inputs, list
            ), "inputs should be list, but received {}".format(type(inputs))
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
            ), "labels should be list, but received {}".format(type(labels))
            assert len(labels_spec) == len(
                labels
            ), "the number of `labels_spec` should be equal to `labels`'s."
            for label_spec, label in zip(labels_spec, labels):
                if label_spec.shape != label.shape:
                    label.desc.set_shape(label_spec.shape)

        return inputs, labels

    def _prepare_reader(self):
        dist_main_prog = self._dist_main_progs[self._mode][self._cur_rank]
        dist_context = self._dist_contexts[self._mode]
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
        new_reader_ops = []
        for idx in reversed(reader_op_indices):
            new_op_desc = dist_main_block.desc._prepend_op()
            new_op_desc.copy_from(dist_main_block.ops[idx].desc)
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

    def _prepare_feed(self, data, user_feeds, mode):
        feeds = {}
        if data is not None:
            if isinstance(data, (list, tuple)):
                if len(data) == 1 and isinstance(data[0], dict):
                    for name, data in data[0].items():
                        feeds[name] = data
                else:
                    raise ValueError("Unsupported data {}".format(data))
            elif isinstance(data, dict):
                for name, data in data.items():
                    feeds[name] = data
            else:
                raise ValueError("Unsupported data {}".format(data))
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

        if mode != "predict":
            _process_fetch_group("loss", self._fetch_vars[mode]["loss"])
        if mode != "predict":
            metrics = self._fetch_vars[mode]["metrics"]
            for i, var_list in enumerate(metrics):
                _process_fetch_group("metrics_" + str(i), var_list)
        if mode == "predict":
            _process_fetch_group("outputs", self._fetch_vars[mode]["outputs"])
        user_fetches_collection = [
            item[1] for item in get_collection(CollectionNames.FETCHES)
        ]
        var_list = (user_fetches_collection or []) + (user_fetches or [])
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
                logs["loss"] = outs[idx][0]
            group_idx += 1
            # logging metrics
            metric_vars = self._fetch_vars[mode]["metrics"]
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
        for name, var in collect_fetches:
            if var.name in fetch_names:
                idx = fetch_names.index(var.name)
                logs_fetch[name or var.name] = outs[idx]
        logs["fetches"] = logs_fetch
        return logs

    def _prepare_program(self, mode):
        # Do the build process
        self._build(mode)
        # Do the planning process
        self._plan(mode)
        # Do the parallel process
        self._parallel(mode)
        # Init comm and startup program
        self._initialize(mode)
        self._has_prepared[mode] = True
=======
        self._mode_init_states = {
            "train": False,
            "eval": False,
            "predict": False
        }
        self._dygraph_mode = False

    def prepare(self,
                optimizer=None,
                loss=None,
                gradient_scale=True,
                metrics=None,
                all_ranks=False):
        if optimizer and not isinstance(
                optimizer,
            (paddle.optimizer.Optimizer, paddle.fluid.optimizer.Optimizer)):
            raise TypeError(
                    "'optimizer' must be object of class `paddle.optimizer.Optimizer`" \
                        " or `paddle.fluid.optimizer.Optimizer`."
                )
        self._optimizer = optimizer
        self._all_ranks = all_ranks

        if loss and not isinstance(loss,
                                   paddle.nn.Layer) and not callable(loss):
            raise TypeError(
                "'loss' must be sub classes of `paddle.nn.Layer` or any callable function."
            )
        self._loss = loss

        metrics = metrics or []
        for metric in to_list(metrics):
            assert isinstance(metric, Metric), \
                "{} is not sub class of Metric".format(
                    metric.__class__.__name__)
        self._metrics = to_list(metrics)
        self._gradient_scale = gradient_scale
        self._planned_mode = None
        self._prepare_single_mode("train")

    def _prepare_single_mode(self, mode):

        self._build(mode)
        # Do the planning process
        self._plan(mode)

        # Do the Optimization tuning
        if self._user_tuning_config and mode == "train":
            self._optimization_tuning(mode)

        # Do the parallel process
        self._parallel(mode, self._all_ranks)

        # Init comm and startup program
        self._initialize(mode)
        self._mode_init_states[mode] = True
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

    def _build(self, mode):
        if _non_static_mode() or self._dygraph_mode:
            paddle.disable_static()
            self._dygraph_mode = True
            self._logger.info("Building model with 'to_static' method.")

<<<<<<< HEAD
            self.program_helper = ProgramHelper(
                self._model,
                self._loss,
                self._metrics,
                self._inputs_spec,
                self._labels_spec,
            )
            # build forward main program
            self.program_helper.build_program(mode)

            self.concrete_program = self.program_helper.concrete_program
            serial_main_prog = self.program_helper.main_program
            serial_startup_prog = self.program_helper.startup_program

            self._inputs = self.program_helper.input_vars
            self._labels = self.program_helper.label_vars
            outputs = self.program_helper.output_vars
            self._losses = self.program_helper.loss_vars
            metrics = self.program_helper.metric_vars
=======
            program_helper = ProgramHelper(self.model, self._loss,
                                           self._metrics, self.inputs_spec,
                                           self.labels_spec)
            # build forward main program
            program_helper.build_program(mode)

            self.concrete_program = program_helper.concrete_program
            serial_main_prog = program_helper.main_program
            serial_startup_prog = program_helper.startup_program

            inputs = program_helper.input_vars
            outputs = program_helper.output_vars
            labels = program_helper.label_vars
            losses = program_helper.loss_vars
            metrics = program_helper.metric_vars
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

            paddle.enable_static()
        else:
            # build program in static mode
            serial_main_prog = self._serial_main_progs.get(mode, None)
            if serial_main_prog is not None:
                return

<<<<<<< HEAD
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
=======
            losses = []
            metrics = []
            serial_main_prog = self._orig_main_prog.clone()
            serial_startup_prog = self._orig_startup_prog.clone()
            # FIXME to support grad clip
            with static.program_guard(serial_main_prog, serial_startup_prog), \
                utils.unique_name.guard():
                inputs_spec = self.inputs_spec
                labels_spec = self.labels_spec if self.labels_spec else []
                inputs = [s._create_feed_layer() for s in inputs_spec]
                labels = [s._create_feed_layer() for s in labels_spec]
                outputs = to_list(self.model(*inputs))
                if mode != "predict" and self._loss:
                    losses = to_list(self._loss(*(outputs + labels)))

                if mode != "predict":
                    for metric in self._metrics:
                        metrics.extend(
                            to_list(metric.compute(*(outputs + labels))))
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

        default_ctx = get_default_distributed_context()
        if not default_ctx.has_annotation:
            # We build the world process group because the data parallel
            # needs all ranks by default.
            new_process_group(list(range(self._nranks)))
            default_ctx.data_parallel = True
<<<<<<< HEAD
            self._inputs = [
                auto_utils.set_data_parallel(var) for var in self._inputs
            ]
            self._labels = [
                auto_utils.set_data_parallel(var) for var in self._labels
            ]

        feed_vars = {"inputs": self._inputs, "labels": self._labels}
=======

        feed_vars = {"inputs": inputs, "labels": labels}
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

        fetch_vars = {
            "outputs": flatten(outputs),
            "loss": self._losses,
            "metrics": metrics,
        }

<<<<<<< HEAD
        if mode != "train":
            serial_main_prog = serial_main_prog.clone(for_test=True)

        auto_utils.set_recompute_ckpts(self._model, self._strategy)
        self._dist_contexts[mode] = DistributedContext(
            serial_main_prog,
            serial_startup_prog,
            self._optimizer,
            self._losses,
            feed_vars,
            fetch_vars,
            self._cluster,
            self._strategy,
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
            self._tuning.to_dict(),
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
=======
        self._set_recompute_ckpts()
        self._dist_contexts[mode] = DistributedContext(
            serial_main_prog, serial_startup_prog, self._optimizer, losses,
            feed_vars, fetch_vars, self.cluster, self.strategy)
        self._dist_contexts[mode].gradient_scale = self._gradient_scale
        self._dist_contexts[mode]._dygraph_mode = self._dygraph_mode

    def _optimization_tuning(self, mode):

        self.mode = mode
        assert "batch_size" in self._user_tuning_config, "Optimization Tuning should provide with batch size."
        assert "dataset" in self._user_tuning_config, "Optimization Tuning should provide with dataset."
        batch_size = self._user_tuning_config["batch_size"]
        dataset = self._user_tuning_config["dataset"]
        dataset.dp_world_size = self._dp_world_size
        dataset.dp_rank = self._dp_rank

        from .tuner.optimization_tuner import OptimizationTuner
        self._optimization_tuner = OptimizationTuner(self._user_tuning_config,
                                                     self._dist_contexts[mode],
                                                     dataset,
                                                     self.inputs_spec,
                                                     self.labels_spec,
                                                     batch_size=batch_size,
                                                     rank=self._cur_rank)

        self._optimization_tuner.tune()

        if self._user_tuning_config["run_after_tuning"]:
            # update the strategy
            self._dist_contexts[
                mode]._strategy = self._optimization_tuner.get_best_config()
        else:
            return
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

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
<<<<<<< HEAD
        # TODO: check this feed_list
=======
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        feed_list = []
        for var in inputs_var + labels_var:
            if var.name in block.vars:
                feed_list.append(block.vars[var.name])

<<<<<<< HEAD
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
        # because we may use it to complete the annotation of the backwarkward and update.
        parallelizer = Parallelizer(
            mode, self._planners[mode].completer, self._dist_contexts[mode]
        )
=======
        self._dp_world_size, self._dp_rank = self._get_data_parallel_info(
            feed_list[0], self._dist_contexts[mode])

    def _parallel(self, mode, all_ranks):
        # Parallelize program based on the planner's results
        # For now, the completer has to be passed to the planner,
        # because we may use it to complete the annotation of the backwarkward and update.
        parallelizer = Parallelizer(mode, self._planners[mode].completer,
                                    self._dist_contexts[mode])
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
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
<<<<<<< HEAD
                assert (
                    op.type == ref_op.type
                ), "'{}' mode op '{}' is different with '{}' op '{}'. ".format(
                    mode, op.type, ref_mode, ref_op.type
                )
                ref_op_dist_attr = (
                    ref_dist_context.get_op_dist_attr_for_program(ref_op)
                )
=======
                assert op.type == ref_op.type, \
                    "'{}' mode op '{}' is different with '{}' op '{}'. ".format(mode, op.type, ref_mode, ref_op.type)
                ref_op_dist_attr = ref_dist_context.get_op_dist_attr_for_program(
                    ref_op)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
                dist_context.set_op_dist_attr_for_program(op, ref_op_dist_attr)

    def _initialize(self, mode):
        # Get the current content from the distributed context
        self._serial_main_progs[mode] = self._dist_contexts[
<<<<<<< HEAD
            mode
        ].serial_main_program
        self._serial_startup_progs[mode] = self._dist_contexts[
            mode
        ].serial_startup_program
        self._dist_main_progs[mode] = self._dist_contexts[
            mode
        ].dist_main_programs
        self._dist_startup_progs[mode] = self._dist_contexts[
            mode
        ].dist_startup_programs
        self._feed_vars[mode] = self._dist_contexts[mode].serial_feed_vars
        self._fetch_vars[mode] = self._dist_contexts[mode].serial_fetch_vars
        self._optimizer = self._dist_contexts[mode]._serial_optimizer
=======
            mode].serial_main_program
        self._serial_startup_progs[mode] = self._dist_contexts[
            mode].serial_startup_program
        self._dist_main_progs[mode] = self._dist_contexts[
            mode].dist_main_programs
        self._dist_startup_progs[mode] = self._dist_contexts[
            mode].dist_startup_programs
        self._feed_vars[mode] = self._dist_contexts[mode].serial_feed_vars
        self._fetch_vars[mode] = self._dist_contexts[mode].serial_fetch_vars
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

        if self._nranks > 1:
            # Traverse different rank programs and traverse each op of them,
            # instantiate communication by process_mapping.
            all_process_groups = get_all_process_groups()
<<<<<<< HEAD
            cur_rank = self._cur_rank
            # NOTE: After the implementation of the unified dynamic and static communication group initialization mode in the future, the initialization logic of full mode will be removed because port occupation error may occur.
            if self._strategy.auto_mode == "full":
                auto_utils.initialize_pg_in_full_mode(
                    all_process_groups, cur_rank
                )
            else:
                for process_group in all_process_groups:
                    if cur_rank not in process_group.ranks:
                        continue
                    process_group.instantiate()

        place = _get_device()
        if isinstance(place, fluid.CUDAPlace):
            place = fluid.CUDAPlace(ParallelEnv().dev_id)

        if self._strategy.seed:
            paddle.seed(self._strategy.seed + self._dp_ranks[0])
            np.random.seed(self._strategy.seed + self._dp_ranks[0])
            random.seed(self._strategy.seed + self._dp_ranks[0])

        if self._dygraph_mode:
            dist_context = self._dist_contexts[mode]
            dist_main_program = self._dist_main_progs[mode][self._cur_rank]
            self.program_helper.init(dist_main_program, place, dist_context)

=======

            # NOTE: add the comm init control in the future for auto search
            for process_group in all_process_groups:
                if self._cur_rank not in process_group.ranks:
                    continue
                process_group.instantiate()

        self._place = _get_device()
        if isinstance(self._place, fluid.CUDAPlace):
            self._place = fluid.CUDAPlace(ParallelEnv().dev_id)

        if self._dygraph_mode:
            paddle.disable_static()
            main_program = self._dist_main_progs[mode][self._cur_rank]
            for param in self.concrete_program.parameters:
                # create var in scope and share parameters to scope
                if param.name not in main_program.global_block().vars:
                    continue
                # get param_var's dist_attr
                var = main_program.global_block().vars[param.name]
                var_dist_attr = self._dist_contexts[
                    mode].get_tensor_dist_attr_for_program(var)
                dist_attr = {
                    "dims_mapping": var_dist_attr.dims_mapping,
                    "process_shape": var_dist_attr.process_mesh.topology,
                    "process_group": var_dist_attr.process_mesh.processes
                }
                # slice param_value with dist_attr
                # share sliced_param_value with param_tensor in global_scope
                from .converter import Converter
                param_tensor = global_scope().var(param.name).get_tensor()
                sliced_param = Converter.slice_with_dist_attr(
                    param.numpy(), dist_attr)
                shared_tensor = paddle.to_tensor(sliced_param,
                                                 place=self._place)
                param_tensor._share_data_with(
                    shared_tensor.value().get_tensor())
            paddle.enable_static()

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        if self._executor is None:
            self._executor = paddle.static.Executor(place)
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

<<<<<<< HEAD
            if hasattr(self, "_state_dict") and hasattr(self, "_dist_attr"):
                self._set_state_dict(
                    mode, self._strict, self._state_dict, self._dist_attr
                )

        if self._strategy.reinit:
            self._logger.info("NOTE: parameters will be re-initialized.")
            dist_startup_prog = self._dist_startup_progs[mode][self._cur_rank]
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
        if not self._has_prepared[self._mode]:
            self._prepare_program(self._mode)
        else:
            self._switch_mode(self._mode)

        train_dataloader = self._prepare_dataloader_from_generator(
            dataset=train_data,
            capacity=70,
            iterable=False,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            collate_fn=collate_fn,
        )

        fetch_names, fetch_indices = self._prepare_fetch(None, mode=self._mode)

        cbks = config_callbacks(
            callbacks,
            engine=self,
            batch_size=batch_size,
            epochs=epochs,
            steps=train_dataloader._steps,
            log_freq=log_freq,
            save_freq=save_freq,
            save_dir=save_dir,
            verbose=verbose,
            metrics=self._metrics_name(),
            acc_step=self._k_steps,
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
                lr = auto_utils.get_lr(self._optimizer)
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
        if not self._has_prepared[self._mode]:
            self._prepare_program(self._mode)
        else:
            self._switch_mode(self._mode)

        valid_dataloader = self._prepare_dataloader_from_generator(
            dataset=valid_data,
            capacity=70,
            iterable=False,
            batch_size=batch_size,
            steps_per_epoch=steps,
            collate_fn=collate_fn,
        )

        fetch_names, fetch_indices = self._prepare_fetch(None, mode=self._mode)

        cbks = config_callbacks(
            callbacks,
            engine=self,
            batch_size=batch_size,
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
        if not self._has_prepared[self._mode]:
            self._prepare_program(self._mode)
        else:
            self._switch_mode(self._mode)

        test_dataloader = self._prepare_dataloader_from_generator(
            dataset=test_data,
            capacity=70,
            iterable=False,
            batch_size=batch_size,
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
        if not self._has_prepared[self._mode]:
            self._prepare_program(self._mode)
        else:
            self._switch_mode(self._mode)

        dataloader = self._prepare_dataloader(
            dataset,
            return_list=False,
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
            batch_size=batch_size,
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
            self._prepare_program(self._mode)
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

        if self._strategy.gradient_merge and batch_size is not None:
            assert (
                batch_size % self._k_steps == 0
            ), "Requires batch_size:[{}] to be divisible by k_steps:[{}].".format(
                batch_size, self._k_steps
            )
            batch_size //= self._k_steps

        dist_main_prog = self._dist_main_progs[self._mode][self._cur_rank]
        dist_startup_prog = self._dist_startup_progs[self._mode][self._cur_rank]
        dist_main_block = dist_main_prog.global_block()

        # NOTE: Get feed_list, then insert dataloader op with sharded var shape.
        # Cause predict_program does not contain labels var,
        # then we will add labels var from serial_program to dist_program,
        # that maintains the length of feed_list equal to the length of dataset's values.
        inputs_var = self._feed_vars[self._mode]["inputs"]
        labels_var = self._feed_vars[self._mode]["labels"]
=======
            if self.strategy.amp and self.strategy.amp_configs['use_pure_fp16']:
                # from paddle.fluid.contrib.mixed_precision.fp16_utils import cast_parameters_to_fp16
                def cast_parameters_to_fp16(place,
                                            program,
                                            scope=None,
                                            to_fp16_var_names=None):
                    """
                    Traverse all parameters in the whole model and set them to the FP16 data type.
                    Whereas, this function will keep parameters of batchnorms in FP32.
                    Args:
                        place(fluid.CPUPlace|fluid.CUDAPlace): `place` is used to restore the FP16 weight tensors.
                        program (Program): The used program.
                        scope(fluid.Scope, optional): `scope` is used to get the FP32 weight tensor values.
                                                    Default is None.
                        to_fp16_var_names(set|list, optional): The data types of vars in `to_fp16_var_names`
                                                            will be set to FP16. Usually, it is the returned
                                                            value of `cast_model_to_fp16` API.
                    """
                    from paddle.framework import core
                    import numpy as np
                    all_parameters = []
                    for block in program.blocks:
                        all_parameters.extend(block.all_parameters())

                    var_scope = scope if scope else paddle.static.global_scope()
                    for param in all_parameters:
                        if param.dtype == core.VarDesc.VarType.FP16:
                            param_t = var_scope.find_var(
                                param.name).get_tensor()
                            data = np.array(param_t)
                            param_t.set(np.float16(data), place)

                cast_parameters_to_fp16(self._place, prune_startup_prog)

    def fit(self,
            train_data,
            batch_size=1,
            epochs=1,
            fetches=None,
            steps_per_epoch=None,
            use_program_cache=False,
            return_numpy=True):
        # TODO: callbacks
        # TODO: evaluate after training

        if not self._mode_init_states['train']:
            raise Exception(
                "train program is not initialized yet, please call engine.prepare() before calling fit() funtion."
            )

        self.mode = 'train'
        assert self.mode in self._dist_main_progs, \
            "train model is not ready, please call `engine.prepare()` first."
        train_dataloader = self._create_dataloader(train_data, batch_size,
                                                   epochs, steps_per_epoch)

        usr_fetch = self._validate_fetches(fetches)
        fetch_loss = self._validate_fetches(self.fetch_vars["loss"])
        fetch_list, fetch_map = self._fetch_map(fetch_loss, usr_fetch)
        for epoch in range(epochs):
            train_logs = {"epoch": epoch}
            for step, _ in enumerate(train_dataloader):
                outs = self._executor.run(self.main_program,
                                          fetch_list=fetch_list,
                                          use_program_cache=use_program_cache,
                                          return_numpy=return_numpy)
                train_logs["step"] = step
                # inner fetches
                if fetch_loss:
                    train_logs["train_loss"] = outs[0][0]
                # user fetches
                user_outs = outs[len(fetch_loss):]
                user_fetch_list = fetch_list[len(fetch_loss):]
                for i, out in enumerate(user_outs):
                    train_logs["train_" + fetch_map[user_fetch_list[i]]] = out
                self._logger.info(train_logs)

    def evaluate(self,
                 eval_data,
                 batch_size=1,
                 fetches=None,
                 use_program_cache=False,
                 return_numpy=True):
        self.mode = 'eval'
        if not self._mode_init_states[self.mode]:
            self._prepare_single_mode(self.mode)

        assert self.mode in self._dist_main_progs, \
            "eval model is not ready, please call `engine.prepare()` first."
        eval_dataloader = self._create_dataloader(eval_data, batch_size)

        usr_fetch = self._validate_fetches(fetches)
        fetch_loss = self._validate_fetches(self.fetch_vars["loss"])
        fetch_metrics = self._validate_fetches(self.fetch_vars["metrics"])
        inner_fetch = dict(fetch_loss, **fetch_metrics)
        fetch_list, fetch_map = self._fetch_map(inner_fetch, usr_fetch)

        for step, _ in enumerate(eval_dataloader):
            eval_logs = {"step": step}
            outs = self._executor.run(self.main_program,
                                      fetch_list=fetch_list,
                                      use_program_cache=use_program_cache,
                                      return_numpy=return_numpy)
            # inner fetches
            if fetch_loss:
                eval_logs["eval_loss"] = outs[0][0]
            # Metric
            if fetch_metrics:
                metric_out = outs[len(fetch_loss):len(inner_fetch)]
                for metric in self._metrics:
                    metric.update(*metric_out)
                    results = metric.accumulate()
                    for i, res in enumerate(to_list(results)):
                        eval_logs["eval_" + metric.name()[i]] = res
            # usr fetches
            usr_outs = outs[len(inner_fetch):]
            usr_fetch_list = fetch_list[len(inner_fetch):]
            for i, out in enumerate(usr_outs):
                eval_logs["eval_" + fetch_map[usr_fetch_list[i]]] = out
            # logger
            self._logger.info(eval_logs)

    def predict(self,
                test_data,
                batch_size=1,
                fetches=None,
                use_program_cache=False,
                return_numpy=True):
        self.mode = 'predict'
        if not self._mode_init_states[self.mode]:
            self._prepare_single_mode(self.mode)

        assert self.mode in self._dist_main_progs, \
            "predict model is not ready, please call `engine.prepare()` first."
        test_dataloader = self._create_dataloader(test_data, batch_size)

        usr_fetch = self._validate_fetches(fetches)
        fetch_outputs = self._validate_fetches(self.fetch_vars["outputs"])
        fetch_list, fetch_map = self._fetch_map(fetch_outputs, usr_fetch)

        outputs = []
        for step, _ in enumerate(test_dataloader):
            predict_logs = {"step": step}
            outs = self._executor.run(self.main_program,
                                      fetch_list=fetch_list,
                                      use_program_cache=use_program_cache,
                                      return_numpy=return_numpy)
            outputs.append(outs[:len(fetch_outputs)])
            for i, out in enumerate(outs):
                predict_logs["pred_" + fetch_map[fetch_list[i]]] = out
            self._logger.info(predict_logs)

        return outputs

    def _create_dataloader(self,
                           dataset,
                           batch_size,
                           epochs=1,
                           steps_per_epoch=None):
        dist_main_prog = self._dist_main_progs[self.mode][self._cur_rank]
        dist_startup_prog = self._dist_startup_progs[self.mode][self._cur_rank]
        dist_context = self._dist_contexts[self.mode]
        dist_main_block = dist_main_prog.global_block()

        # NOTE: Get feed_list from dist_program, then insert dataloader op
        # with sharded var shape. Because predict_program does not contain
        # labels var, so we will filter dataset's value with length of feed_list.
        inputs_var = self._feed_vars[self.mode]["inputs"]
        labels_var = self._feed_vars[self.mode]["labels"]
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        feed_list = []
        for var in inputs_var + labels_var:
            if var.name in dist_main_block.vars:
                feed_list.append(dist_main_block.vars[var.name])
<<<<<<< HEAD
            else:
                copy_var = dist_main_block._clone_variable(var, var.persistable)
                copy_var.desc.set_original_id(var.desc.original_id())
                feed_list.append(copy_var)
=======

        # remove the first three ops if multi run fit/evaluate/predict
        op_size = len(dist_main_block.ops)
        if dist_main_block.ops[0].type == 'create_py_reader':
            op_size -= 3
            for _ in range(3):
                dist_main_block._remove_op(0, sync=False)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

        # insert read op at the end of program
        places = paddle.static.cuda_places()
        with static.program_guard(dist_main_prog, dist_startup_prog):
<<<<<<< HEAD
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

        if self._strategy.gradient_merge and batch_size is not None:
            assert (
                batch_size % self._k_steps == 0
            ), "Requires batch_size:[{}] to be divisible by k_steps:[{}].".format(
                batch_size, self._k_steps
            )
            batch_size //= self._k_steps

        dist_main_prog = self._dist_main_progs[self._mode][self._cur_rank]
        dist_startup_prog = self._dist_startup_progs[self._mode][self._cur_rank]
        dist_main_block = dist_main_prog.global_block()

        # NOTE: Get feed_list, then insert dataloader op with sharded var shape.
        # Cause predict_program does not contain labels var,
        # then we will add labels var from serial_program to dist_program,
        # that maintains the length of feed_list equal to the length of dataset's values.
        inputs_var = self._feed_vars[self._mode]["inputs"]
        labels_var = self._feed_vars[self._mode]["labels"]
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
            )
        self._prepare_reader()
=======
            dataloader = NonIterableGeneratorLoader(
                dataset,
                feed_list,
                places,
                batch_size,
                epochs,
                steps_per_epoch,
                data_parallel_world_size=self._dp_world_size,
                data_parallel_rank=self._dp_rank)

        # move read op from the end of program to the start of program
        new_op_size = len(dist_main_block.ops)
        for _ in range(new_op_size - 1, op_size - 1, -1):
            op = dist_main_block.ops[new_op_size - 1]
            new_op_desc = dist_main_block.desc._prepend_op()
            new_op_desc.copy_from(op.desc)
            new_op = Operator(dist_main_block,
                              new_op_desc,
                              type=new_op_desc.type())
            dist_main_block.ops.insert(0, new_op)
            dist_op = DistributedOperator(new_op)
            dist_context.add_dist_op_for_program(dist_op)
        for _ in range(new_op_size - op_size):
            dist_main_block._remove_op(new_op_size, sync=False)
        dist_main_block._sync_with_cpp()
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        return dataloader

    def _tune(self, tune_data, tune_sample_split=None, batch_size=1):
        self._mode = 'train'
        self._inputs_spec, self._labels_spec = self._prepare_data_spec(
            tune_data, tune_sample_split, batch_size
        )
        self._optimization_tuning(self._mode, tune_data, batch_size)

    def _validate_spec(self, specs):
        specs = auto_utils.to_list(specs)
        self._k_steps = self._strategy.gradient_merge.k_steps
        if specs is not None:
            for i, spec in enumerate(specs):
                if not isinstance(spec, InputSpec):
                    raise TypeError(
                        "'spec' must be object of class `paddle.static.InputSpec`."
                    )
                if spec.name is None:
                    raise ValueError(
<<<<<<< HEAD
                        "Requires Input[{}].name != None, but receive `None` with {}.".format(
                            i, spec
                        )
                    )
                if self._k_steps > 1:
                    shape = list(spec.shape)
                    assert (
                        shape[0] % self._k_steps == 0
                    ), "Requires batch_size[{}] to be divisible by k_steps[{}].".format(
                        spec.shape[0], self._k_steps
                    )
                    shape[0] //= self._k_steps
                    spec.shape = shape
        return specs or []

    def _validate_vars(self, vars):
        vars = auto_utils.to_list(vars)
        if vars is not None:
            for i, var in enumerate(vars):
                if not isinstance(var, Variable):
                    raise TypeError("'var' must be a `Variable`.")
        return vars or []
=======
                        "Requires Input[{}].name != None, but receive `None` with {}."
                        .format(i, spec))
        return specs
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

    def _is_local_var(self, var):
        var_name = _to_name_str(var)
        return var_name in self.main_program.global_block().vars

<<<<<<< HEAD
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
            mode in self._dist_main_progs
        ), "{} model is not ready, please call `prepare()` first.".format(mode)
        self.to_mode(mode)
        self._optimizer = self._dist_contexts[mode]._serial_optimizer

    def to_mode(self, mode):
        assert mode in [
            "train",
            "eval",
            "predict",
        ], "mode {} should be one of ['train', 'eval', 'predict']".format(mode)
        self._mode = mode

    def _set_state_dict(self, mode, strict, state_dict, dist_attr):
        program = self._dist_main_progs[mode][self._cur_rank]
        dist_context = self._dist_contexts[mode]
        cur_dist_attr = auto_utils.get_dist_attr(program, dist_context)
        converter = Converter(state_dict, dist_attr, cur_dist_attr)
        state_dict = converter.convert(strict=strict)
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
=======
    def _validate_fetches(self, fetches):
        # 1. Check user-defined fetches type
        # 2. Prepare fetches_dict like {user_defined_name: var_name}
        if not fetches:
            return {}
        if isinstance(fetches, dict):
            fetch_var_names = list(map(_to_name_str, fetches.values()))
            fetches_dict = dict(zip(fetch_var_names, list(fetches.keys())))
        elif isinstance(fetches, list):
            fetch_var_names = list(map(_to_name_str, fetches))
            fetches_dict = dict(zip(fetch_var_names, fetch_var_names))
        else:
            raise TypeError("'fetches' only support 'dict' and 'list', "
                            "but got '{}'".format(str(type(fetches))))
        return dict(
            filter(lambda x: self._is_local_var(x[0]), fetches_dict.items()))

    def _fetch_map(self, inner_fetch, usr_fetch):
        # replace inner fetch name if usr set for it
        for iname in inner_fetch:
            if iname in usr_fetch:
                inner_fetch[iname] = usr_fetch[iname]
                usr_fetch.pop(iname)
        fetches = dict(inner_fetch, **usr_fetch)
        return list(fetches.keys()), fetches

    def _get_data_parallel_info(self, var, dist_context):
        # get data parallel world size and current data parallel rank
        from .utils import _get_comm_group, _get_corresponding_rank

        tensor_dist_attr = dist_context.get_tensor_dist_attr_for_program(var)
        process_mesh = tensor_dist_attr.process_mesh
        dims_mapping = tensor_dist_attr.dims_mapping

        if self._cur_rank not in process_mesh.processes:
            rank_id = _get_corresponding_rank(dist_context, process_mesh,
                                              self._cur_rank)
        else:
            rank_id = self._cur_rank
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

        Examples:

            .. code-block:: python
                import paddle
                import paddle.vision.transforms as T
                from paddle.distributed.fleet import auto
                from paddle.vision.datasets import MNIST

<<<<<<< HEAD
                transform = T.Compose([
                    T.Transpose(),
                    T.Normalize([127.5], [127.5])
                ])
                train_dataset = MNIST(mode='train', transform=transform)
=======
    def _set_recompute_ckpts(self):
        # NOTE hack to enable recompute in engine api for GPT-3
        # TODO support more PaddleNLP/CV models here

        config = self.strategy.recompute_configs

        # extract ckpts by specific model
        self.model
        if isinstance(self.model, paddle.nn.Layer):
            if hasattr(
                    self.model, "model"
            ) and self.model.model.__class__.__name__ == 'GPTForPretraining':
                exact_ckpts = self.model.model.gpt.checkpoints
        else:
            exact_ckpts = config["checkpoints"]

        # modify strategy
        if self.strategy.recompute:
            config["checkpoints"] = exact_ckpts[:]
            self.strategy.recompute_configs = config
            logs = {
                'Model Class': self.model.model.__class__.__name__,
                'Applied Recompute ckpts': exact_ckpts
            }
            self._logger.info(logs)

    def save(self, path, training=True, mode=None):
        if not mode:
            mode = self.mode
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

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
<<<<<<< HEAD
            assert self._mode in self._serial_main_progs
            serial_program = self._serial_main_progs[self._mode]
            dist_main_prog = self._dist_main_progs[self._mode][self._cur_rank]
            dist_context = self._dist_contexts[self._mode]
            self._saver.save(
                path,
                serial_program=serial_program,
                dist_main_program=dist_main_prog,
                dist_context=dist_context,
            )
        else:
            assert "predict" in self._dist_main_progs
            feed_vars = self._feed_vars["predict"]['inputs']
            fetch_vars = self._fetch_vars["predict"]['outputs']
            dist_main_prog = self._dist_main_progs["predict"][self._cur_rank]
            self._saver.save_inference_model(
                path,
                feed_vars,
                fetch_vars,
                self._executor,
                program=dist_main_prog,
            )
=======
            assert 'train' in self._serial_main_progs, \
                "training model is not ready, please call `engine.prepare()` first."
            serial_program = self._serial_main_progs["train"]
            dist_main_prog = self._dist_main_progs["train"][self._cur_rank]
            dist_context = self._dist_contexts["train"]
            self._saver.save(path,
                             serial_program=serial_program,
                             dist_main_program=dist_main_prog,
                             dist_context=dist_context)
        else:
            assert mode, "Please set the 'mode' you want to save."
            feed_vars = self._feed_vars[mode]['inputs']
            fetch_vars = self._fetch_vars[mode]['outputs']
            dist_main_prog = self._dist_main_progs[mode][self._cur_rank]
            self._saver.save_inference_model(path,
                                             feed_vars,
                                             fetch_vars,
                                             self._executor,
                                             program=dist_main_prog)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

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

<<<<<<< HEAD
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
            if _non_static_mode() or self._dygraph_mode:
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

=======
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
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
<<<<<<< HEAD
        return self._serial_startup_progs[self._mode]

    @property
    def fetch_vars(self):
        return self._fetch_vars[self._mode]

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels
=======
        return self._serial_startup_progs[self.mode]

    @property
    def fetch_vars(self):
        return self._fetch_vars[self.mode]
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
