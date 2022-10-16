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
import logging
import random
import numpy as np
from collections import defaultdict
import socket

import paddle
import paddle.utils as utils

from paddle import fluid, profiler, static
from paddle.metric import Metric
from paddle.static import InputSpec
from paddle.fluid import core
from paddle.fluid import Variable
from paddle.fluid.layers.utils import flatten
from paddle.fluid.executor import global_scope, _to_name_str
from paddle.fluid.framework import Operator, _non_static_mode
from paddle.fluid.framework import _current_expected_place as _get_device
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.distributed import fleet

from .converter import Converter
from ..collective import _get_global_env
from .helper import ProgramHelper
from .cluster import Cluster, get_default_cluster
from .planner_v2 import Planner
from .parallelizer_v2 import Parallelizer
from .dist_op import DistributedOperator
from .dist_saver import DistributedSaver
from .dist_loader import DistributedDataLoaderFromGenerator, DistributedDataLoader
from .utils import to_list, get_logger, get_dist_attr
from .process_group import new_process_group, get_all_process_groups
from .dist_context import DistributedContext, get_default_distributed_context
from .strategy import Strategy
from .interface import CollectionNames, get_collection


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

    def __init__(self,
                 model=None,
                 loss=None,
                 optimizer=None,
                 metrics=None,
                 cluster=None,
                 strategy=None):

        if model and not isinstance(model,
                                    paddle.nn.Layer) and not callable(model):
            raise TypeError(
                "'model must be sub classes of `paddle.nn.Layer` or any callable function."
            )
        self._model = model

        # if loss and not isinstance(loss,
        #                            paddle.nn.Layer) and not callable(loss):
        #     raise TypeError(
        #         "'loss' must be sub classes of `paddle.nn.Layer` or any callable function."
        #     )
        self._loss = loss

        if optimizer and not isinstance(
                optimizer,
            (paddle.optimizer.Optimizer, paddle.fluid.optimizer.Optimizer)):
            raise TypeError(
                "'optimizer' must be object of class `paddle.optimizer.Optimizer`"
                " or `paddle.fluid.optimizer.Optimizer`.")
        self._optimizer = self._validate_opt(optimizer)

        metrics = metrics or []
        for metric in to_list(metrics):
            assert isinstance(metric, Metric), \
                "{} is not sub class of Metric".format(
                    metric.__class__.__name__)
        self._metrics = to_list(metrics)

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

        if os.getenv("POD_NAME"):
            print("Distribute training by paddle.distributed.launch",
                  flush=True)
            fleet.init(is_collective=True)

        self._executor = None
        self._cur_rank = paddle.distributed.get_rank()
        self._nranks = paddle.distributed.get_world_size()
        self._saver = DistributedSaver()

        self._logger = get_logger(logging.INFO)

        self._orig_main_prog = static.default_main_program()
        self._orig_startup_prog = static.default_startup_program()
        self._orig_dist_context = get_default_distributed_context()
        self._dist_contexts = {}
        self._serial_main_progs = {}
        self._serial_startup_progs = {}
        self._dist_main_progs = defaultdict(dict)  # dist main programs
        self._dist_startup_progs = defaultdict(dict)  # dist startup programs
        self._feed_vars = {}
        self._fetch_vars = {}
        self._planners = {}
        self._has_prepared = {"train": False, "eval": False, "predict": False}
        self._has_prepared_reader = {
            "train": False,
            "eval": False,
            "predict": False
        }
        self._inputs_spec = []
        self._labels_spec = []
        self._inputs = []
        self._labels = []

        self._skip_build = False
        self._outside_dataloader = False
        self._planned_mode = None
        self._dygraph_mode = False
        self._tuning = self._strategy.tuning

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
            raise ValueError(
                "Data should be a Dataset or IterableDatset, but received {}.".
                format(type(data).__name__))
        inputs = to_list(inputs)
        labels = to_list(labels)

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
                _adjust_item_spec(num_shards, spec)
                spec = InputSpec.from_tensor(item, name)
                if batch_size is None:
                    specs.append(spec)
                else:
                    specs.append(spec.batch(batch_size))
            else:
                specs.append(InputSpec([batch_size], type(item), name))

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

    def _prepare_data_tensor(self,
                             inputs_spec,
                             labels_spec,
                             inputs=None,
                             labels=None):
        if _non_static_mode() or self._dygraph_mode:
            return None, None
        inputs_spec = inputs_spec if inputs_spec else []
        labels_spec = labels_spec if labels_spec else []
        if inputs_spec:
            assert isinstance(inputs_spec, list), \
                "inputs should be list, but received {}".format(type(inputs_spec))
            if inputs is None:
                inputs = [s._create_feed_layer() for s in inputs_spec]
            else:
                assert isinstance(inputs, list), \
                    "inputs should be list, but received {}".format(type(inputs))
                for input_spec, input in zip(inputs_spec, inputs):
                    if input_spec.shape != input.shape:
                        input.desc.set_shape(input_spec.shape)
        if labels_spec:
            assert isinstance(labels_spec, list), \
                "labels should be list, but received {}".format(type(labels_spec))
            if labels is None:
                labels = [s._create_feed_layer() for s in labels_spec]
            else:
                assert isinstance(labels, list), \
                    "labels should be list, but received {}".format(type(labels))
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
            "create_py_reader", "create_double_buffer_reader", "read"
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
            new_op = Operator(dist_main_block,
                              new_op_desc,
                              type=new_op_desc.type())
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
            assert isinstance(user_feeds, dict), \
                "user_feeds must be a dict, but receive {}".format(type(user_feeds).__name__)
            for name, data in user_feeds.items():
                feeds[name] = data
        return feeds

    def _prepare_fetch(self, user_fetches, mode):
        if user_fetches is not None:
            assert isinstance(user_fetches, list), \
                "user_fetches must be a list, but receive {}".format(type(user_fetches).__name__)
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

    def _prepare_logger(self,
                        outs,
                        epoch=None,
                        step=None,
                        lr=None,
                        fetch_names=None,
                        fetch_indices=None,
                        profiler_log="",
                        mode=None):
        logs = "[{}] ".format(mode)
        if epoch is not None:
            logs += "epoch: {:d} ".format(epoch)
        if step is not None:
            logs += "step: {:d} ".format(step)
        if lr is not None:
            logs += "lr: {:5e} ".format(lr)
        group_idx = 0
        # logging loss
        if mode != "predict":
            loss_indices = fetch_indices[group_idx]
            for idx in loss_indices:
                logs += "loss: {:8f} ".format(outs[idx][0])
            group_idx += 1
        # logging metrics
        if mode != "predict":
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
                        for i, res in enumerate(to_list(results)):
                            logs += "{}: {:8f} ".format(metric.name()[i], res)
                    group_idx += 1
        # Skip logging outputs
        if mode == "predict":
            group_idx += 1
        # logging user fetches
        fetches_logging = get_collection(CollectionNames.LOGGING)
        for name, var in fetches_logging:
            if var.name in fetch_names:
                idx = fetch_names.index(var.name)
                # Use the user defined name for logging
                logs += "{}: {} ".format(name, outs[idx])
        logs += profiler_log
        self._logger.info(logs)

    def _prepare_history(self, outs, fetch_indices=None, mode=None):
        history = {}
        group_idx = 0
        # store loss
        if mode != "predict":
            loss_indices = fetch_indices[group_idx]
            loss_values = []
            for idx in loss_indices:
                loss_values.append(outs[idx][0])
            history["loss"] = loss_values
            group_idx += 1
        # store metrics
        if mode != "predict":
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
                        history[tuple(metric.name())] = to_list(results)
                    group_idx += 1
        # store outputs
        if mode == "predict":
            outputs_indices = fetch_indices[group_idx]
            outputs_values = []
            for idx in outputs_indices:
                outputs_values.append(outs[idx])
            history["outputs"] = outputs_values
            group_idx += 1
        # store user fetches
        fetches_indices = fetch_indices[group_idx]
        fetches_values = []
        for idx in fetches_indices:
            fetches_values.append(outs[idx])
        history["fetches"] = fetches_values
        return history

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

    def _build(self, mode):
        if _non_static_mode() or self._dygraph_mode:
            paddle.disable_static()
            self._dygraph_mode = True
            self._logger.info("Building model with 'to_static' method.")

            inputs_spec = self._inputs_spec
            labels_spec = self._labels_spec if self._labels_spec else []
            self.program_helper = ProgramHelper(self._model, self._loss,
                                                self._metrics, inputs_spec,
                                                labels_spec)
            # build forward main program
            self.program_helper.build_program(mode)

            self.concrete_program = self.program_helper.concrete_program
            serial_main_prog = self.program_helper.main_program
            serial_startup_prog = self.program_helper.startup_program

            inputs = self.program_helper.input_vars
            outputs = self.program_helper.output_vars
            labels = self.program_helper.label_vars
            losses = self.program_helper.loss_vars
            metrics = self.program_helper.metric_vars

            self._inputs = inputs
            self._labels = labels

            paddle.enable_static()
        else:
            # build program in static mode
            serial_main_prog = self._serial_main_progs.get(mode, None)
            if serial_main_prog is not None:
                return

            outputs = []
            losses = []
            metrics = []
            inputs = self._inputs if self._inputs else []
            labels = self._labels if self._labels else []
            serial_main_prog = self._orig_main_prog.clone()
            serial_startup_prog = self._orig_startup_prog.clone()
            if not self._skip_build:
                with static.program_guard(serial_main_prog, serial_startup_prog), \
                    utils.unique_name.guard():
                    outputs = to_list(self._model(*inputs))
                    if mode != "predict" and self._loss:
                        losses = to_list(self._loss(*(outputs + labels)))

                    if mode != "predict" and (outputs or labels):
                        for metric in self._metrics:
                            metrics.append(
                                to_list(metric.compute(*(outputs + labels))))
            else:
                losses = to_list(self._loss)

        default_ctx = get_default_distributed_context()
        if not default_ctx.has_annotation:
            # We build the world process group because the data parallel
            # needs all ranks by default.
            new_process_group(list(range(self._nranks)))
            default_ctx.data_parallel = True

        feed_vars = {"inputs": inputs, "labels": labels}

        fetch_vars = {
            "outputs": flatten(outputs),
            "loss": losses,
            "metrics": metrics
        }

        if mode != "train":
            serial_main_prog = serial_main_prog.clone(for_test=True)

        self._set_recompute_ckpts()
        self._dist_contexts[mode] = DistributedContext(
            serial_main_prog, serial_startup_prog, self._optimizer, losses,
            feed_vars, fetch_vars, self._cluster, self._strategy)
        self._dist_contexts[mode].gradient_scale = self._strategy.gradient_scale

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
        self._optimization_tuner = OptimizationTuner(self._tuning.to_dict(),
                                                     self._dist_contexts[mode],
                                                     dataset,
                                                     self._inputs_spec,
                                                     self._labels_spec,
                                                     batch_size=batch_size,
                                                     rank=self._cur_rank)

        self._optimization_tuner.tune()

        if self._tuning.run_after_tuning:
            # update the strategy
            self._dist_contexts[
                mode]._strategy = self._optimization_tuner.get_best_config()

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
            dp_world_size, dp_rank = self._get_input_split_info(
                feed_var, self._dist_contexts[mode])
            self._dp_world_sizes.append(dp_world_size)
            self._dp_ranks.append(dp_rank)

    def _parallel(self, mode, all_ranks=False):
        # Parallelize program based on the planner's results
        # For now, the completer has to be passed to the planner,
        # because we may use it to complete the annotation of the backwarkward and update.
        parallelizer = Parallelizer(mode, self._planners[mode].completer,
                                    self._dist_contexts[mode])
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
                assert op.type == ref_op.type, \
                    "'{}' mode op '{}' is different with '{}' op '{}'. ".format(mode, op.type, ref_mode, ref_op.type)
                ref_op_dist_attr = ref_dist_context.get_op_dist_attr_for_program(
                    ref_op)
                dist_context.set_op_dist_attr_for_program(op, ref_op_dist_attr)

    def _initialize(self, mode):
        # Get the current content from the distributed context
        self._serial_main_progs[mode] = self._dist_contexts[
            mode].serial_main_program
        self._serial_startup_progs[mode] = self._dist_contexts[
            mode].serial_startup_program
        self._dist_main_progs[mode] = self._dist_contexts[
            mode].dist_main_programs
        self._dist_startup_progs[mode] = self._dist_contexts[
            mode].dist_startup_programs
        self._feed_vars[mode] = self._dist_contexts[mode].serial_feed_vars
        self._fetch_vars[mode] = self._dist_contexts[mode].serial_fetch_vars
        self._lr_optimizer = self._dist_contexts[mode]._lr_optimizer

        if self._nranks > 1:
            # Traverse different rank programs and traverse each op of them,
            # instantiate communication by process_mapping.
            all_process_groups = get_all_process_groups()
            if self._strategy.auto_mode == "full":
                has_recv_by_socket = []
                # This is a magic number
                magic_num = 500
                genv = _get_global_env()
                cur_rank_ip, cur_rank_port = genv.current_endpoint.split(":")
                cur_rank_recv_port = int(cur_rank_port) + magic_num
                server_socket = None
                # Large enough for recv rank
                buff_size = 1024
                server_socket = socket.socket(socket.AF_INET,
                                              socket.SOCK_STREAM)
                server_socket.bind((cur_rank_ip, cur_rank_recv_port))
                # The 10 is an empirical value
                server_socket.listen(10)
                client_sockets = {}
                for process_group in all_process_groups:
                    if self._cur_rank not in process_group.ranks:
                        continue
                    if len(process_group.ranks) == 2:
                        index = process_group.ranks.index(self._cur_rank)
                        is_send = True if index == 0 else False
                        if is_send:
                            recv_rank = process_group.ranks[1]
                            recv_rank_ip, recv_rank_port = genv.trainer_endpoints[
                                recv_rank].split(":")
                            connect_port = int(recv_rank_port) + magic_num
                            client_socket = socket.socket(
                                socket.AF_INET, socket.SOCK_STREAM)
                            client_socket.connect((recv_rank_ip, connect_port))
                            client_socket.send(
                                str(self._cur_rank).encode('utf-8'))
                            rank = client_socket.recv(buff_size).decode('utf-8')
                            rank = int(rank)
                            if rank != recv_rank:
                                raise ValueError(
                                    "Please check comm pair, the recv rank should be {} but got {}."
                                    .format(recv_rank, rank))
                            else:
                                print(
                                    "It is able to instantiate {} as sender now."
                                    .format(process_group.ranks))
                            client_socket.close()
                        else:
                            send_rank = process_group.ranks[0]
                            while True:
                                if send_rank not in has_recv_by_socket:
                                    client_socket, recv_addr = server_socket.accept(
                                    )
                                    rank = int(
                                        client_socket.recv(buff_size).decode())
                                    client_sockets[rank] = client_socket
                                    has_recv_by_socket.append(rank)
                                else:
                                    client_sockets[send_rank].send(
                                        str(self._cur_rank).encode("utf-8"))
                                    client_sockets[send_rank].close()
                                    print(
                                        "It is able to instantiate {} as recver now."
                                        .format(process_group.ranks))
                                    break
                    process_group.instantiate()
                server_socket.close()
            else:
                for process_group in all_process_groups:
                    if self._cur_rank not in process_group.ranks:
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

            if hasattr(self, "_state_dict") and hasattr(self, "_dist_attr"):
                self._set_state_dict(mode, self._strict, self._state_dict,
                                     self._dist_attr)

        if self._strategy.reinit:
            self._logger.info("NOTE: parameters wiil be re-initialized.")
            dist_startup_prog = self._dist_startup_progs[mode][self._cur_rank]
            self._executor.run(dist_startup_prog)

    def fit(self,
            train_data,
            train_sample_split=None,
            batch_size=1,
            epochs=1,
            steps_per_epoch=None,
            valid_data=None,
            valid_sample_split=None,
            valid_freq=1,
            valid_steps=None,
            collate_fn=None,
            callbacks=None):
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
            train_data, train_sample_split, batch_size)
        self._inputs, self._labels = self._prepare_data_tensor(
            self._inputs_spec, self._labels_spec)
        if not self._has_prepared[self._mode]:
            self._prepare_program(self._mode)
        else:
            self._switch_mode(self._mode)
        train_dataloader = self._prepare_dataloader_from_generator(
            dataset=train_data,
            capacity=70,
            # use_double_buffer=use_double_buffer,
            iterable=False,
            # return_list=return_list,
            # use_multiprocess=use_multiprocess,
            # drop_last=drop_last,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            collate_fn=collate_fn)
        fetch_names, fetch_indices = self._prepare_fetch(None, mode=self._mode)
        lr_scheduler = self._get_lr_scheduler(self.main_program)

        with profiler.Profiler(timer_only=True) as prof:
            for epoch in range(epochs):
                for step, _ in enumerate(train_dataloader):
                    try:
                        outs = self._executor.run(
                            self.main_program,
                            fetch_list=fetch_names,
                            use_program_cache=self._strategy.use_cache,
                            return_numpy=self._strategy.return_numpy)
                    except core.EOFException:
                        break
                    if lr_scheduler and step % self._k_steps == 0:
                        lr_scheduler.step()
                    lr = self._get_lr(self._lr_optimizer)

                    prof.step()

                    self._prepare_logger(outs, epoch, step, lr,
                                         fetch_names, fetch_indices,
                                         prof.step_info(), self._mode)
                    history = self._prepare_history(outs, fetch_indices,
                                                    self._mode)

                if valid_data and epoch % valid_freq == 0:
                    self.evaluate(valid_data, valid_sample_split, batch_size,
                                  valid_steps, collate_fn, callbacks)
                    self._switch_mode("train")
                else:
                    self._reset_metrics()
            return history

    def evaluate(self,
                 valid_data,
                 valid_sample_split=None,
                 batch_size=1,
                 steps=None,
                 collate_fn=None,
                 callbacks=None):
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
            valid_data, valid_sample_split, batch_size)
        self._inputs, self._labels = self._prepare_data_tensor(
            self._inputs_spec, self._labels_spec)
        if not self._has_prepared[self._mode]:
            self._prepare_program(self._mode)
        else:
            self._switch_mode(self._mode)
        assert self._mode in self._dist_main_progs, \
            "eval model is not ready, please call `engine._prepare_program('eval')` first."
        valid_dataloader = self._prepare_dataloader_from_generator(
            dataset=valid_data,
            # feed_list=feed_list,
            capacity=70,
            # use_double_buffer=use_double_buffer,
            iterable=False,
            # return_list=return_list,
            # use_multiprocess=use_multiprocess,
            # drop_last=drop_last,
            # places=places,
            batch_size=batch_size,
            # epochs=epochs,
            steps_per_epoch=steps,
            collate_fn=collate_fn)
        fetch_names, fetch_indices = self._prepare_fetch(None, mode=self._mode)

        for step, _ in enumerate(valid_dataloader):
            try:
                outs = self._executor.run(
                    self.main_program,
                    fetch_list=fetch_names,
                    use_program_cache=self._strategy.use_cache,
                    return_numpy=self._strategy.return_numpy)
            except core.EOFException:
                break
            self._prepare_logger(outs, None, step, None, fetch_names,
                                 fetch_indices, "", self._mode)
            history = self._prepare_history(outs, fetch_indices, self._mode)
        self._reset_metrics()
        return history

    def predict(self,
                test_data,
                test_sample_split=None,
                batch_size=1,
                steps=None,
                collate_fn=None,
                callbacks=None):
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
            test_data, test_sample_split, batch_size)
        self._inputs, self._labels = self._prepare_data_tensor(
            self._inputs_spec, self._labels_spec)
        if not self._has_prepared[self._mode]:
            self._prepare_program(self._mode)
        else:
            self._switch_mode(self._mode)
        assert self._mode in self._dist_main_progs, \
            "predict model is not ready, please call `engine._prepare_program('predict')` first."
        test_dataloader = self._prepare_dataloader_from_generator(
            dataset=test_data,
            # feed_list=feed_list,
            capacity=70,
            # use_double_buffer=use_double_buffer,
            iterable=False,
            # return_list=return_list,
            # use_multiprocess=use_multiprocess,
            # drop_last=drop_last,
            # places=places,
            batch_size=batch_size,
            # epochs=epochs,
            steps_per_epoch=steps,
            collate_fn=collate_fn)
        fetch_names, fetch_indices = self._prepare_fetch(None, mode=self._mode)

        for step, _ in enumerate(test_dataloader):
            try:
                outs = self._executor.run(
                    self.main_program,
                    fetch_list=fetch_names,
                    use_program_cache=self._strategy.use_cache,
                    return_numpy=self._strategy.return_numpy)
            except core.EOFException:
                break
            self._prepare_logger(outs, None, step, None, fetch_names,
                                 fetch_indices, "", self._mode)
            history = self._prepare_history(outs, fetch_indices, self._mode)

        return history

    def dataloader(
            self,
            dataset,
            # return_list=True,
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
            mode=None):
        if mode is not None:
            self.to_mode(mode)
        self._inputs_spec, self._labels_spec = self._prepare_data_spec(
            dataset, sample_split, batch_size)
        self._inputs, self._labels = self._prepare_data_tensor(
            self._inputs_spec, self._labels_spec)
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
            steps_per_epoch=steps_per_epoch)
        return dataloader

    def dataloader_from_generator(
            self,
            dataset,
            capacity=70,
            use_double_buffer=True,
            iterable=True,
            # return_list=False,
            use_multiprocess=False,
            drop_last=True,
            batch_size=1,
            epochs=1,
            steps_per_epoch=None,
            collate_fn=None,
            sample_split=1,
            mode=None):
        if mode is not None:
            self.to_mode(mode)
        self._inputs_spec, self._labels_spec = self._prepare_data_spec(
            dataset, sample_split, batch_size)
        self._inputs, self._labels = self._prepare_data_tensor(
            self._inputs_spec, self._labels_spec)
        if not self._has_prepared[self._mode]:
            self._prepare_program(self._mode)
        else:
            self._switch_mode(self._mode)
        dataloader = self._prepare_dataloader_from_generator(
            dataset=dataset,
            # feed_list=feed_list,
            capacity=capacity,
            use_double_buffer=use_double_buffer,
            iterable=iterable,
            return_list=False,
            use_multiprocess=use_multiprocess,
            drop_last=drop_last,
            # places=places,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            collate_fn=collate_fn)
        return dataloader

    def prepare(self,
                inputs_spec=None,
                labels_spec=None,
                inputs=None,
                labels=None,
                main_program=None,
                startup_program=None,
                mode=None):
        if mode is not None:
            self.to_mode(mode)
        if inputs or labels:
            self._skip_build = True
            self._inputs_spec = inputs_spec
            self._labels_spec = labels_spec
            self._inputs, self._labels = self._prepare_data_tensor(
                self._inputs_spec, self._labels_spec, inputs, labels)
            self._orig_main_prog = main_program
            if self._orig_main_prog is None:
                self._orig_main_prog = static.default_main_program()
            self._orig_startup_prog = startup_program
            if self._orig_startup_prog is None:
                self._orig_startup_prog = static.default_startup_program()
            if not self._has_prepared[self._mode]:
                self._prepare_program(self._mode)
            else:
                self._switch_mode(self._mode)
        elif inputs_spec or labels_spec:
            self._inputs_spec = inputs_spec
            self._labels_spec = labels_spec
            self._outside_dataloader = True
            self._inputs, self._labels = self._prepare_data_tensor(
                self._inputs_spec, self._labels_spec)
            self._orig_main_prog = main_program
            if self._orig_main_prog is None:
                self._orig_main_prog = static.default_main_program()
            self._orig_startup_prog = startup_program
            if self._orig_startup_prog is None:
                self._orig_startup_prog = static.default_startup_program()
            if not self._has_prepared[self._mode]:
                self._prepare_program(self._mode)
            else:
                self._switch_mode(self._mode)
        else:
            assert self._inputs_spec and self._labels_spec, \
                "Please call the dataloader(...) before calling prepare(...)"

    def run(
        self,
        data=None,
        # program=None,
        feed=None,
        fetch_list=None,
        # feed_var_name='feed',
        # fetch_var_name='fetch',
        # scope=None,
        # return_numpy=True,
        # use_program_cache=False,
        # return_merged=True,
        # use_prune=False,
        mode=None):
        if mode is not None:
            self.to_mode(mode)
        feed_dict = self._prepare_feed(data, feed, self._mode)
        fetch_names, fetch_indices = self._prepare_fetch(fetch_list, self._mode)
        if self._outside_dataloader and not self._has_prepared_reader[
                self._mode]:
            self._prepare_reader()
        outs = self._executor.run(self.main_program,
                                  feed=feed_dict,
                                  fetch_list=fetch_names,
                                  use_program_cache=self._strategy.use_cache,
                                  return_numpy=self._strategy.return_numpy)
        self._prepare_logger(outs, None, None, None, fetch_names, fetch_indices,
                             "", self._mode)
        history = self._prepare_history(outs, fetch_indices, self._mode)
        return history

    def _prepare_dataloader(self,
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
                            steps_per_epoch=None):

        if self._strategy.gradient_merge and batch_size is not None:
            assert batch_size % self._k_steps == 0, \
                "Requires batch_size:[{}] to be divisible by k_steps:[{}].".format(batch_size, self._k_steps)
            batch_size //= self._k_steps

        dist_main_prog = self._dist_main_progs[self._mode][self._cur_rank]
        dist_startup_prog = self._dist_startup_progs[self._mode][self._cur_rank]
        dist_context = self._dist_contexts[self._mode]
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
                data_parallel_rank=self._dp_ranks)

        return dataloader

    def _prepare_dataloader_from_generator(self,
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
                                           collate_fn=None):

        if self._strategy.gradient_merge and batch_size is not None:
            assert batch_size % self._k_steps == 0, \
                "Requires batch_size:[{}] to be divisible by k_steps:[{}].".format(batch_size, self._k_steps)
            batch_size //= self._k_steps

        dist_main_prog = self._dist_main_progs[self._mode][self._cur_rank]
        dist_startup_prog = self._dist_startup_progs[self._mode][self._cur_rank]
        dist_context = self._dist_contexts[self._mode]
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

        # # remove the first three ops if multi run fit/evaluate/predict
        # self._op_size = len(dist_main_block.ops)
        # if dist_main_block.ops[0].type == 'create_py_reader':
        #     op_size -= 3
        #     for _ in range(3):
        #         dist_main_block._remove_op(0, sync=False)

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
                data_parallel_rank=self._dp_ranks)
        self._prepare_reader()
        # # move read op from the end of program to the start of program
        # new_op_size = len(dist_main_block.ops)
        # for _ in range(new_op_size - 1, op_size - 1, -1):
        #     op = dist_main_block.ops[new_op_size - 1]
        #     new_op_desc = dist_main_block.desc._prepend_op()
        #     new_op_desc.copy_from(op.desc)
        #     new_op = Operator(dist_main_block,
        #                       new_op_desc,
        #                       type=new_op_desc.type())
        #     dist_main_block.ops.insert(0, new_op)
        #     dist_op = DistributedOperator(new_op)
        #     dist_context.add_dist_op_for_program(dist_op)
        # for _ in range(new_op_size - op_size):
        #     dist_main_block._remove_op(new_op_size, sync=False)
        # dist_main_block._sync_with_cpp()
        return dataloader

    def _tune(self, tune_data, tune_sample_split=None, batch_size=1):
        self._mode = 'train'
        self._inputs_spec, self._labels_spec = self._prepare_data_spec(
            tune_data, tune_sample_split, batch_size)
        self._inputs, self._labels = self._prepare_data_tensor(
            self._inputs_spec, self._labels_spec)
        self._optimization_tuning(self._mode, tune_data, batch_size)

    def _validate_spec(self, specs):
        specs = to_list(specs)
        self._k_steps = self._strategy.gradient_merge.k_steps
        if specs is not None:
            for i, spec in enumerate(specs):
                assert isinstance(spec, InputSpec)
                if spec.name is None:
                    raise ValueError(
                        "Requires Input[{}].name != None, but receive `None` with {}."
                        .format(i, spec))
                if self._k_steps > 1:
                    shape = list(spec.shape)
                    assert shape[0] % self._k_steps == 0, \
                        "Requires batch_size[{}] to be divisible by k_steps[{}].".format(spec.shape[0], self._k_steps)
                    shape[0] //= self._k_steps
                    spec.shape = shape
        return specs

    def _is_local_var(self, var):
        var_name = _to_name_str(var)
        return var_name in self.main_program.global_block().vars

    def _get_input_split_info(self, var, dist_context):
        # deduce how the input data is split among the cluster
        from .utils import _get_comm_group, _get_corresponding_rank

        tensor_dist_attr = dist_context.get_tensor_dist_attr_for_program(var)
        process_mesh = tensor_dist_attr.process_mesh
        dims_mapping = tensor_dist_attr.dims_mapping

        if self._cur_rank not in process_mesh.processes:
            rank_id = _get_corresponding_rank(dist_context, process_mesh,
                                              self._cur_rank)
        else:
            rank_id = self._cur_rank

        batch_size_axis = dims_mapping[0]
        if batch_size_axis > -1 and process_mesh.topology[batch_size_axis] > 1:
            group_ranks = _get_comm_group(process_mesh.processes,
                                          process_mesh.topology,
                                          batch_size_axis, rank_id)
            return len(group_ranks), group_ranks.index(rank_id)

        return 1, 0

    def _set_recompute_ckpts(self):
        # NOTE hack to enable recompute in engine api for GPT-3
        # TODO support more PaddleNLP/CV models here

        recompute = self._strategy.recompute

        # extract ckpts by specific model
        if isinstance(self._model, paddle.nn.Layer):
            if hasattr(self._model,
                       "gpt") and self._model.__class__.__name__ in [
                           'GPTForPretraining', 'GPTForPretrainingAuto'
                       ]:
                exact_ckpts = self._model.gpt.checkpoints
            else:
                exact_ckpts = recompute.checkpoints
        else:
            exact_ckpts = recompute.checkpoints

        # modify strategy
        if recompute.enable:
            recompute.checkpoints = exact_ckpts[:]
            logs = {
                'Model Class': self._model.__class__.__name__,
                'Applied Recompute ckpts': exact_ckpts
            }
            self._logger.info(logs)

    def _validate_opt(self, optimizer):
        if optimizer is not None:
            optimizer._parameter_list = None
            optimizer._param_groups = None
        return optimizer

    def _reset_metrics(self):
        for metric in self._metrics:
            metric.reset()

    def _switch_mode(self, mode):
        self.to_mode(mode)
        self._initialize(mode)

    def to_mode(self, mode):
        assert mode in ["train", "eval", "predict"], \
            "mode {} should be one of ['train', 'eval', 'predict']".format(mode)
        self._mode = mode

    def _set_state_dict(self, mode, strict, state_dict, dist_attr):
        program = self._dist_main_progs[mode][self._cur_rank]
        dist_context = self._dist_contexts[mode]
        cur_dist_attr = get_dist_attr(program, dist_context)
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
            assert 'train' in self._serial_main_progs, \
                "training model is not ready, please call `engine._prepare_program('train')` first."
            serial_program = self._serial_main_progs["train"]
            dist_main_prog = self._dist_main_progs["train"][self._cur_rank]
            dist_context = self._dist_contexts["train"]
            self._saver.save(path,
                             serial_program=serial_program,
                             dist_main_program=dist_main_prog,
                             dist_context=dist_context)
        else:
            mode = "predict"
            feed_vars = self._feed_vars[mode]['inputs']
            fetch_vars = self._fetch_vars[mode]['outputs']
            dist_main_prog = self._dist_main_progs[mode][self._cur_rank]
            self._saver.save_inference_model(path,
                                             feed_vars,
                                             fetch_vars,
                                             self._executor,
                                             program=dist_main_prog)

    def load(self, path, strict=True, load_optimizer=True):
        """
        Load the stored model, parameters and optimizer states.

        Args:
            path (str): The prefix of files storing the model states and
                optimizer states.
            strict (bool, optional): Whether to skip the loading of mismatch
                parameter or raise an error when mismatch happens (not found
                the parameter in file storing model states of or receives a
                mismatch shape). Default: False.
            load_optimizer (bool, optional): If True, the stored optimizer
                states is restored. Otherwise, the optimizer states is initialized
                from scratch. Default: False.

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
            path, load_optimizer)
        return self._state_dict, self._dist_attr

    @staticmethod
    def _get_lr_scheduler(program):
        lr_sheduler = None
        if hasattr(program, 'lr_sheduler'):
            from paddle.optimizer.lr import LRScheduler
            lr_sheduler = program.lr_sheduler
            assert isinstance(lr_sheduler, LRScheduler), "must be LRScheduler"
        return lr_sheduler

    def _get_lr(self, optimizer):
        if isinstance(optimizer, paddle.optimizer.Optimizer):
            return optimizer.get_lr()
        elif isinstance(optimizer, paddle.fluid.optimizer.Optimizer):
            if isinstance(optimizer._learning_rate, float):
                return optimizer._learning_rate
            else:
                return optimizer._learning_rate()
        else:
            raise TypeError(
                    "'optimizer' must be object of class `paddle.optimizer.Optimizer`" \
                        " or `paddle.fluid.optimizer.Optimizer`, but got {}.".format(type(optimizer))
                )

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

    @property
    def fetch_vars(self):
        return self._fetch_vars[self._mode]

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels
