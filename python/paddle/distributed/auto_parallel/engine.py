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
import time
import socket

import paddle
import paddle.utils as utils
import paddle.distributed.auto_parallel as auto

from paddle import fluid, static
from paddle.io import Dataset
from paddle.metric import Metric
from paddle.static import InputSpec
from paddle.fluid import core
from paddle.fluid import program_guard
from paddle.fluid.layers.utils import flatten
from paddle.fluid.executor import global_scope, _to_name_str
from paddle.fluid.backward import append_backward
from paddle.fluid.framework import Operator
from paddle.fluid.framework import _current_expected_place as _get_device
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.distributed import fleet
from paddle.distributed.utils import get_logger
from paddle.distributed.passes import new_pass, PassContext

from ..collective import _get_global_env
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
        # if self.cluster is None:
        #     self.cluster = get_default_cluster()
        self.strategy = strategy
        if self.strategy is None:
            self.strategy = fleet.DistributedStrategy()

        self._executor = None
        self._cur_rank = paddle.distributed.get_rank()
        self._nranks = paddle.distributed.get_world_size()
        self._saver = DistributedSaver()
        self._logger = get_logger(logging.INFO)

        self._default_strategy = None
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
        self._modes = ['train', 'eval', 'predict']
        # Build forward program
        self._build()

        # Do auto parallel process
        for mode in self._modes:
            # Do the planning process
            self._plan(mode)
            # Do the parallel process
            self._parallel(mode, all_ranks)
            # Init comm and startup program
            self._initialize(mode)

    def _build(self):
        for mode in self._modes:
            serial_main_prog = self._serial_main_progs.get(mode, None)
            if serial_main_prog is not None:
                return

            losses = []
            metrics = []
            serial_main_prog = self._orig_main_prog.clone()
            serial_startup_prog = self._orig_startup_prog.clone()
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

            default_ctx = get_default_distributed_context()
            if not default_ctx.has_annotation or self._default_strategy:
                inputs = [self._set_data_parallel(var) for var in inputs]
                labels = [self._set_data_parallel(var) for var in labels]

            # self._feed_vars[mode] = {"inputs": inputs, "labels": labels}
            feed_vars = {"inputs": inputs, "labels": labels}

            # self._fetch_vars[mode] = {
            #     "outputs": flatten(outputs),
            #     "loss": losses,
            #     "metrics": metrics
            # }
            fetch_vars = {
                "outputs": flatten(outputs),
                "loss": losses,
                "metrics": metrics
            }

            self._dist_contexts[mode] = DistributedContext(
                serial_main_prog, serial_startup_prog, self._optimizer, losses,
                feed_vars, fetch_vars, self.cluster, self.strategy)
            self._dist_contexts[mode].gradient_scale = self._gradient_scale

    def _plan(self, mode):
        if self._planned_mode is None:
            self._planned_mode = mode
        else:
            self._init_dist_context(mode)

        self.planner = Planner(mode, self._dist_contexts[mode])
        self.planner.plan()

    def _parallel(self, mode, all_ranks):
        # Parallelize program based on the planner's results
        # For now, the completer has to be passed to the planner,
        # because we may use it to complete the annotation of the backwarkward and update.
        parallelizer = Parallelizer(mode, self.planner.completer,
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

        if self._nranks > 1:
            # Traverse different rank programs and traverse each op of them,
            # instantiate communication by process_mapping.
            all_process_groups = get_all_process_groups()
            print("engine.py process group")
            for process_group in all_process_groups:
                print(process_group.ranks, process_group.id)
            has_recv_by_socket = []
            magic_num = 5000  # this is a magic number and the rank number for training is usually less than 5000.
            genv = _get_global_env()
            cur_rank_ip, cur_rank_port = genv.current_endpoint.split(":")
            print("The cur_rank_ip is {}, cur_rank_port is {}".format(
                cur_rank_ip, cur_rank_port))
            cur_rank_recv_port = int(cur_rank_port) + magic_num
            server_socket = None
            buff_size = 1024  # it is large enough for recv rank
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind((cur_rank_ip, cur_rank_recv_port))
            server_socket.listen(10)  # the 10 is an empirical value
            client_sockets = {}
            for process_group in all_process_groups:
                if self._cur_rank not in process_group.ranks:
                    continue
                if len(process_group.ranks) == 2:
                    index = process_group.ranks.index(self._cur_rank)
                    is_send = True if index == 0 else False
                    if is_send:
                        recv_rank = process_group.ranks[1]
                        # 此时新建一个socket，发送给recv方当前的rank，然后另一个socket进行接收，直到接收到当前rank的数据，再进行初始化
                        recv_rank_ip, recv_rank_port = genv.trainer_endpoints[
                            recv_rank].split(":")
                        connect_port = int(recv_rank_port) + magic_num
                        client_socket = socket.socket(socket.AF_INET,
                                                      socket.SOCK_STREAM)
                        client_socket.connect((recv_rank_ip, connect_port))
                        client_socket.send(str(self._cur_rank).encode('utf-8'))
                        print("has send {} to {}:{}.".format(
                            self._cur_rank, recv_rank_ip, connect_port))
                        rank = client_socket.recv(buff_size).decode('utf-8')
                        print("engine.py rank: ", rank)
                        rank = int(rank)
                        if rank != recv_rank:
                            raise ValueError(
                                "Please check comm pair, the recv rank should be {} but got {}."
                                .format(recv_rank, rank))
                        else:
                            print(
                                "engine.py It is able instantiate {} as sender now."
                                .format(process_group.ranks))
                        client_socket.close()
                    else:
                        # [0, 7]
                        send_rank = process_group.ranks[0]
                        while True:
                            if send_rank not in has_recv_by_socket:
                                client_socket, recv_addr = server_socket.accept(
                                )
                                rank = int(
                                    client_socket.recv(buff_size).decode())
                                client_sockets[rank] = client_socket
                                has_recv_by_socket.append(rank)
                                print("has recv {} from {}.".format(
                                    rank, recv_addr))
                            else:
                                client_sockets[send_rank].send(
                                    str(self._cur_rank).encode("utf-8"))
                                client_sockets[send_rank].close()
                                print(
                                    "engine.py It is able instantiate {} as recver now."
                                    .format(process_group.ranks))
                                break
                print("instantiating process group: ", process_group.ranks,
                      process_group.id)
                process_group.instantiate()
            server_socket.close()

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
                print("engine.py startup_program")
                self._executor.run(prune_startup_prog)

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
        self.mode = 'train'
        assert self.mode in self._dist_main_progs, \
            "train model is not ready, please call `engine.prepare()` first."
        train_dataloader = self._create_dataloader(train_data, batch_size,
                                                   epochs, steps_per_epoch)

        usr_fetch = self._to_map_fetch(fetches)
        fetch_loss = self._inner_fetch(self.fetch_vars["loss"])
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
                    train_logs["train_" +
                               fetch_map[user_fetch_list[i]]] = out[0]
                self._logger.info(train_logs)

    def evaluate(self,
                 eval_data,
                 batch_size=1,
                 fetches=None,
                 use_program_cache=False,
                 return_numpy=True):
        self.mode = 'eval'
        assert self.mode in self._dist_main_progs, \
            "eval model is not ready, please call `engine.prepare()` first."
        eval_dataloader = self._create_dataloader(eval_data, batch_size)

        usr_fetch = self._to_map_fetch(fetches)
        fetch_loss = self._inner_fetch(self.fetch_vars["loss"])
        fetch_metrics = self._inner_fetch(self.fetch_vars["metrics"])
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
                eval_logs["eval_loss"] = outs[0]
            # Metric
            if fetch_metrics:
                metric_out = outs[len(fetch_loss):len(inner_fetch)]
                for metric in self._metrics:
                    metric.update(*metric_out)
                    results = metric.accumulate()
                    for i, res in enumerate(to_list(results)):
                        eval_logs["eval_" + metric.name()[i]] = res
            # usr fetches
            usr_out = outs[len(inner_fetch):]
            usr_fetch_list = fetch_list[len(inner_fetch):]
            for i, out in enumerate(usr_out):
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
        assert self.mode in self._dist_main_progs, \
            "predict model is not ready, please call `engine.prepare()` first."
        test_dataloader = self._create_dataloader(test_data, batch_size)

        usr_fetch = self._to_map_fetch(fetches)
        fetch_outputs = self._inner_fetch(self.fetch_vars["outputs"])
        fetch_list, fetch_map = self._fetch_map(fetch_outputs, usr_fetch)

        dist_main_prog = self._dist_main_progs[self.mode][self._cur_rank]
        print("engine.py predict step****")
        # print_program_with_dist_attr(dist_main_prog, self.dist_context)
        outputs = []
        for step, _ in enumerate(test_dataloader):
            predict_logs = {"step": step}
            outs = self._executor.run(self.main_program,
                                      fetch_list=fetch_list,
                                      use_program_cache=use_program_cache,
                                      return_numpy=return_numpy)
            outputs.append(outs[:len(fetch_outputs)])
            for i, out in enumerate(outs):
                predict_logs["pred_" + fetch_map[fetch_list[i]]] = out[0]
            self._logger.info(predict_logs)

        return outputs

    def _local_var(self, var):
        var_name = _to_name_str(var)
        return var_name in self.main_program.global_block().vars

    def _to_map_fetch(self, fetches):
        if not fetches:
            return {}
        if isinstance(fetches, dict):
            fetch_var_names = list(map(_to_name_str, fetches.values()))
            usr_fetches = dict(zip(fetch_var_names, list(fetches.keys())))
        elif isinstance(fetches, list):
            fetch_var_names = list(map(_to_name_str, fetches))
            usr_fetches = dict(zip(fetch_var_names, fetch_var_names))
        return dict(filter(lambda x: self._local_var(x[0]),
                           usr_fetches.items()))

    def _inner_fetch(self, fetch_vars):
        fetch_list = list(
            map(lambda x: x.name, list(filter(self._local_var, fetch_vars))))
        inner_fetches = dict(zip(fetch_list, fetch_list))
        return inner_fetches

    def _fetch_map(self, inner_fetch, usr_fetch):
        # replace inner fetch name if usr set for it
        for iname in inner_fetch:
            if iname in usr_fetch:
                inner_fetch[iname] = usr_fetch[iname]
                usr_fetch.pop(iname)
        fetches = dict(inner_fetch, **usr_fetch)
        return list(fetches.keys()), fetches

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
        feed_list = []
        for var in inputs_var + labels_var:
            if var.name in dist_main_block.vars:
                feed_list.append(dist_main_block.vars[var.name])
        dp_world_size, dp_rank = self._get_data_parallel_info(
            feed_list[0], dist_context)

        # remove the first three ops if multi run fit/evaluate/predict
        op_size = len(dist_main_block.ops)
        if dist_main_block.ops[0].type == 'create_py_reader':
            op_size -= 3
            for _ in range(3):
                dist_main_block._remove_op(0, sync=False)

        # insert read op at the end of program
        places = paddle.static.cuda_places()
        with static.program_guard(dist_main_prog, dist_startup_prog):
            dataloader = NonIterableGeneratorLoader(
                dataset,
                inputs_var + labels_var,
                places,
                batch_size,
                epochs,
                steps_per_epoch,
                data_parallel_world_size=dp_world_size,
                data_parallel_rank=dp_rank)

        # move read op from the end of program to the start of program
        new_op_size = len(dist_main_block.ops)
        # for _ in range(new_op_size - 1, op_size - 1, -1):
        #     op = dist_main_block.ops[new_op_size - 1]
        #     new_op_desc = dist_main_block.desc._prepend_op()
        #     new_op_desc.copy_from(op.desc)
        #     new_op = Operator(
        #         dist_main_block, new_op_desc, type=new_op_desc.type())
        #     dist_main_block.ops.insert(0, new_op)
        #     dist_op = DistributedOperator(new_op)
        #     dist_context.add_dist_op_for_program(dist_op)
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
            auto.shard_tensor(var,
                              dist_attr={
                                  "process_mesh": [0],
                                  "dims_mapping":
                                  [-1 for _ in range(len(var.shape))]
                              })
        else:
            self._default_strategy = 'dp'
            auto.shard_tensor(var,
                              dist_attr={
                                  "process_mesh":
                                  list(range(self._nranks)),
                                  "dims_mapping":
                                  [0] + [-1 for _ in range(len(var.shape) - 1)]
                              })

        return var

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

        batch_size_axis = dims_mapping[0]
        if batch_size_axis > -1 and process_mesh.topology[batch_size_axis] > 1:
            group_ranks = _get_comm_group(process_mesh.processes,
                                          process_mesh.topology,
                                          batch_size_axis, rank_id)
            return len(group_ranks), group_ranks.index(rank_id)

        return None, None

    def save(self, path, training=True, mode=None):
        if not mode:
            mode = self.mode

        if training:
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

    @property
    def fetch_vars(self):
        return self._fetch_vars[self.mode]
