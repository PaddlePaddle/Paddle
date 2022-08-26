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

import time
import copy
import logging
from collections import defaultdict

import paddle
import paddle.utils as utils

from paddle import fluid, static
from paddle.io import Dataset
from paddle.jit import to_static
from paddle.metric import Metric
from paddle.static import InputSpec
from paddle.fluid import core
from paddle.fluid import program_guard
from paddle.fluid.layers.utils import flatten
from paddle.fluid.executor import global_scope, _to_name_str
from paddle.fluid.backward import append_backward
from paddle.fluid.framework import Operator, Parameter, _non_static_mode
from paddle.fluid.framework import _current_expected_place as _get_device
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.distributed import fleet
from paddle.distributed.passes import new_pass, PassContext

from .hepler import ProgramHelper
from ..collective import _get_global_env
from .cluster import Cluster, get_default_cluster
from .planner_v2 import Planner
from .parallelizer_v2 import Parallelizer
from .dist_op import DistributedOperator
from .dist_saver import DistributedSaver
from .dist_loader import NonIterableGeneratorLoader
from .utils import make_data_unshard, set_grad_var_shape
from .utils import print_program_with_dist_attr, to_list
from .process_group import new_process_group, get_all_process_groups, get_world_process_group
from .dist_context import DistributedContext, get_default_distributed_context


class Engine:

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

        self._executor = None
        self._cur_rank = paddle.distributed.get_rank()
        self._nranks = paddle.distributed.get_world_size()
        self._saver = DistributedSaver()

        # TODO: add logger module
        self._logger = logging.getLogger()
        self._logger.propagate = False
        if not self._logger.handlers:
            self._logger.setLevel(logging.INFO)
            log_handler = logging.StreamHandler()
            log_format = logging.Formatter(
                '[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s'
            )
            log_handler.setFormatter(log_format)
            self._logger.addHandler(log_handler)

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

    def _build(self, mode):
        if _non_static_mode() or self._dygraph_mode:
            paddle.disable_static()
            self._dygraph_mode = True
            self._logger.info("Building model with 'to_static' method.")

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

            paddle.enable_static()
        else:
            # build program in static mode
            serial_main_prog = self._serial_main_progs.get(mode, None)
            if serial_main_prog is not None:
                return

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
        dataset.dp_world_size = self._input_split_size
        dataset.dp_rank = self._input_split_rank

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
        feed_list = []
        for var in inputs_var + labels_var:
            if var.name in block.vars:
                feed_list.append(block.vars[var.name])

        self._input_split_size, self._input_split_rank = self._get_input_split_info(
            feed_list[0], self._dist_contexts[mode])

    def _parallel(self, mode, all_ranks):
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
            collate_fn=None,
            use_cache=False,
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
                                                   epochs, steps_per_epoch,
                                                   collate_fn)

        usr_fetch = self._validate_fetches(fetches)
        fetch_loss = self._validate_fetches(self.fetch_vars["loss"])
        fetch_list, fetch_map = self._fetch_map(fetch_loss, usr_fetch)
        lr_scheduler = self.get_lr_scheduler(self.main_program)

        for epoch in range(epochs):
            train_logs = {"epoch: {:d} ": epoch}
            for step, _ in enumerate(train_dataloader):

                outs = self._executor.run(self.main_program,
                                          fetch_list=fetch_list,
                                          use_program_cache=use_cache,
                                          return_numpy=return_numpy)
                train_logs["step: {:d} "] = step
                if lr_scheduler is not None:
                    lr_scheduler.step()
                    train_logs["lr: {:5e} "] = self._lr_optimizer.get_lr()
                # inner fetches
                if fetch_loss:
                    train_logs["loss: {:9f} "] = outs[0][0]
                # user fetches
                user_outs = outs[len(fetch_loss):]
                user_fetch_list = fetch_list[len(fetch_loss):]
                for i, out in enumerate(user_outs):
                    train_logs[fetch_map[user_fetch_list[i]] + ": {}"] = out
                # logger
                string = '[train] ' + ''.join(list(train_logs.keys()))
                self._logger.info(string.format(*list(train_logs.values())))

    def evaluate(self,
                 eval_data,
                 batch_size=1,
                 fetches=None,
                 collate_fn=None,
                 use_cache=False,
                 return_numpy=True):
        self.mode = 'eval'
        if not self._mode_init_states[self.mode]:
            self._prepare_single_mode(self.mode)

        assert self.mode in self._dist_main_progs, \
            "eval model is not ready, please call `engine.prepare()` first."
        eval_dataloader = self._create_dataloader(eval_data,
                                                  batch_size,
                                                  collate_fn=collate_fn)

        usr_fetch = self._validate_fetches(fetches)
        fetch_loss = self._validate_fetches(self.fetch_vars["loss"])
        fetch_metrics = self._validate_fetches(self.fetch_vars["metrics"])
        inner_fetch = dict(fetch_loss, **fetch_metrics)
        fetch_list, fetch_map = self._fetch_map(inner_fetch, usr_fetch)

        for step, _ in enumerate(eval_dataloader):
            eval_logs = {"step: {:d} ": step}
            outs = self._executor.run(self.main_program,
                                      fetch_list=fetch_list,
                                      use_program_cache=use_cache,
                                      return_numpy=return_numpy)
            # inner fetches
            if fetch_loss:
                eval_logs["loss: {:9f} "] = outs[0][0]
            # Metric
            if fetch_metrics:
                metric_out = outs[len(fetch_loss):len(inner_fetch)]
                for metric in self._metrics:
                    metric.update(*metric_out)
                    results = metric.accumulate()
                    for i, res in enumerate(to_list(results)):
                        eval_logs[metric.name()[i] + ": {:9f} "] = res
            # usr fetches
            usr_outs = outs[len(inner_fetch):]
            usr_fetch_list = fetch_list[len(inner_fetch):]
            for i, out in enumerate(usr_outs):
                eval_logs[fetch_map[usr_fetch_list[i]] + ": {}"] = out
            # logger
            string = '[eval] ' + ''.join(list(eval_logs.keys()))
            self._logger.info(string.format(*list(eval_logs.values())))

    def predict(self,
                test_data,
                batch_size=1,
                fetches=None,
                collate_fn=None,
                use_cache=False,
                return_numpy=True):
        self.mode = 'predict'
        if not self._mode_init_states[self.mode]:
            self._prepare_single_mode(self.mode)

        assert self.mode in self._dist_main_progs, \
            "predict model is not ready, please call `engine.prepare()` first."
        test_dataloader = self._create_dataloader(test_data,
                                                  batch_size,
                                                  collate_fn=collate_fn)

        usr_fetch = self._validate_fetches(fetches)
        fetch_outputs = self._validate_fetches(self.fetch_vars["outputs"])
        fetch_list, fetch_map = self._fetch_map(fetch_outputs, usr_fetch)

        outputs = []
        for step, _ in enumerate(test_dataloader):
            predict_logs = {"step: {:d} ": step}
            outs = self._executor.run(self.main_program,
                                      fetch_list=fetch_list,
                                      use_program_cache=use_cache,
                                      return_numpy=return_numpy)
            outputs.append(outs[:len(fetch_outputs)])
            for i, out in enumerate(outs):
                predict_logs[fetch_map[fetch_list[i]] + ": {}"] = out
            # logger
            string = '[pred] ' + ''.join(list(predict_logs.keys()))
            self._logger.info(string.format(*list(predict_logs.values())))

        return outputs

    def _create_dataloader(self,
                           dataset,
                           batch_size,
                           epochs=1,
                           steps_per_epoch=None,
                           collate_fn=None):
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
                feed_list,
                places,
                batch_size,
                epochs,
                steps_per_epoch,
                collate_fn,
                data_parallel_world_size=self._input_split_size,
                data_parallel_rank=self._input_split_rank)

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

    def _is_local_var(self, var):
        var_name = _to_name_str(var)
        return var_name in self.main_program.global_block().vars

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

        return None, None

    def _set_recompute_ckpts(self):
        # NOTE hack to enable recompute in engine api for GPT-3
        # TODO support more PaddleNLP/CV models here

        config = self.strategy.recompute_configs

        # extract ckpts by specific model
        if isinstance(self.model, paddle.nn.Layer):
            if hasattr(
                    self.model, "gpt"
            ) and self.model.__class__.__name__ == 'GPTForPretraining':
                exact_ckpts = self.model.gpt.checkpoints
        else:
            exact_ckpts = config["checkpoints"]

        # modify strategy
        if self.strategy.recompute:
            config["checkpoints"] = exact_ckpts[:]
            self.strategy.recompute_configs = config
            logs = {
                'Model Class': self.model.__class__.__name__,
                'Applied Recompute ckpts': exact_ckpts
            }
            self._logger.info(logs)

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

    @staticmethod
    def get_lr_scheduler(program):
        lr_sheduler = None
        if hasattr(program, 'lr_sheduler'):
            from paddle.optimizer.lr import LRScheduler
            lr_sheduler = program.lr_sheduler
            assert isinstance(lr_sheduler, LRScheduler), "must be LRScheduler"
        return lr_sheduler

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
