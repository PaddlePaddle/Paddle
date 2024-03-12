# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

######

import os
from collections import defaultdict
from distutils.util import strtobool
from functools import reduce

import paddle
from paddle import framework
from paddle.distributed import fleet
from paddle.fluid.dygraph import base as imperative_base
from paddle.fluid.framework import EagerParamBase
from paddle.nn import ClipGradByGlobalNorm

from ...utils import timer_helper as timer
from ...utils.log_util import logger
from ...utils.tensor_fusion_helper import (
    HOOK_ACTION,
    FusedCommBuffer,
    assign_group_by_size,
)

g_shard_use_reduce = int(os.environ.get("FLAGS_shard_use_reduce", 1))
g_shard_norm_align_dp = int(os.environ.get("FLAGS_shard_norm_align_dp", 0))

g_shard_sort_reduce_root = int(
    os.environ.get("FLAGS_shard_sort_reduce_root", 1)
)  # it will remove in the future

g_shard_fused_gradient = int(os.environ.get("FLAGS_shard_fused_gradient", 0))


if g_shard_norm_align_dp:
    assert (
        not g_shard_use_reduce
    ), "g_shard_norm_align_dp is not support if g_shard_use_reduce is true"


def _is_trainable(param):
    return not param.stop_gradient


class DygraphShardingOptimizer:
    """
    A wrapper for Sharding Optimizer in Dygraph.

    .. warning: DygraphShardingOptimizer is experimental and subject to change.

    .. ZeRO: https://arxiv.org/abs/1910.02054

    """

    # TODO (JZ-LIANG)
    # TO support following featrues in future:
    # 1. fused update parameter sync
    # 2. parameters_groups
    # 3. dynamic trainable params, which is the case bewteen pretraining and finetuning
    # 4. option to choose fuse comm (more GPU MEM need) or un-fuse comm

    def __init__(self, optimizer, hcg):
        logger.info("init DygraphShardingOptimizer")
        if not hasattr(optimizer, '_apply_optimize') or not callable(
            optimizer._apply_optimize
        ):
            raise ValueError(
                "the optimzier object should have _apply_optimize function"
            )

        self._using_param_groups = isinstance(
            optimizer._parameter_list[0], dict
        )

        self._parameter_list = []
        self._param_2_group_id = {}
        if self._using_param_groups:
            for idx, param_group in enumerate(optimizer._param_groups):
                for param in param_group['params']:
                    self._param_2_group_id[id(param)] = idx
                    self._parameter_list.append(param)
        else:
            self._parameter_list = optimizer._parameter_list

        self._inner_opt = optimizer
        self._hcg = hcg
        self._sharding_world_size = self._hcg.get_sharding_parallel_world_size()
        self._sharding_rank = self._hcg.get_sharding_parallel_rank()

        self._rank2params = self._partition_parameters()
        self._param2rank = self._map_param_to_rank()

        if self._using_param_groups:
            param_groups = [
                {"params": []} for _ in range(len(optimizer._param_groups))
            ]
            for idx, pg in enumerate(optimizer._param_groups):
                param_groups[idx].update(
                    {k: v for k, v in pg.items() if k != 'params'}
                )
            for param in self._rank2params[self._sharding_rank]:
                group_id = self._param_2_group_id[id(param)]
                param_groups[group_id]['params'].append(param)

            self._set_inner_opt_attr('_param_groups', param_groups)
            self._set_inner_opt_attr(
                '_parameter_list', self._rank2params[self._sharding_rank]
            )
            self._param_groups = self._parameter_list
        else:
            self._set_inner_opt_attr(
                '_param_groups', self._rank2params[self._sharding_rank]
            )
            self._set_inner_opt_attr(
                '_parameter_list', self._rank2params[self._sharding_rank]
            )

        strategy = fleet.fleet._user_defined_strategy
        sharding_configs = strategy.hybrid_configs["sharding_configs"]
        pp_configs = strategy.hybrid_configs["pp_configs"]

        self._pp_overlap = pp_configs.sharding_comm_overlap
        acc_steps = sharding_configs.accumulate_steps
        self.comm_overlap = sharding_configs.comm_overlap
        comm_group = self._hcg.get_sharding_parallel_group()
        self._use_fuse_gradients = g_shard_fused_gradient
        self._use_pipelie_parallel = strategy.hybrid_configs["pp_degree"] > 1

        assert (
            not self.comm_overlap or self._use_fuse_gradients
        ), "If you use comm overlap in sharding, you should set FLAGS_shard_fused_gradient to True"

        assert not (
            self._use_pipelie_parallel and self._use_fuse_gradients
        ), "You can not use pipelie parallel and fused gradient at the same time"

        if self._use_fuse_gradients:
            # Build communication buffers once and store them
            if not hasattr(self, 'comm_buffers'):
                self.comm_buffers = self._build_comm_buffers(
                    comm_group,
                    acc_steps,
                    group_size=128 * 1024 * 1024,
                )
                # NOTE(shenliang03): Sort the comm_buffers by dst rank,
                # it will improve the performance in reduce communicate. Default
                # g_shard_sort_reduce_root is True.
                if g_shard_sort_reduce_root:
                    self.comm_buffers.sort(key=lambda x: x._dst)

        if not self._pp_overlap and self.comm_overlap:
            assert (
                acc_steps > 0
            ), "acc_steps should be larger than 0 when using comm_overlap in sharding"
            self.register_reduce_overlap_hook(use_comm=True)

    def _build_comm_buffers(self, comm_group, acc_steps, group_size):
        parameter_list = list(self._parameter_list)

        if not parameter_list:
            return []

        # Using defaultdict for automatic list creation
        fused_parameter_group = defaultdict(list)

        for p in parameter_list:
            assert p.name in self._param2rank
            dst_rank = self._param2rank[p.name]
            fused_parameter_group[dst_rank].append(p)

        # Pre-compute absolute destination ranks
        absolute_dst_ranks = {
            rank: comm_group.ranks[rank] for rank in fused_parameter_group
        }

        comm_buffers = []
        for dst, params in fused_parameter_group.items():
            var_groups = assign_group_by_size(params, group_size)
            abs_dst = absolute_dst_ranks[dst]

            # Using list comprehension for buffer creation
            buffers = [
                FusedCommBuffer(
                    group_idx,
                    parameters,
                    comm_group,
                    acc_steps,
                    HOOK_ACTION.REDUCE,
                    abs_dst,
                    release_grads=False,
                )
                for group_idx, parameters in var_groups.items()
            ]
            comm_buffers.extend(buffers)

        return comm_buffers

    def register_reduce_overlap_hook(self, use_comm):
        # Register backward hooks for each parameter in the buffer
        for buffer in self.comm_buffers:
            for param in buffer._params:
                # Directly register the hook function with necessary parameters
                param._register_backward_hook(
                    self._create_backward_hook(buffer, param, use_comm)
                )

    def _create_backward_hook(self, buffer, param, use_comm):
        """Creates a backward hook function for autograd."""

        @paddle.autograd.no_grad()
        def fused_allreduce(*_):
            # Directly add gradient to the buffer
            buffer.add_grad(param, use_comm=use_comm)

        return fused_allreduce

    def clear_grad(self, set_to_zero=True):
        """
        should clear grad for all parameters in model
        """
        for p in self._parameter_list:
            if hasattr(p, "main_grad") and p.main_grad is not None:
                assert p._grad_ivar() is None
                if set_to_zero:
                    p.main_grad.zero_()
                else:
                    p.main_grad._clear()
                    p.main_grad = None
            elif not hasattr(p, "main_grad"):
                p.clear_gradient(set_to_zero)

    def filter_parameters(self, parameter_list, hcg):
        sharding_parallel_rank = hcg.get_sharding_parallel_rank()
        parameter_list = [
            param
            for param in parameter_list
            if self._param2rank[param.name] == sharding_parallel_rank
        ]
        return parameter_list

    def _partition_parameters(self):
        """
        Partitions parameters among sharding ranks.

        Return:
        Dict[int, List]
        """
        # TODO(JZ-LIANG) support multiple partition methods
        # method1: greedy even but unorder
        # method2: roughly even with oreder

        mapping = {}
        for rank_ in range(self._sharding_world_size):
            mapping[rank_] = []
        sizes = [0] * self._sharding_world_size

        parameters = list(self._parameter_list)
        need_sort_parameters = strtobool(
            os.getenv('FLAGS_sharding_sort_parameters', '1')
        )
        if need_sort_parameters:
            parameters.sort(
                key=lambda p: reduce(lambda x, y: x * y, p.shape), reverse=True
            )

        for param in parameters:
            rank = sizes.index(min(sizes))
            mapping[rank].append(param)
            numel = reduce(lambda x, y: x * y, param.shape)
            assert (
                numel > 0
            ), "param [{}] should larger than 0, but it is [{}]".format(
                param.name, numel
            )
            sizes[rank] += numel

        return mapping

    def _map_param_to_rank(self):
        """
        mapping parameters to the shard which holds it.

        Return:
        Dict[str, int]
        """
        mapping = {}
        for rank, params in self._rank2params.items():
            for param in params:
                mapping[param.name] = rank
        return mapping

    @paddle.autograd.no_grad()
    def reduce_gradients(self, parameter_list, hcg):
        if self._pp_overlap:
            return

        if self.comm_overlap:
            for buffer in self.comm_buffers:
                buffer.scale_and_split_grads()
            return

        if self._use_fuse_gradients:
            for buffer in self.comm_buffers:
                buffer._comm_grads()
                buffer.scale_and_split_grads()
            return

        sharding_nrank = hcg.get_sharding_parallel_group().nranks
        for param in parameter_list:
            g_var = None
            if param.trainable and (param._grad_ivar() is not None):
                g_var = param._grad_ivar()
            if param.trainable and hasattr(param, "main_grad"):
                assert (
                    param._grad_ivar() is None
                ), "param.grad should be None when using main_grad"
                g_var = param.main_grad
            if g_var is not None:
                g_var.scale_(1.0 / sharding_nrank)
                param_rank = self._param2rank[param.name]
                if not g_shard_use_reduce:
                    paddle.distributed.all_reduce(
                        g_var,
                        group=hcg.get_sharding_parallel_group(),
                        sync_op=True,
                    )
                else:
                    # TODO(pangengzheng): change to reduce operation when there is no diff in calculating global norm values in HybridParallelClipGrad compared to dp.
                    paddle.distributed.reduce(
                        g_var,
                        dst=hcg.get_sharding_parallel_group().ranks[param_rank],
                        group=hcg.get_sharding_parallel_group(),
                        sync_op=True,
                    )

    def _sharding_sync_parameters(self):
        """
        sync parameter across sharding group
        """
        # TODO speed up this functional

        with framework.no_grad():
            # TODO detach not need (?)
            for rank, params in self._rank2params.items():
                for param in params:
                    paddle.distributed.broadcast(
                        param,
                        # the collective API need src rank to be the global rank id
                        # instead of the relative logic rank id within group
                        src=self._hcg.get_sharding_parallel_group().ranks[rank],
                        group=self._hcg.get_sharding_parallel_group(),
                        sync_op=True,
                    )

    def _update_trainable(self):
        """
        allow user to update trainable parameters list during training
        """
        raise NotImplementedError

    def minimize(
        self, loss, startup_program=None, parameters=None, no_grad_set=None
    ):
        # NOTE in dygraph mode, the only different between step and minimize is that minimize
        # allow user to customize the parameters for updating on each step
        assert (
            not self._using_param_groups
        ), "minimize() is not support if using param_groups"
        input_param_names = {param.name for param in parameters}
        parameters = list(
            filter(
                lambda x: x.name in input_param_names,
                self._rank2params[self._sharding_rank],
            )
        )
        result = self._inner_opt.minimize(
            loss, startup_program, parameters, no_grad_set
        )

        # sync parameters across sharding ranks
        self._sharding_sync_parameters()

        return result

    @imperative_base.no_grad
    @framework.dygraph_only
    def step(self):
        # TODO Check whether the model trainable param changed and update state accordingly
        # hack to grad_clip all parameters,
        # otherwise the self._inner_opt will only grad_clip the self._rank2params[self._sharding_rank] params
        # TODO(pangengzheng): remove the hacked grad_clip codes here when there is no diff in calculating global norm values in HybridParallelClipGrad compared to dp.
        origin_clip = self._inner_opt._grad_clip
        if not self._using_param_groups:
            params_grads = []
            for param in self._parameter_list:
                if (
                    hasattr(param, "regularizer")
                    and param.regularizer is not None
                ):
                    raise ValueError(
                        "param {} should not has the regularizer attribute".format(
                            param.name
                        )
                    )
                if param.stop_gradient:
                    continue
                grad_var = param._grad_ivar()
                if hasattr(param, "main_grad") and param.main_grad is not None:
                    grad_var = param.main_grad
                params_grads.append((param, grad_var))
            if g_shard_norm_align_dp:
                params_grads = self._inner_opt._grad_clip(params_grads)
                # set inner_opt._grad_clip None to avoid repeatedly grad_clip gradients inside inner_opt._apply_optimize
                self._set_inner_opt_attr('_grad_clip', None)
            update_param_names = [
                p.name for p in self._rank2params[self._sharding_rank]
            ]
            update_params_grads = [
                (p, g) for p, g in params_grads if p.name in update_param_names
            ]
            self._apply_optimize(
                loss=None,
                startup_program=None,
                params_grads=update_params_grads,
            )
            if g_shard_norm_align_dp:
                # restore the grad clip
                self._set_inner_opt_attr('_grad_clip', origin_clip)
        else:
            # optimize parameters in groups
            for param_group in self._inner_opt._param_groups:
                params_grads = defaultdict(lambda: [])

                # TODO(shenliang03): support ClipGradByGlobalNorm in sharding when using param_groups
                grad_clip = param_group['grad_clip']
                assert not isinstance(
                    grad_clip, ClipGradByGlobalNorm
                ), "ClipGradByGlobalNorm is not support if using param_groups in sharding"

                for param in param_group['params']:
                    if param.stop_gradient:
                        continue

                    grad_var = param._grad_ivar()
                    if (
                        hasattr(param, "main_grad")
                        and param.main_grad is not None
                    ):
                        grad_var = param.main_grad

                    params_grads['params'].append((param, grad_var))
                params_grads.update(
                    {k: v for k, v in param_group.items() if k != 'params'}
                )
                self._apply_optimize(
                    loss=None, startup_program=None, params_grads=params_grads
                )

        # sync parameters across sharding ranks
        self._sharding_sync_parameters()

    @framework.dygraph_only
    def set_state_dict(self, state_dict):
        inner_state = {}
        parameters = self._rank2params[self._sharding_rank]

        if "LR_Scheduler" in state_dict:
            inner_state["LR_Scheduler"] = state_dict.pop("LR_Scheduler")

        if "master_weights" in state_dict:
            master = state_dict.pop("master_weights")
            inner_state["master_weights"] = {}
            for p in parameters:
                for k, v in master.items():
                    if p.name == k:
                        v.name = self._inner_opt._gen_master_weight_var_name(p)
                        inner_state["master_weights"][k] = v

        for p in parameters:
            for k, v in state_dict.items():
                if p.name in k:
                    inner_state[k] = v

        self._inner_opt.set_state_dict(inner_state)

    def _set_inner_opt_attr(self, attr_name, value):
        inner_opt = self._inner_opt
        inner_opt_name = '_inner_opt'
        if not isinstance(attr_name, str):
            raise TypeError(
                "attr_name should be str type, but is {}".format(
                    type(attr_name)
                )
            )
        while hasattr(inner_opt, attr_name):
            setattr(inner_opt, attr_name, value)
            if (
                hasattr(inner_opt, inner_opt_name)
                and getattr(inner_opt, inner_opt_name, None) is not None
            ):
                inner_opt = getattr(inner_opt, inner_opt_name, None)
            else:
                break

    def __getattr__(self, item):
        return getattr(self._inner_opt, item)


class DygraphShardingOptimizerV2:
    """
    A wrapper for Sharding Optimizer in Dygraph, which split params
    .. warning: DygraphShardingOptimizer is experimental and subject to change.
    .. ZeRO: https://arxiv.org/abs/1910.02054
    """

    # TODO (JZ-LIANG)
    # TO support following featrues in future:
    # 1. fused update parameter sync
    # 2. parameters_groups
    # 3. dynamic trainable params, which is the case bewteen pretraining and finetuning
    # 4. option to choose fuse comm (more GPU MEM need) or un-fuse comm
    # 5. do not shard small params

    def __init__(self, optimizer, hcg):
        logger.info("init DygraphShardingOptimizerV2")
        assert (
            g_shard_use_reduce
        ), "can not be not g_shard_use_reduce if DygraphShardingOptimizerV2 is used"
        # TODO(pangengzheng): support param_groups
        if isinstance(optimizer._parameter_list[0], dict):
            raise TypeError(
                "Do not support param_groups now, please set optimizer._parameter_list as a list of Parameter"
            )
        if not hasattr(optimizer, '_apply_optimize') or not callable(
            optimizer._apply_optimize
        ):
            raise ValueError(
                "the optimzier object should have _apply_optimize function"
            )

        self._inner_opt = optimizer
        self._hcg = hcg
        self._sharding_world_size = self._hcg.get_sharding_parallel_world_size()
        self._sharding_rank = self._hcg.get_sharding_parallel_rank()

        self._parameter_list = optimizer._parameter_list

        # param name -> slice_param
        self._slice_params = {}
        # comm_buffer_list = []
        self._comm_buffer_list = []

        # slice parameter list
        self._local_parameter_list = [
            self._create_slice_param(p) for p in optimizer._parameter_list
        ]

        strategy = fleet.fleet._user_defined_strategy

        self.pp_overlap = strategy.hybrid_configs[
            'pp_configs'
        ].sharding_comm_overlap
        self.pp_release_grads = strategy.hybrid_configs[
            'pp_configs'
        ].release_gradients

        self._set_inner_opt_attr('_parameter_list', self._local_parameter_list)
        self._set_inner_opt_attr('_param_groups', self._local_parameter_list)

        sharding_configs = strategy.hybrid_configs["sharding_configs"]
        acc_steps = sharding_configs.accumulate_steps
        self._enable_timer = strategy.hybrid_configs["enable_optimizer_timer"]

        if self._enable_timer:
            if not timer.is_timer_initialized():
                timer.set_timers()
            self.timers = timer.get_timers()

        comm_group = self._hcg.get_sharding_parallel_group()

        self.comm_overlap = sharding_configs.comm_overlap

        # NOTE(shenliang03): `group_size` will affect the result of the parameter fuse,
        # which in turn affects save/load. Therefore, it is best not to modify 256MB
        # to prevent compatibility issues.
        self._build_comm_buffers(
            comm_group, acc_steps, group_size=256 * 1024 * 1024
        )
        # NOTE(shenliang03): Sort the comm_buffers by dst rank,
        # it will improve the performance in reduce communicate. Default
        # g_shard_sort_reduce_root is True.
        if g_shard_sort_reduce_root:
            self._comm_buffer_list.sort(key=lambda x: x._dst)

        assert (
            not self.comm_overlap or acc_steps > 0
        ), "acc_steps should be larger than 0 when using comm_overlap in sharding"

        assert (
            not self.pp_overlap or not self.comm_overlap
        ), "pp_overlap and comm_overlap should not be True at the same time"

        self._use_pipelie_parallel = strategy.hybrid_configs["pp_degree"] > 1
        assert not (
            self._use_pipelie_parallel and self.comm_overlap
        ), "You should not use pipelie parallel and comm_overlap at the same time"

        if not self.pp_overlap and self.comm_overlap:
            self.register_reduce_overlap_hook(use_comm=True)

    def register_reduce_overlap_hook(self, use_comm):
        # Register backward hooks for each parameter in the buffer
        for buffer in self._comm_buffer_list:
            for param in buffer._params:
                # Directly register the hook function with necessary parameters
                param._register_backward_hook(
                    self._create_backward_hook(buffer, param, use_comm)
                )

    def _create_backward_hook(self, buffer, param, use_comm):
        """Creates a backward hook function for autograd."""

        @paddle.autograd.no_grad()
        def fused_allreduce(*_):
            # Directly add gradient to the buffer
            buffer.add_grad(param, use_comm=use_comm)

        return fused_allreduce

    def _build_comm_buffers(self, comm_group, acc_steps, group_size):
        if self.pp_overlap:
            return

        var_groups = assign_group_by_size(self._parameter_list, group_size)
        for group_idx, parameters in var_groups.items():
            buffer = FusedCommBuffer(
                group_idx,
                parameters,
                comm_group,
                acc_steps,
                act=HOOK_ACTION.REDUCE_SCATTER,
                release_grads=self.pp_release_grads,
            )
            self._comm_buffer_list.append(buffer)

    def clear_grad(self, set_to_zero=True):
        """
        should clear grad for all parameters in model
        """
        if not self.pp_release_grads:
            assert set_to_zero, "should not erase grad buffer"

        def clear_grad_func(p):
            if hasattr(p, "main_grad") and p.main_grad is not None:
                assert p._grad_ivar() is None
                if set_to_zero:
                    p.main_grad.zero_()
                else:
                    p.main_grad._clear()
                    p.main_grad = None
            elif not hasattr(p, "main_grad"):
                p.clear_gradient(set_to_zero)

        for p in self._parameter_list:
            clear_grad_func(p)

        if self.pp_release_grads and not self.pp_overlap:
            for comm_buffer in self._comm_buffer_list:
                comm_buffer._clear_grad_storage()

    def filter_parameters(self, parameter_list, hcg):

        parameter_list = [
            self._slice_params[param.name] for param in parameter_list
        ]
        parameter_list = [
            param for param in parameter_list if param._is_initialized()
        ]
        return parameter_list

    def reduce_gradients(self, parameter_list, hcg):
        with framework.no_grad():
            for buffer in self._comm_buffer_list:
                if self.pp_release_grads and buffer.grad_storage is None:
                    for param in buffer.params:
                        buffer._copy_grad_to_buffer(param)

                if not self.comm_overlap:
                    buffer._comm_grads()

                buffer.scale_and_split_grads()

    def _sharding_sync_parameters(self):
        """
        sync parameter across sharding group
        """
        if self._enable_timer:
            self.timers("sync-parameters").start()

        with framework.no_grad():
            for comm_buffer in self._comm_buffer_list:
                comm_buffer.sync_params()

        if self._enable_timer:
            self.timers("sync-parameters").stop()

    def _update_trainable(self):
        """
        allow user to update trainable parameters list during training
        """
        raise NotImplementedError

    def minimize(
        self, loss, startup_program=None, parameters=None, no_grad_set=None
    ):
        # NOTE in dygraph mode, the only different between step and minimize is that minimize
        # allow user to customize the parameters for updating on each step
        raise AssertionError("not supported yet")

    def _create_slice_param(self, param):
        # the buffer underlining is not initialized yet
        slice_param = EagerParamBase(shape=[1], dtype=param.dtype)
        slice_param.name = param.name
        if hasattr(param, "is_distributed"):
            slice_param.is_distributed = param.is_distributed

        def copy_attr(attr_name):
            setattr(slice_param, attr_name, getattr(param, attr_name))

        copy_attr("optimize_attr")
        copy_attr("do_model_average")
        copy_attr("need_clip")

        self._slice_params[param.name] = slice_param
        return slice_param

    def _collect_comm_buffers(self):
        if self._comm_buffer_list:
            return
        for param in self._parameter_list:
            if not hasattr(param, "comm_buffer_ref"):
                continue
            comm_buffer_ref = param.comm_buffer_ref
            del param.comm_buffer_ref
            comm_buffer = comm_buffer_ref()
            self._comm_buffer_list.append(comm_buffer)

        assert self._comm_buffer_list

    def _assign_slice_grad(self):
        param_num = 0
        for comm_buffer in self._comm_buffer_list:
            param_num = param_num + len(comm_buffer.params)
            for param in comm_buffer.params:
                assert param.name in self._slice_params
                slice_param = self._slice_params[param.name]
                if self.pp_release_grads and hasattr(slice_param, "main_grad"):
                    assert not slice_param.main_grad._is_initialized()
                    del slice_param.main_grad
                comm_buffer.assign_slice_grad(param, slice_param)
        assert param_num == len(self._parameter_list)

    def step(self):
        # TODO Check whether the model trainable param changed and update state accordingly
        # hack for pp comm overlap
        self._collect_comm_buffers()
        self._assign_slice_grad()

        if self._enable_timer:
            self.timers("apply-optimize").start()

        if not isinstance(self._parameter_list[0], dict):
            params_grads = []
            for param in self._parameter_list:
                if (
                    hasattr(param, "regularizer")
                    and param.regularizer is not None
                ):
                    raise ValueError(
                        f"param {param.name} should not has the regularizer attribute"
                    )
                if param.stop_gradient:
                    continue
                # update on slice
                assert param.name in self._slice_params
                param = self._slice_params[param.name]
                grad_var = param._grad_ivar()
                if hasattr(param, "main_grad") and param.main_grad is not None:
                    grad_var = param.main_grad
                if grad_var is not None:
                    params_grads.append((param, grad_var))

            self._apply_optimize(
                loss=None,
                startup_program=None,
                params_grads=params_grads,
            )

        if self._enable_timer:
            self.timers("apply-optimize").stop()

        # sync parameters across sharding ranks
        self._sharding_sync_parameters()

    @framework.dygraph_only
    def set_state_dict(self, state_dict):
        inner_state = {}
        parameters = self._parameter_list

        if "LR_Scheduler" in state_dict:
            inner_state["LR_Scheduler"] = state_dict.pop("LR_Scheduler")

        if "master_weights" in state_dict:
            master = state_dict.pop("master_weights")
            inner_state["master_weights"] = {}
            for p in parameters:
                for k, v in master.items():
                    if p.name == k:
                        v.name = self._inner_opt._gen_master_weight_var_name(p)
                        inner_state["master_weights"][k] = v

        for p in parameters:
            for k, v in state_dict.items():
                if p.name in k:
                    inner_state[k] = v

        self._inner_opt.set_state_dict(inner_state)

    def _set_inner_opt_attr(self, attr_name, value):
        inner_opt = self._inner_opt
        inner_opt_name = '_inner_opt'
        if not isinstance(attr_name, str):
            raise TypeError(
                f"attr_name should be str type, but is {type(attr_name)}"
            )
        while hasattr(inner_opt, attr_name):
            setattr(inner_opt, attr_name, value)
            inner_opt = getattr(inner_opt, inner_opt_name, None)
            if inner_opt is None:
                break

    def __getattr__(self, item):
        return getattr(self._inner_opt, item)
