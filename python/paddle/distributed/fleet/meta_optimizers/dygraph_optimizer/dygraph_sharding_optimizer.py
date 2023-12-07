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
import warnings
from distutils.util import strtobool
from functools import reduce

import paddle
from paddle import framework
from paddle.base.dygraph import base as imperative_base
from paddle.base.framework import EagerParamBase
from paddle.distributed import fleet

from ...utils.log_util import logger
from ...utils.tensor_fusion_helper import (
    HOOK_ACTION,
    FusedCommBuffer,
    assign_group_by_size,
    fused_parameters,
)

g_shard_use_reduce = int(os.environ.get("FLAGS_shard_use_reduce", 1))
g_shard_norm_align_dp = int(os.environ.get("FLAGS_shard_norm_align_dp", 0))

if g_shard_norm_align_dp:
    assert (
        not g_shard_use_reduce
    ), "g_shard_norm_align_dp is not supported if g_shard_use_reduce is true"


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
        # the self._parameter_list holds the whole model paramters
        self._parameter_list = optimizer._parameter_list
        self._origin_parameter_list = self._parameter_list
        self._inner_opt = optimizer
        self._hcg = hcg
        self._sharding_world_size = self._hcg.get_sharding_parallel_world_size()
        self._sharding_rank = self._hcg.get_sharding_parallel_rank()

        strategy = fleet.fleet._user_defined_strategy
        self.tensor_fusion = strategy.hybrid_configs[
            'sharding_configs'
        ].tensor_fusion

        self.accumulate_steps = strategy.hybrid_configs[
            'sharding_configs'
        ].accumulate_steps
        self.comm_overlap = strategy.hybrid_configs[
            'sharding_configs'
        ].comm_overlap
        self.fuse_optimizer = strategy.hybrid_configs[
            'sharding_configs'
        ].fuse_optimizer
        pp_overlap = strategy.hybrid_configs['pp_configs'].sharding_comm_overlap
        if self.tensor_fusion or self.comm_overlap:
            assert (
                not pp_overlap
            ), "Can not enable pp's sharding_comm_overlap and sharding's tensor_fusion at the same time."

        self._use_main_grad = hasattr(self._parameter_list[0], "main_grad")
        self._rank2decay = {}
        self._rank2fused = {}
        self._comm_buffers = []

        self._rank2params = self._partition_parameters()
        self._param2rank = self._map_param_to_rank()

        if not self.tensor_fusion and not self.comm_overlap:
            local_params = self._rank2params[self._sharding_rank]
            self._set_inner_opt_attr('_parameter_list', local_params)
            self._set_inner_opt_attr('_param_groups', local_params)
        else:
            if self.fuse_optimizer:
                lr = None
                for param in self._origin_parameter_list:
                    if hasattr(param, "optimize_attr"):
                        param_lr = param.optimize_attr['learning_rate']
                        if lr is None:
                            lr = param_lr
                        elif lr != param_lr:
                            warnings.warn(
                                "Parameters have different learning rate, "
                                "won't do fusion on the optimizer."
                            )
                            self.fuse_optimizer = False
                            break
            self.origin_decay_param_fun = getattr(
                self._inner_opt, '_apply_decay_param_fun', None
            )
            self._tensor_fusion()

            decay_params = [
                p.name for p in self._rank2decay[self._sharding_rank]
            ]
            local_fused_params = self._rank2fused[self._sharding_rank]
            apply_decay_param_fun = lambda x: x in decay_params

            all_fused_params = []
            for v in self._rank2fused.values():
                all_fused_params += v
            self._parameter_list = all_fused_params
            self._param_groups = all_fused_params

            self._set_inner_opt_attr('_parameter_list', local_fused_params)
            self._set_inner_opt_attr('_param_groups', local_fused_params)
            if self.comm_overlap:
                # Only set local param for check finite when comm overlap.
                # Under comm overlap, all grads will be communicated before check_finite.
                # Therefore, each sharding rank can get all grads' info at check_finite.
                # Without comm overlap, all grads will be communicated after check_finite,
                # which means each sharding rank should do check_finite to all grads.
                self._local_parameter_list = local_fused_params
            if self.origin_decay_param_fun is not None:
                self._set_inner_opt_attr(
                    '_apply_decay_param_fun', apply_decay_param_fun
                )
            # Note: during the tensor fusion for parameters, the allocator will apply for
            # some extra GPU memory for the fused big paramters. This extra GPU memory will
            # be useless at once the fusion has done. But the Paddle's allocator won't
            # release those memory, it will hold that part in the memory poll. So after
            # tensor fusion, the 'reserved' memory will increase but the 'allocate' memory
            # won't change. To avoid failure on some other applications (such as some nvtx
            # operations), here we manulay let the allocator release the cached memory.
            paddle.device.cuda.empty_cache()

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
                if self.tensor_fusion:
                    if set_to_zero:
                        p.grad.zero_()
                    else:
                        p.grad._clear()
                        p.grad = None
                else:
                    p.clear_gradient(set_to_zero)

    def _tensor_fusion(self):
        comm_group = self._hcg.get_sharding_parallel_group()
        for i in range(self._sharding_world_size):
            params = self._rank2params[i]
            dst = comm_group.ranks[i]
            # TODO(sharding dev): make scale_after_comm a field to be configured by user
            decay_fused, all_fused, all_buffer = fused_parameters(
                params,
                use_main_grad=self._use_main_grad,
                fuse_param=True,
                comm_overlap=self.comm_overlap,
                comm_group=comm_group,
                dst=dst,
                acc_step=self.accumulate_steps,
                scale_after_comm=False,
                apply_decay_param_fun=self.origin_decay_param_fun,
            )
            if self.comm_overlap:
                self._comm_buffers += all_buffer
            self._rank2decay[i] = decay_fused
            self._rank2fused[i] = all_fused
            for p in all_fused:
                self._param2rank[p.name] = i

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
            numel = reduce(lambda x, y: x * y, param.shape, 1)
            assert (
                numel > 0
            ), f"param [{param.name}] should larger than 0, but it is [{numel}]"
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

    def filter_parameters(self, parameter_list, hcg):
        sharding_parallel_rank = hcg.get_sharding_parallel_rank()
        parameter_list = [
            param
            for param in parameter_list
            if self._param2rank[param.name] == sharding_parallel_rank
        ]
        return parameter_list

    def reduce_gradients(self, parameter_list, hcg):
        # TODO merge grad / nrank with dp
        logger.debug("sharding start gradients sync")
        if self.comm_overlap:
            for buffer in self._comm_buffers:
                buffer.scale_grads()
            return
        with framework.no_grad():
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
                            dst=hcg.get_sharding_parallel_group().ranks[
                                param_rank
                            ],
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
            valid_rank_to_params = (
                self._rank2params
                if not self.tensor_fusion
                else self._rank2fused
            )
            for rank, params in valid_rank_to_params.items():
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
        target_param_list = (
            self._origin_parameter_list
            if (not self.tensor_fusion or not self.fuse_optimizer)
            else self._parameter_list
        )
        if not isinstance(target_param_list[0], dict):
            params_grads = []
            for param in target_param_list:
                if (
                    hasattr(param, "regularizer")
                    and param.regularizer is not None
                ):
                    raise ValueError(
                        f"param {param.name} should not has the regularizer attribute"
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
            rank_params = (
                self._rank2params[self._sharding_rank]
                if (not self.tensor_fusion or not self.fuse_optimizer)
                else self._rank2fused[self._sharding_rank]
            )
            update_param_names = [p.name for p in rank_params]
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
                f"attr_name should be str type, but is {type(attr_name)}"
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
        ), "g_shard_use_reduce must be true if DygraphShardingOptimizerV2 is used"

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

        # Accessing user defined strategy
        strategy = fleet.fleet._user_defined_strategy
        sharding_config = strategy.hybrid_configs['sharding_configs']
        pp_config = strategy.hybrid_configs['pp_configs']

        # Asserting tensor fusion not supported
        self.tensor_fusion = sharding_config.tensor_fusion
        assert not self.tensor_fusion, "not supported yet"

        # Setting accumulate steps and communication overlap
        acc_steps = sharding_config.accumulate_steps
        self.comm_overlap = sharding_config.comm_overlap

        # Setting pipeline parallelism overlap
        self.pp_overlap = pp_config.sharding_comm_overlap

        # TODO(liuzhenhai):support it latter
        assert not self.comm_overlap, "not supported yet"

        self._build_comm_buffers(acc_steps)
        # NOTE(shenliang03): Sort the comm_buffers by dst rank,
        # it will improve the performance in reduce communicate. Default
        # g_shard_sort_reduce_root is True.
        self._comm_buffer_list.sort(key=lambda x: x._dst)

        self._set_inner_opt_attr('_parameter_list', self._local_parameter_list)
        self._set_inner_opt_attr('_param_groups', self._local_parameter_list)

        # Ensure acc_steps is greater than 0 when comm_overlap is used
        if self.comm_overlap:
            assert (
                acc_steps > 0
            ), "acc_steps should be larger than 0 when using comm_overlap in sharding"

        # Ensure pp_overlap and comm_overlap are not both True
        assert not (
            self.pp_overlap and self.comm_overlap
        ), "pp_overlap and comm_overlap should not be True at the same time"

        # Determine the use of pipeline parallelism
        self._use_pipeline_parallel = strategy.hybrid_configs["pp_degree"] > 1

        # Ensure pipelie parallel and comm_overlap are not used together
        if self._use_pipeline_parallel:
            assert (
                not self.comm_overlap
            ), "You should not use pipeline parallel and comm_overlap at the same time"

        # Register reduce overlap hook if comm_overlap is used without pp_overlap
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

    def _build_comm_buffers(self, acc_steps, group_size=256 * 1024 * 1024):
        if self.pp_overlap:
            return

        comm_group = self._hcg.get_sharding_parallel_group()
        var_groups = assign_group_by_size(self._parameter_list, group_size)
        for group_idx, parameters in var_groups.items():
            buffer = FusedCommBuffer(
                group_idx,
                parameters,
                comm_group,
                acc_steps,
                act=HOOK_ACTION.REDUCE_SCATTER,
            )
            self._comm_buffer_list.append(buffer)

    def clear_grad(self, set_to_zero=True):
        """
        should clear grad for all parameters in model
        """
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
                if self.tensor_fusion:
                    if set_to_zero:
                        p.grad.zero_()
                    else:
                        p.grad._clear()
                        p.grad = None
                else:
                    p.clear_gradient(set_to_zero)

        for p in self._parameter_list:
            clear_grad_func(p)

    def filter_parameters(self, parameter_list, hcg):
        parameter_list = [
            self._slice_params[param.name] for param in parameter_list
        ]
        parameter_list = [
            param for param in parameter_list if param._is_initialized()
        ]
        return parameter_list

    def reduce_gradients(self, parameter_list, hcg):
        # TODO merge grad / nrank with dp
        logger.debug("sharding start gradients sync")
        with framework.no_grad():
            for comm_buffer in self._comm_buffer_list:
                if not self.comm_overlap:
                    comm_buffer._comm_grads()

                comm_buffer.scale_grads()

    def _sharding_sync_parameters(self):
        """
        sync parameter across sharding group
        """

        logger.debug("sharding start sync parameters")
        with framework.no_grad():
            for comm_buffer in self._comm_buffer_list:
                comm_buffer.sync_params()

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
        # not initialized yet
        slice_param = EagerParamBase(shape=[1], dtype=param.dtype)
        slice_param.name = param.name

        def copy_attr(attr_name):
            if hasattr(param, attr_name):
                setattr(slice_param, attr_name, getattr(param, attr_name))

        copy_attr("is_distributed")
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
                comm_buffer.assign_slice_grad(param, slice_param)

        assert param_num == len(self._parameter_list)

    def step(self):
        # TODO Check whether the model trainable param changed and update state accordingly
        # hack for pp comm overlap
        self._collect_comm_buffers()
        self._assign_slice_grad()

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
