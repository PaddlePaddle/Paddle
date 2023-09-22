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
from functools import reduce

import paddle
from paddle import framework
from paddle.base.framework import EagerParamBase
from paddle.distributed import fleet

from ...utils.log_util import logger
from ...utils.tensor_fusion_helper import fused_parameters

g_shard_use_reduce = int(os.environ.get("FLAGS_shard_use_reduce", 1))
g_shard_norm_align_dp = int(os.environ.get("FLAGS_shard_norm_align_dp", 0))
g_shard_split_param = int(os.environ.get("FLAGS_shard_split_param", 0))

if g_shard_norm_align_dp:
    assert (
        not g_shard_use_reduce
    ), "g_shard_norm_align_dp is not supported if g_shard_use_reduce is true"
    assert (
        not g_shard_split_param
    ), "g_shard_norm_align_dp is not supported if g_shard_split_param is true"


def _is_trainable(param):
    return not param.stop_gradient


class DygraphShardingOptimizerV1:
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
            origin_decay_param_fun = getattr(
                self._inner_opt, '_apply_decay_param_fun', None
            )
            if origin_decay_param_fun is not None:
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
        for param in self._parameter_list:
            rank = sizes.index(min(sizes))
            mapping[rank].append(param)
            numel = reduce(lambda x, y: x * y, param.shape, 1)
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

        logger.debug("sharding start sync parameters")
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

    def step(self):
        # TODO Check whether the model trainable param changed and update state accordingly

        # hack to grad_clip all parameters,
        # otherwise the self._inner_opt will only grad_clip the self._rank2params[self._sharding_rank] params
        # TODO(pangengzheng): remove the hacked grad_clip codes here when there is no diff in calculating global norm values in HybridParallelClipGrad compared to dp.
        origin_clip = self._inner_opt._grad_clip
        if not isinstance(self._parameter_list[0], dict):
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
            rank_params = (
                self._rank2params[self._sharding_rank]
                if not self.tensor_fusion
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

        # slice parameter list
        self._local_parameter_list = [
            self._create_slice_param(p) for p in optimizer._parameter_list
        ]

        # padded_param_buffer
        self._padded_param_buffer = {}

        # padded_param_grad
        self._padded_grad_buffer = {}

        strategy = fleet.fleet._user_defined_strategy
        self.tensor_fusion = strategy.hybrid_configs[
            'sharding_configs'
        ].tensor_fusion

        assert not self.tensor_fusion, "not supported yet"

        self.accumulate_steps = strategy.hybrid_configs[
            'sharding_configs'
        ].accumulate_steps
        self.comm_overlap = strategy.hybrid_configs[
            'sharding_configs'
        ].comm_overlap
        pp_overlap = strategy.hybrid_configs['pp_configs'].sharding_comm_overlap

        assert not self.comm_overlap, "not supported yet"
        assert not pp_overlap, "not supported yet"

        self._set_inner_opt_attr('_parameter_list', self._local_parameter_list)
        self._set_inner_opt_attr('_param_groups', self._local_parameter_list)

    def clear_grad(self, set_to_zero=True):
        """
        should clear grad for all parameters in model
        """

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
            if hasattr(p, "shard_slice"):
                clear_grad_func(p.shard_slice)

        if not set_to_zero:
            for k, v in self._padded_grad_buffer.items():
                v._clear_data()
            self._padded_grad_buffer = {}

    def reduce_gradients(self, parameter_list, hcg):
        # TODO merge grad / nrank with dp
        logger.debug("sharding start gradients sync")
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
                    g_var = self._get_padded_grad(param, g_var)
                    # reduce scatter
                    assert g_var._numel() % self._sharding_world_size == 0
                    # reuse grad buffer
                    shard_size = g_var._numel() // self._sharding_world_size
                    begin = shard_size * self._sharding_rank
                    end = begin + shard_size
                    reduce_scattered = g_var._slice(begin, end)
                    paddle.distributed.reduce_scatter(
                        reduce_scattered,
                        g_var,
                        group=self._hcg.get_sharding_parallel_group(),
                    )
                    self._assign_slice_grad(param, reduce_scattered)

    def _sharding_sync_parameters(self):
        """
        sync parameter across sharding group
        """
        # TODO speed up this functional

        logger.debug("sharding start sync parameters")
        with framework.no_grad():
            params = zip(self._parameter_list, self._local_parameter_list)
            for full, slice in params:
                gather_list = []
                paddle.distributed.all_gather(
                    gather_list,
                    slice,
                    group=self._hcg.get_sharding_parallel_group(),
                    sync_op=True,
                )
                padded_full = paddle.concat(gather_list, axis=0)
                self._assign_param(full, padded_full)

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

    def _fill_slice_param(self, p):
        """
        create padded buffer for parameter
        """
        shard_slice = p.shard_slice
        begin = self._sharding_rank * shard_slice._numel()
        end = begin + shard_slice._numel()
        if p.name not in self._padded_param_buffer:
            padded_buffer = self._create_padded_buffer(p)
            padded_buffer._slice(0, p._numel())._share_buffer_to(p)
            self._padded_param_buffer[p.name] = padded_buffer
        padded_buffer = self._padded_param_buffer[p.name]
        if not padded_buffer._is_shared_buffer_with(p):
            del self._padded_param_buffer[p.name]
            self._fill_slice_param(p)
            return
        padded_buffer._slice(begin, end)._share_buffer_to(shard_slice)

    def _create_slice_param(self, param):
        # partition param to slice
        padded_size = self._get_padded_size(param)
        shard_size = padded_size // self._sharding_world_size
        param_slice = EagerParamBase(shape=[shard_size], dtype=param.dtype)
        param_slice.name = f"shard@{param.name}"
        param.shard_slice = param_slice
        return param_slice

    def _get_padded_size(self, t, align_dtype=None):
        if align_dtype is None:
            align_dtype = t.dtype
        device_alignment = 256
        type_align = {
            paddle.float16.value: 2,
            paddle.bfloat16.value: 2,
            paddle.float32.value: 4,
        }
        align_size = device_alignment // type_align[align_dtype]
        align_size = align_size * self._sharding_world_size
        size = t._numel()
        padded_size = ((size + align_size - 1) // align_size) * align_size
        return padded_size

    def _create_padded_buffer(self, t, align_dtype=None):
        if align_dtype is None:
            align_dtype = t.dtype
        # extend grad to padded buffer
        size = t._numel()
        padded_size = self._get_padded_size(t, align_dtype)
        with framework.no_grad():
            # no need padding
            if size == padded_size:
                return t.reshape([-1])
            # TODO(liuzhnehai): check capicity, if capicity is enough, reallocation can be avoided
            t_shape = t.shape
            t.flatten_()
            t_padded = paddle.zeros(shape=[padded_size], dtype=t.dtype)
            t_padded[0:size] = t
            t.get_tensor()._set_dims(t_shape)
            return t_padded

    def _assign_param(self, param, padded_buffer):
        # recover param from padded buffer
        size = param._numel()
        padded_size = padded_buffer._numel()
        assert padded_size >= size
        with framework.no_grad():
            padded_buffer._share_buffer_to(param)
            self._padded_param_buffer[param.name] = padded_buffer
            param.shard_slice._clear_data()

    def _get_padded_grad(self, param, g_var):
        if param.name not in self._padded_grad_buffer:
            padded_grad = self._create_padded_buffer(g_var, param.dtype)
            self._padded_grad_buffer[param.name] = padded_grad
            return padded_grad
        padded_grad = self._padded_grad_buffer[param.name]
        if not padded_grad._is_shared_buffer_with(g_var):
            del self._padded_grad_buffer[param.name]
            return self._get_padded_grad(param, g_var)
        return padded_grad

    def _assign_slice_grad(self, param, slice_grad):
        # assign slice grad to parameter
        with framework.no_grad():
            assert hasattr(param, "shard_slice")
            self._fill_slice_param(param)
            param._clear_data()
            shard_slice = param.shard_slice
            if hasattr(param, "main_grad"):
                # TODO(liuzhnehai): share buffer to if not None
                shard_slice.main_grad = slice_grad
            else:
                shard_slice._copy_gradient_from(slice_grad)

    def step(self):
        # TODO Check whether the model trainable param changed and update state accordingly

        if not isinstance(self._parameter_list[0], dict):
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
                # update on slice
                assert hasattr(param, "shard_slice")
                param = param.shard_slice
                grad_var = param._grad_ivar()
                if hasattr(param, "main_grad") and param.main_grad is not None:
                    grad_var = param.main_grad
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
        parameters = self._rank2params[self._sharding_rank]

        if "LR_Scheduler" in state_dict:
            inner_state["LR_Scheduler"] = state_dict.pop("LR_Scheduler")

        if "master_weights" in state_dict:
            master = state_dict.pop("master_weights")
            inner_state["master_weights"] = {}
            for p in parameters:
                for k, v in master.items():
                    if f"shard@{p.name}" == k:
                        v.name = self._inner_opt._gen_master_weight_var_name(p)
                        inner_state["master_weights"][k] = v

        for p in parameters:
            for k, v in state_dict.items():
                if f"shard@{p.name}" in k:
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


DygraphShardingOptimizer = (
    DygraphShardingOptimizerV2
    if g_shard_split_param
    else DygraphShardingOptimizerV1
)
