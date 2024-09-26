# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from collections import OrderedDict

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.base.libpaddle import pir
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.static.reshard_funcs.nd_mesh_reshard_func import (
    get_1D_sub_process_mesh,
)
from paddle.distributed.fleet.utils.tensor_fusion_helper import (
    align,
    get_current_device_type,
)
from paddle.optimizer import Optimizer


class ShardingStage1Optimizer(Optimizer):
    """
    .. ZeRO: https://arxiv.org/abs/1910.02054

    """

    def __init__(self, optimizer, shard_fn=None):
        assert (
            optimizer is not None
        ), "The argument `optimizer` cannot be empty."
        assert isinstance(
            optimizer, (paddle.optimizer.AdamW, paddle.optimizer.SGD)
        ), "`paddle.distributed.ShardOptimizer` only supports AdamW and SGD optimizer for now."

        self.__dict__["_inner_opt"] = optimizer

        self._shard_fn = shard_fn
        self._sharding_mesh_axis = None
        self._sharding_degree = None
        self._set_and_check_sharding_prop_from_param()
        self._shard_fn._set_sharding_mesh_axis(self._sharding_mesh_axis)

    def _set_and_check_sharding_prop_from_param(self):
        if (self._shard_fn._mesh is not None) and (
            len(self._shard_fn._mesh._shape) == 1
        ):
            self._sharding_degree = self._shard_fn._mesh.get_dim_size(0)
            self._sharding_mesh_axis = 0
        else:
            param_list = self._inner_opt._parameter_list
            for param in param_list:
                if not param.is_dist():
                    continue
                mesh = param.process_mesh
                placements = param.placements

                if self._sharding_degree is None:
                    # set the sharding degree if it has not been set
                    if any(
                        isinstance(placement, dist.Shard)
                        for placement in placements
                    ):
                        for idx, placement in enumerate(placements):
                            if isinstance(placement, dist.Replicate):
                                self._sharding_degree = mesh.dim_size(idx)
                                self._sharding_mesh_axis = idx
                                break
                else:
                    # check the placement on sharding axis is Replicate
                    assert isinstance(
                        placements[self._sharding_mesh_axis], dist.Replicate
                    ), "The placement on sharding_mesh_axis should be Replicate"

                    # check the sharding degree since it has already been set
                    assert (
                        mesh.dim_size(self._sharding_mesh_axis)
                        == self._sharding_degree
                    ), "The sharding degree of all parameters must be equal currently."

        assert (
            self._sharding_degree is not None
        ), "The sharding degree is None in ShardOptimizer"

    def apply_gradients(self, params_grads):
        strategy = fleet.fleet._user_defined_strategy
        sharding_config = strategy.hybrid_configs['sharding_configs']
        comm_buffer_size_MB = sharding_config.comm_buffer_size_MB
        parameters = []
        grads = []
        for param, grad in params_grads:
            parameters.append(param)
            grads.append(grad)
        group_size = comm_buffer_size_MB * 1024 * 1024
        group_indices = pir.assign_value_group_by_size(
            parameters, [group_size, group_size]
        )
        var_groups = OrderedDict()
        for group_idx, indices in enumerate(group_indices):
            group_size = 0
            group_param_list = []
            group_grad_list = []
            for index in indices:
                var_groups.setdefault(group_idx, []).append(
                    parameters[index].name
                )
                group_param_list.append(parameters[index])
                group_grad_list.append(grads[index])
                # group_size += np.prod(parameters[index].shape)
            fused_param = self._fuse_group_param(group_param_list)
            dtype = grads[0].dtype
            align_size = (
                fleet.utils.tensor_fusion_helper.alignment[
                    get_current_device_type()
                ]
                // align[dtype]
            )
            align_size = align_size * self._sharding_degree
            _, fused_grad = paddle._C_ops.coalesce_tensor_(
                group_grad_list,
                dtype,
                True,
                False,
                False,
                0.0,
                True,
                align_size,
                -1,
                [],
                [],
            )
            assert (
                fused_param.shape == fused_grad.shape
            ), f"The current group's fused_param shape is {fused_param.shape}, fuse_grad shape is {fused_grad.shape}. The two not equal."
        print(paddle.static.default_main_program())
        print(paddle.static.default_startup_program())
        return var_groups

    def _fuse_group_param(self, group_param_list):
        startup_program = paddle.static.default_startup_program()
        main_program = paddle.static.default_main_program()
        with paddle.static.program_guard(startup_program):

            def get_param_from_startup(startup, name):
                for op in startup.global_block().ops:
                    if (
                        op.name() == 'builtin.set_parameter'
                        and name == op.attrs()['parameter_name']
                    ):
                        return op.operand(0).source()
                raise ValueError(
                    f"can't find param ({name}) in startup program"
                )

            startup_param_list = []
            fuse_param_name = "fused"
            for param in group_param_list:
                startup_param = get_param_from_startup(
                    startup_program, param.name
                )
                startup_param_list.append(startup_param)
                fuse_param_name = fuse_param_name + "_" + param.name
            dtype = startup_param_list[0].dtype
            align_size = (
                fleet.utils.tensor_fusion_helper.alignment[
                    get_current_device_type()
                ]
                // align[dtype]
            )
            align_size = align_size * self._sharding_degree
            _, fused_param = paddle._C_ops.coalesce_tensor(
                startup_param_list,
                dtype,
                True,
                False,
                False,
                0.0,
                True,
                align_size,
                -1,
                [],
                [],
            )
            fused_param.persistable = True
            paddle._pir_ops.set_persistable_value(fused_param, fuse_param_name)
            # hcg = fleet.fleet._hcg
            sub_mesh = get_1D_sub_process_mesh(
                fused_param.process_mesh, self._sharding_mesh_axis
            )
            group_size = fused_param.shape[0]
            shard_size = group_size // self._sharding_degree
            rank = sub_mesh.process_ids.index(dist.get_rank())
            rank_begin = rank * shard_size
            rank_end = rank_begin + shard_size
            total_buffer_size = 0
            print("comm_group.rank  is", rank)
            print("shard size is", shard_size)
            print("rank begin is:", rank_begin, "rank_end is:", rank_end)
            for param in group_param_list:
                size = np.prod(param.shape)
                padded_size = (
                    (size + align_size - 1) // align_size
                ) * align_size
                param_begin = max(total_buffer_size, rank_begin)
                total_buffer_size += padded_size
                param_end = min(total_buffer_size, rank_end)
                print(
                    "called paddle._C_ops.tensor_slice",
                    param.name,
                    "param_begin is ",
                    param_begin,
                    "param_end is ",
                    param_end,
                    flush=1,
                )
                if param_begin < param_end:
                    slice_param = paddle._C_ops.tensor_slice(
                        fused_param, param_begin, param_end
                    )
                    print("called paddle._C_ops.tensor_slice_", flush=1)

        return fused_param

    def _pir_create_optimization_pass(
        self, parameters_and_grads, param_group_idx=0
    ):
        """Add optimization operators to update gradients to tensors.

        Args:
          parameters_and_grads(list(tuple(Tensor, Tensor))):
            a list of (tensor, gradient) pair to update.

        Returns:
          return_op_list: a list of operators that will complete one step of
            optimization. This will include parameter update ops, global step
            update ops and any other custom ops required by subclasses to manage
            their internal state.
        """
        # Accessing user defined strategy
        print(self.param_groups, flush=1)
        for para, grad in parameters_and_grads:
            print(para.name)
        # strategy = fleet.fleet._user_defined_strategy
        # sharding_config = strategy.hybrid_configs['sharding_configs']
        # comm_buffer_size_MB = sharding_config.comm_buffer_size_MB
        # self._build_comm_buffers(parameters_and_grads, group_size = comm_buffer_size_MB * 1024 * 1024)

        # print(parameters_and_grads)
        # print(paddle.static.default_main_program())
        # print(paddle.static.default_startup_program())
        # print("************************* find errir!", flush=1)

    def __getattr__(self, item):
        if "_inner_opt" in self.__dict__:
            if item == "_inner_opt":
                return self.__dict__[item]
            return getattr(self.__dict__["_inner_opt"], item)
        else:
            raise AttributeError

    def __setattr__(self, item, value):
        if item == '_inner_opt':
            msg = f'{type(self).__name__}._inner_opt is READ ONLY'
            raise AttributeError(msg)
        return setattr(self._inner_opt, item, value)
