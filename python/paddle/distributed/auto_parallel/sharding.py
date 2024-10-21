# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.distributed.auto_parallel.static.process_group import (
    new_process_group,
)
from paddle.distributed.auto_parallel.static.reshard_funcs.nd_mesh_reshard_func import (
    get_1D_sub_process_mesh,
)
from paddle.distributed.fleet.utils.tensor_fusion_helper import (
    align,
    get_current_device_type,
)
from paddle.optimizer import Optimizer


class ShardingOptimizerStage1(Optimizer):
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
        sub_mesh = get_1D_sub_process_mesh(mesh, self._sharding_mesh_axis)
        self.sharding_group = new_process_group(sorted(sub_mesh.process_ids))
        assert (
            self._sharding_degree == self.sharding_group.nranks
        ), "The sharding degree must be equal to sharding_group size. but received {self._sharding_degree} and self.sharding_group.nranks()"

    def apply_gradients(self, params_grads):
        strategy = fleet.fleet._user_defined_strategy
        sharding_config = strategy.hybrid_configs['sharding_configs']
        comm_buffer_size_MB = sharding_config.comm_buffer_size_MB
        parameters = []
        grads = []
        for param, grad in params_grads:
            param_dist_attr = param.dist_attr()
            grad_dist_attr = grad.dist_attr()
            assert (
                param_dist_attr is not None
            ), f"parameter dist attribute must not None. but received {param.name} : {param}."
            assert (
                grad_dist_attr is not None
            ), f"gradient dist attribute must not None. but received {param.name} grad : {grad}."
            assert (
                param_dist_attr.process_mesh == grad_dist_attr.process_mesh
            ), f"Parameter and grad should have same process_mesh. but received name:{param.name}, parameter:{param}, grad: {grad}."
            assert (
                self._sharding_mesh_axis in grad_dist_attr.partial_dims
            ), f"gradient should partial in sharding mesh axis. but received parameter name:{param.name}, sharding_mesh_axis:{self._sharding_mesh_axis}, grad: {grad}."
            sub_mesh = get_1D_sub_process_mesh(
                param_dist_attr.process_mesh, self._sharding_mesh_axis
            )
            assert (
                sorted(sub_mesh.process_ids) == self.sharding_group.ranks
            ), f" all parameter must have the same sharding group. but received {param.name} sharding group is : {sub_mesh.process_ids}, global sharding group is: {self.sharding_group.ranks}"

            assert (
                param_dist_attr.partial_dims == set()
            ), f"Sharding fusion do not support parital parameter. but received {param.name} : {param}."
            assert (
                param_dist_attr.dims_mapping == grad_dist_attr.dims_mapping
            ), f"Parameter and grad should have same dims_mapping. but received name:{param.name}, parameter:{param}, grad: {grad}."
            assert (
                param.shape == grad.shape
            ), f"Parameter and grad should have same global shape. but received name:{param.name}, parameter:{param}, grad: {grad}."
            assert (
                param._local_shape == grad._local_shape
            ), f"Parameter and grad should have same local shape. but received name:{param.name}, parameter:{param}, grad: {grad}."
            parameters.append(param)
            grads.append(grad)

        main_program = paddle.static.default_main_program()
        target_block = main_program.global_block()
        last_op = target_block.ops[-1]

        group_size = comm_buffer_size_MB * 1024 * 1024
        group_indices = pir.assign_value_group_by_size(
            parameters, [group_size, group_size]
        )
        var_groups = OrderedDict()
        new_params_grads = []
        all_gather_param_info_list = []
        for group_idx, indices in enumerate(group_indices):
            group_param_list = []
            group_grad_list = []
            for index in indices:
                var_groups.setdefault(group_idx, []).append(
                    parameters[index].name
                )
                group_param_list.append(parameters[index])
                group_grad_list.append(grads[index])
                grads[index].persistable = True

            slice_param_dict, main_shard_fused_param, main_fused_param = (
                self._fuse_group_param(group_param_list)
            )
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
            fused_grad.persistable = True
            shard_size = fused_grad._local_shape[0] // self._sharding_degree
            rank = self.sharding_group.ranks.index(dist.get_rank())
            rank_begin = rank * shard_size
            rank_end = rank_begin + shard_size
            view_shard_fused_grad = paddle._C_ops.tensor_slice(
                fused_grad, rank_begin, rank_end
            )

            shard_fused_grad = paddle._C_ops.reduce_scatter(
                fused_grad, self.sharding_group.id, self._sharding_degree
            )

            paddle._C_ops.share_var([view_shard_fused_grad, shard_fused_grad])
            all_gather_param_info_list.append(
                (
                    main_shard_fused_param,
                    main_fused_param,
                )
            )

            for slice_param, param_info in slice_param_dict.items():
                index, param_begin, param_end = param_info
                slice_grad = paddle._C_ops.tensor_slice(
                    shard_fused_grad, param_begin, param_end
                )
                partail_status = (
                    group_grad_list[index].dist_attr().partial_status
                )
                partail_status.pop(self._sharding_mesh_axis)
                slice_grad_dist_attr = pir.create_tensor_dist_attribute(
                    slice_grad.process_mesh, [-1], partail_status
                )
                slice_grad.set_type(
                    pir.cvt_to_dist_type(
                        slice_grad.type(), slice_grad_dist_attr
                    )
                )
                new_params_grads.append((slice_param, slice_grad))

        if self._inner_opt._grad_clip is not None:
            self._inner_opt._grad_clip.should_comm_on_shard_dim = True
            self._inner_opt._grad_clip.sharding_group = self.sharding_group
        self._inner_opt.apply_gradients(new_params_grads)
        for (
            shard_param,
            fused_param,
        ) in all_gather_param_info_list:
            allgather_value = paddle._C_ops.all_gather(
                shard_param, self.sharding_group.id, self._sharding_degree
            )
            paddle._C_ops.share_var([fused_param, allgather_value])
        start_index = target_block.ops.index(last_op) + 1
        return target_block.ops[start_index:]

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
            fuse_param_name = "fused@"
            for param in group_param_list:
                startup_param = get_param_from_startup(
                    startup_program, param.name
                )
                startup_param_list.append(startup_param)
                fuse_param_name = fuse_param_name + "-" + param.name
            dtype = startup_param_list[0].dtype
            align_size = (
                fleet.utils.tensor_fusion_helper.alignment[
                    get_current_device_type()
                ]
                // align[dtype]
            )
            align_size = align_size * self._sharding_degree
            _, fused_param = paddle._C_ops.coalesce_tensor_(
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

            group_size = 0
            for param in group_param_list:
                size = np.prod(param._local_shape)
                padded_size = (
                    (size + align_size - 1) // align_size
                ) * align_size
                group_size += padded_size
            fused_type = paddle.pir.create_shaped_type(
                fused_param.type(), [group_size]
            )
            fused_param.set_type(
                pir.cvt_to_dist_type(fused_type, fused_param.dist_attr())
            )
            fused_param.persistable = True
            paddle._pir_ops.set_persistable_value(fused_param, fuse_param_name)
            main_fused_param = main_program.global_block().add_kwarg(
                fuse_param_name, fused_param.type()
            )
            main_fused_param.persistable = True

            shard_size = group_size // self._sharding_degree
            rank = self.sharding_group.ranks.index(dist.get_rank())
            rank_begin = rank * shard_size
            rank_end = rank_begin + shard_size
            shard_fused_param = paddle._C_ops.tensor_slice(
                fused_param, rank_begin, rank_begin + shard_size
            )
            shard_fused_param.persistable = True
            paddle._pir_ops.set_persistable_value(
                shard_fused_param, "shard@" + fuse_param_name
            )
            main_shard_fused_param = main_program.global_block().add_kwarg(
                "shard@" + fuse_param_name, shard_fused_param.type()
            )
            main_shard_fused_param.persistable = True
            total_buffer_size = 0
            slice_param_dict = {}

            for index, param in enumerate(group_param_list):
                size = np.prod(param._local_shape)
                padded_size = (
                    (size + align_size - 1) // align_size
                ) * align_size
                param_begin = max(total_buffer_size - rank_begin, 0)
                total_buffer_size += padded_size
                param_end = min(total_buffer_size - rank_begin, shard_size)
                if param_begin < param_end:
                    init_slice_param = paddle._C_ops.tensor_slice(
                        shard_fused_param, param_begin, param_end
                    )
                    init_slice_param.persistable = True
                    slice_param_name = "slice@" + param.name
                    paddle._pir_ops.set_parameter(
                        init_slice_param, slice_param_name
                    )
                    main_program.set_parameters_from(startup_program)
                    with paddle.static.program_guard(main_program):
                        pir.reset_insertion_point_to_start()
                        slice_param = paddle._pir_ops.parameter(
                            slice_param_name
                        )
                        slice_param.persistable = True
                        slice_param.set_type(init_slice_param.type())
                        slice_param.trainable = param.trainable
                        slice_param.stop_gradient = param.stop_gradient
                        slice_param.optimize_attr = param.optimize_attr
                        slice_param.regularizer = param.regularizer
                        slice_param.do_model_average = param.do_model_average
                        slice_param.need_clip = param.need_clip
                        slice_param.is_distributed = param.is_distributed
                        slice_param.is_parameter = param.is_parameter
                    slice_param_dict[slice_param] = (
                        index,
                        param_begin,
                        param_end,
                    )
        return slice_param_dict, main_shard_fused_param, main_fused_param

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
