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


import logging

import paddle
import paddle.distributed as dist
from paddle.base.core import TensorDistAttr
from paddle.base.executor import global_scope
from paddle.base.framework import auto_complete_op_role
from paddle.distributed.auto_parallel.static.process_group import (
    new_process_group,
)
from paddle.distributed.auto_parallel.static.utils import (
    get_pp_stage_by_process_mesh,
)
from paddle.distributed.fleet.meta_optimizers.common import OpRole
from paddle.static.pir_io import get_pir_parameters

from ..auto_parallel.static.utils import (
    get_logger,
)
from .pass_base import PassBase, register_pass

logger = get_logger(logging.INFO)


@register_pass("auto_parallel_sync_shared_params")
class AutoParallelSyncSharedParamsPass(PassBase):
    def __init__(self):
        super().__init__()
        self.params_maybe_shared = []
        self.src_ranks = []
        self.dst_ranks = []
        self.comm_group = {}

    def _check_self(self):
        pipeline_strategy = self.get_attr('pipeline_strategy')
        if (not pipeline_strategy.enable) or pipeline_strategy.pp_degree <= 1:
            return False
        return True

    def _check_conflict(self, other_pass):
        return True

    def _find_fist_opt_user(self, main_program):
        for op in main_program.global_block().ops:
            if op.op_role == 2:
                return op

    def _get_comm_group(self, ranks=[]):
        ranks = sorted(ranks)
        if tuple(ranks) in self.comm_group:
            return self.comm_group[tuple(ranks)]
        # The communication group of this `all_reduce` op satisfies len (ranks)==2.
        # When `force_new_group=False` is set, the `send&recv` group will be returned,
        # At this point, `all_reduce` and `send&recv` share the same group, and
        # the process will hang up.
        group = new_process_group(ranks, force_new_group=True)
        self.comm_group[tuple(ranks)] = group.id
        return group.id

    def sync_shared_parameters(self, main_program, startup_program):
        if not self._check_self():
            logger.info(
                "AutoParallelSyncSharedParamsPass need support pipeline parallel, skip pass."
            )
            return []
        new_shared_params = []
        params, _ = get_pir_parameters(main_program)
        for param in params:
            users = param.all_used_ops()
            for user_op in users:
                if user_op.name() == "dist_op.reshard":
                    reshard_op = user_op
                    dist_attr = reshard_op.dist_attr
                    src_dist_attr = dist_attr.operand(0).as_tensor_dist_attr()
                    dst_dist_attr = dist_attr.result(0).as_tensor_dist_attr()
                    src_mesh = src_dist_attr.process_mesh
                    dst_mesh = dst_dist_attr.process_mesh

                    # Shared parameter needs reshard on diff stage.
                    pipeline_strategy = self.get_attr('pipeline_strategy')
                    pp_degree = pipeline_strategy.pp_degree
                    src_stage = get_pp_stage_by_process_mesh(
                        src_mesh, pp_degree
                    )
                    dst_stage = get_pp_stage_by_process_mesh(
                        dst_mesh, pp_degree
                    )
                    if (
                        src_stage is None
                        or dst_stage is None
                        or src_stage == dst_stage
                    ):
                        continue

                    # Get shared parameter name
                    param_name = param.get_defining_op().str_attr(
                        'parameter_name'
                    )

                    # Add shared parameter builtin.parameter with "shared_" prefix.
                    with auto_complete_op_role(main_program, OpRole.Forward):
                        with paddle.static.program_guard(
                            main_program, startup_program
                        ):
                            shared_param = paddle.pir.core.create_parameter(
                                dtype=param.dtype,
                                shape=param.shape,
                                name="shared_" + param_name,
                                process_mesh=dst_mesh,
                                placements=src_dist_attr.placements,
                                initializer=paddle.nn.initializer.Constant(
                                    value=0
                                ),
                            )
                    main_program.set_parameters_from(startup_program)

                    # Record new shared parameter.
                    new_shared_params.append("shared_" + param_name)

                    # Set value for new shared parameter.
                    concrete_program = self.get_attr("concrete_program")
                    dy_params = concrete_program.parameters[0]
                    dy_param = None
                    for tmp_param in dy_params:
                        if tmp_param.name == param_name:
                            dy_param = tmp_param
                            break
                    assert (
                        dy_param is not None
                    ), f"The parameter {param_name} was not found in the concrete_degram"

                    new_dist_attr = TensorDistAttr()
                    new_dist_attr.process_mesh = dst_mesh
                    new_dist_attr.dims_mapping = src_dist_attr.dims_mapping
                    with paddle.no_grad():
                        dy_shared_param = paddle.base.core.reshard(
                            dy_param, new_dist_attr
                        )
                    paddle.device.synchronize()
                    if dy_shared_param._is_initialized():
                        pir_shared_param = (
                            global_scope()
                            .var("shared_" + param_name)
                            .get_tensor()
                        )
                        pir_shared_param._share_data_with(
                            dy_shared_param.get_tensor().get_tensor()
                        )

                    # record in params_maybe_shared
                    self.params_maybe_shared.append(
                        {
                            'src_mesh': src_mesh,
                            'dst_mesh': dst_mesh,
                            'src_dist_attr': src_dist_attr,
                            'dst_dist_attr': dst_dist_attr,
                            'param_name': param_name,
                        }
                    )

                    # New shared parameter must has same dist_attr with shared parameter
                    new_src_dist_attr = (
                        paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                            dst_dist_attr.process_mesh,
                            src_dist_attr.dims_mapping,
                            src_dist_attr.partial_status,
                        )
                    )
                    if new_src_dist_attr == dst_dist_attr:
                        # Remove useless reshared op.
                        reshard_op.result(0).replace_all_uses_with(shared_param)
                        reshard_op.erase()

                    else:
                        # Update reshard op dist_attr.
                        reshard_op.dist_attr = (
                            paddle.base.libpaddle.pir.create_op_dist_attribute(
                                dst_mesh,
                                [new_src_dist_attr],
                                [dst_dist_attr],
                                -1,
                            )
                        )
                        reshard_op.operand(0).set_source(shared_param)

                    self.src_ranks.extend(src_mesh.process_ids)
                    self.dst_ranks.extend(dst_mesh.process_ids)

        if len(self.params_maybe_shared) == 0:
            logger.info("No parameter need to share, skip pass.")
            return []

        # Must initialize the redundant communication group for the allreduce op here.
        # Otherwise, it will hang during gradient synchronization.
        for idx in range(len(self.src_ranks)):
            rank_1 = self.src_ranks[idx]
            rank_2 = self.dst_ranks[idx]
            new_process_group(sorted([rank_1, rank_2]))
            self._get_comm_group([rank_1, rank_2])

        return new_shared_params

    def sync_shared_parameter_gradient(
        self, main_program, startup_program, params_grads
    ):
        if not self._check_self():
            logger.info(
                "AutoParallelSyncSharedParamsPass need support pipeline parallel, skip pass."
            )
            return params_grads

        if len(self.params_maybe_shared) == 0:
            logger.info("No parameter need to share, skip pass.")
            return params_grads

        # Only support one shared parameter.
        # TODO: support more shared parameters
        assert (
            len(self.params_maybe_shared) == 1
        ), "Currently, only one shared parameter is supported, and it cannot support more at the moment."

        cur_rank = paddle.distributed.get_rank()

        if cur_rank not in self.src_ranks and cur_rank not in self.dst_ranks:
            return params_grads

        pre_name = ""
        if cur_rank in self.dst_ranks:
            pre_name = "shared_"

        for param_mess in self.params_maybe_shared:
            param_name = pre_name + param_mess['param_name']
            src_mesh_ids = param_mess['src_mesh'].process_ids
            dst_mesh_ids = param_mess['dst_mesh'].process_ids

            # Get (param, grad) value
            param_value = main_program.get_parameter_value_by_name(param_name)

            grad_idx = None
            for p_idx, (p_param, _) in enumerate(params_grads):
                if p_param.is_same(param_value):
                    grad_idx = p_idx
                    break
            assert (
                grad_idx is not None
            ), f"Parameter {param_name} not found in params_grades, unable to find corresponding gradient value."
            grad_value = params_grads[p_idx][1]

            # Create allreduce op comm group.
            cur_rank = paddle.distributed.get_rank()
            if cur_rank in self.src_ranks:
                idx = src_mesh_ids.index(cur_rank)
                peer_rank = dst_mesh_ids[idx]
            if cur_rank in self.dst_ranks:
                idx = dst_mesh_ids.index(cur_rank)
                peer_rank = src_mesh_ids[idx]
            ar_group_id = self._get_comm_group([cur_rank, peer_rank])

            # Insert allreduce op in the end of backward.
            insert_pos = self._find_fist_opt_user(main_program)
            paddle.pir.set_insertion_point(insert_pos)

            # Build allreduce op to sync gradient.
            with auto_complete_op_role(main_program, OpRole.Backward):
                allreduce_val = paddle._C_ops.all_reduce(
                    grad_value,
                    ar_group_id,
                    dist.ReduceOp.SUM,
                )
                allreduce_val.update_dist_attr(grad_value.dist_attr())
            allreduce_op = allreduce_val.get_defining_op()

            # Update all_used_ops
            for user in grad_value.all_used_ops():
                if user.name() == "pd_op.all_reduce":
                    continue
                for idx, operand in enumerate(user.operands()):
                    if user.operand_source(idx).is_same(grad_value):
                        user.operand(idx).set_source(allreduce_val)

            # Update (param, grad) value
            params_grads[p_idx] = (param_value, allreduce_val)

        return params_grads

    def _apply_single_impl(self, main_program, startup_program, context):
        return
