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
from paddle.base.executor import global_scope
from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_KEY, OpRole

from ..auto_parallel.static.dist_attribute import TensorDistAttr
from ..auto_parallel.static.process_group import new_process_group
from ..auto_parallel.static.utils import (
    get_logger,
    naive_set_dist_op_attr_for_program_by_mesh_and_mapping,
)
from .pass_base import PassBase, register_pass

logger = get_logger(logging.INFO)


@register_pass("auto_parallel_sync_shared_params")
class SyncSharedParamsPass(PassBase):
    def __init__(self):
        super().__init__()
        self.params_maybe_shared = []
        self.set_attr("dist_context", None)
        self.set_attr("global_rank", None)
        self.src_ranks = []
        self.dst_ranks = []
        self.rankp2p = {}
        self.set_attr("dist_params_grads", None)

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def pre_analysis(self, main_program, startup_program, dist_params_grads):
        self.dist_params_grads = dist_params_grads
        self.dist_context = self.get_attr("dist_context")
        main_block = main_program.global_block()
        for idx, op in enumerate(main_block.ops):
            if op.type == 'assign':
                var_name = op.input("X")[0]
                if not main_block.has_var(var_name):
                    logger.warning(
                        f"The assign op input {var_name} is not in block, please check the program."
                    )
                    continue
                # 1. check param
                var = main_block.var(var_name)
                if not var.is_parameter:
                    continue
                if not self._param_from_embedding(main_block, var_name):
                    continue
                # 2. check not the same mesh
                var_dist_attr = (
                    self.dist_context.get_tensor_dist_attr_for_program(var)
                )
                dist_op = self.dist_context.get_dist_op_for_program(op)
                op_input_dist_attr = dist_op.dist_attr.get_input_dist_attr(
                    var_name
                )
                src_mesh = var_dist_attr.process_mesh
                dst_mesh = op_input_dist_attr.process_mesh
                if src_mesh == dst_mesh:
                    continue
                self.params_maybe_shared.append(
                    {
                        'src_mesh': src_mesh,
                        'dst_mesh': dst_mesh,
                        'src_dist_attr': var_dist_attr,
                        'dst_dist_attr': op_input_dist_attr,
                        'param_name': var_name,
                        'param': var,
                    }
                )
                self.src_ranks.extend(var_dist_attr.process_mesh.process_ids)
                self.dst_ranks.extend(
                    op_input_dist_attr.process_mesh.process_ids
                )

        if len(self.params_maybe_shared) == 0:
            return 1
        assert (
            len(self.params_maybe_shared) == 1
        ), "only support single shared params now"

        if len(self.src_ranks) != len(self.dst_ranks):
            return 2
        for src_rank, dst_rank in zip(self.src_ranks, self.dst_ranks):
            self.rankp2p[dst_rank] = src_rank
            self.rankp2p[src_rank] = dst_rank

        if self.get_attr("global_rank") in self.dst_ranks:
            # record opt op
            for idx, op in enumerate(main_block.ops):
                if op._is_optimize_op():
                    var_name = op.output("ParamOut")[0]
                    if (
                        not var_name
                        == self.params_maybe_shared[0]['param_name']
                    ):
                        continue

                    opt_info = {
                        "opt_op": op,
                    }
                    self.params_maybe_shared[0].update(opt_info)
        logger.info(f"params_maybe_shared: {self.params_maybe_shared}")
        return self.params_maybe_shared

    def _apply_single_impl(self, main_program, startup_program, context):
        if len(self.params_maybe_shared) == 0:
            return context

        if self.get_attr("global_rank") in self.src_ranks:
            self._apply_single_impl_stage_src(
                main_program, startup_program, context
            )
        else:
            self._apply_single_impl_stage_dst(
                main_program, startup_program, context
            )
        self._create_var_in_scope()
        return context

    def _create_var_in_scope(self):
        for param in self.dist_context.concrete_program.parameters:
            if param.name == self.params_maybe_shared[0]["param_name"]:
                serial_main_program = (
                    self.dist_context.concrete_program.main_program
                )
                var = serial_main_program.global_block().vars[param.name]
                var_dist_attr = (
                    self.dist_context.get_tensor_dist_attr_for_program(var)
                )
                new_var_dist_attr = TensorDistAttr()
                new_var_dist_attr.process_mesh = self.params_maybe_shared[0][
                    "dst_mesh"
                ]
                new_var_dist_attr.dims_mapping = var_dist_attr.dims_mapping
                with paddle.no_grad():
                    tmp = paddle.base.core.reshard(param, new_var_dist_attr)
                paddle.device.synchronize()
                if tmp._is_initialized():
                    dense_tensor = global_scope().var(param.name).get_tensor()
                    dense_tensor._share_data_with(tmp.get_tensor().get_tensor())

    def _apply_single_impl_stage_src(
        self, main_program, startup_program, context
    ):
        # 1. check
        main_block = main_program.global_block()
        recv_info = None
        send_info = None
        sum_info = None
        param_info = self.params_maybe_shared[0]
        param_name = param_info["param_name"]
        for idx, op in enumerate(main_block.ops):
            if op.type == 'send_v2':
                var_name = op.input("X")[0]
                if not var_name == param_name:
                    continue
                assert send_info is None, "only one send_v2 op is allowed"
                send_info = {
                    "var": main_block.var(var_name),
                    "var_name": var_name,
                    "op": op,
                    "op_idx": idx,
                    "ring_id": op.attr("ring_id"),
                    "dist_op": self.dist_context.get_dist_op_for_program(op),
                    "dist_attr": self.dist_context.get_dist_op_for_program(
                        op
                    ).dist_attr,
                }
            elif op.type == 'recv_v2':
                var_name = op.output("Out")[0]
                if not var_name.startswith(param_name + "@GRAD"):
                    continue
                assert recv_info is None, "only one recv_v2 op is allowed"
                recv_info = {
                    "var": main_block.var(var_name),
                    "var_name": var_name,
                    "op": op,
                    "op_idx": idx,
                    "ring_id": op.attr("ring_id"),
                    "dist_op": self.dist_context.get_dist_op_for_program(op),
                    "dist_attr": self.dist_context.get_dist_op_for_program(
                        op
                    ).dist_attr,
                }
            elif op.type == 'sum':
                var_name = op.output("Out")[0]
                if not var_name == param_name + "@GRAD":
                    continue
                assert sum_info is None, "only one sum_info op is allowed"
                sum_info = {
                    "var": main_block.var(var_name),
                    "var_name": var_name,
                    "op": op,
                    "op_idx": idx,
                    "dist_op": self.dist_context.get_dist_op_for_program(op),
                    "dist_attr": self.dist_context.get_dist_op_for_program(
                        op
                    ).dist_attr,
                }

        if recv_info is None or send_info is None or sum_info is None:
            return

        # 2. add broadcast
        startup_block = startup_program.global_block()
        param = startup_block.var(param_name)
        cur_rank = self.get_attr("global_rank")
        ranks = [cur_rank, self.rankp2p[cur_rank]]
        ranks.sort()
        sync_group = new_process_group(ranks)
        new_op = startup_block.append_op(
            type='c_broadcast',
            inputs={'X': [param]},
            outputs={'Out': [param]},
            attrs={
                'ring_id': sync_group.id,
                'root': 0,
                'use_calc_stream': True,
                OP_ROLE_KEY: OpRole.Forward,
            },
        )
        # 3. insert c_allreduce_sum, fill_constant, remove recv_v2, send_v2,
        # 3.1 insert allreduce_sum
        param_grad = sum_info["var"]
        allreduce_op = main_block._insert_op_without_sync(
            index=sum_info["op_idx"] + 1,
            type="c_allreduce_sum",
            inputs={"X": param_grad},
            outputs={"Out": param_grad},
            attrs={
                'ring_id': sync_group.id,
                'use_calc_stream': True,
                OP_ROLE_KEY: OpRole.Backward,
            },
        )
        allreduce_op_dist_attr = sum_info["dist_attr"]
        allreduce_op_dist_attr.set_input_dist_attr(
            param_grad.name,
            sum_info["dist_attr"].get_output_dist_attr(sum_info["var_name"]),
        )
        allreduce_op_dist_attr.set_output_dist_attr(
            param_grad.name,
            sum_info["dist_attr"].get_output_dist_attr(sum_info["var_name"]),
        )
        self.dist_context.set_op_dist_attr_for_program(
            allreduce_op, allreduce_op_dist_attr
        )

        # 3.2 insert fill_constant.
        out_var = recv_info["var"]
        fill_constant_op = main_block._insert_op_without_sync(
            index=recv_info["op_idx"] + 1,
            type='fill_constant',
            outputs={'Out': [out_var]},
            attrs={
                'dtype': out_var.dtype,
                'shape': out_var.shape,
                'value': 0.0,
                'force_cpu': False,
                OP_ROLE_KEY: OpRole.Backward,
            },
        )
        fill_constant_op_dist_attr = recv_info["dist_attr"]
        fill_constant_op_dist_attr.set_output_dist_attr(
            out_var.name,
            recv_info["dist_attr"].get_output_dist_attr(recv_info["var_name"]),
        )
        self.dist_context.set_op_dist_attr_for_program(
            fill_constant_op, fill_constant_op_dist_attr
        )

        # 3.3 remove recv_v2
        main_block._remove_op(recv_info["op_idx"])

        # 3.4 remove send_v2
        main_block._remove_op(send_info["op_idx"])

        main_block._sync_with_cpp()
        startup_program.global_block()._sync_with_cpp()

    def _apply_single_impl_stage_dst(
        self, main_program, startup_program, context
    ):
        # 1. check
        main_block = main_program.global_block()
        recv_info = None
        send_info = None
        sum_info = None
        param_info = self.params_maybe_shared[0]
        param_name = param_info["param_name"]
        for idx, op in enumerate(main_block.ops):
            if op.type == 'recv_v2':
                var_name = op.output("Out")[0]
                if not var_name.startswith(param_name):
                    continue

                assert recv_info is None, "only one recv_v2 op is allowed"
                recv_info = {
                    "var": main_block.var(var_name),
                    "var_name": var_name,
                    "op": op,
                    "op_idx": idx,
                    "ring_id": op.attr("ring_id"),
                    "dist_op": self.dist_context.get_dist_op_for_program(op),
                    "dist_attr": self.dist_context.get_dist_op_for_program(
                        op
                    ).dist_attr,
                }
            elif op.type == 'send_v2':
                var_name = op.input("X")[0]
                if not var_name.startswith(param_name + "@GRAD"):
                    continue
                assert send_info is None, "only one send_v2 op is allowed"
                send_info = {
                    "var": main_block.var(var_name),
                    "var_name": var_name,
                    "op": op,
                    "op_idx": idx,
                    "ring_id": op.attr("ring_id"),
                    "dist_op": self.dist_context.get_dist_op_for_program(op),
                    "dist_attr": self.dist_context.get_dist_op_for_program(
                        op
                    ).dist_attr,
                }
        if recv_info is None or send_info is None:
            return

        # 2. add broadcast
        startup_block = startup_program.global_block()
        param = startup_block.var(param_name)
        cur_rank = self.get_attr("global_rank")
        ranks = [cur_rank, self.rankp2p[cur_rank]]
        ranks.sort()
        sync_group = new_process_group(ranks)
        new_op = startup_block.append_op(
            type='c_broadcast',
            inputs={'X': [param]},
            outputs={'Out': [param]},
            attrs={
                'ring_id': sync_group.id,
                'root': 0,
                'use_calc_stream': True,
                OP_ROLE_KEY: OpRole.Forward,
            },
        )
        # 3. insert assign*2, c_allreduce_sum, remove send_v2, recv_v2,
        # 3.1 insert assign.
        new_param_grad = main_block.create_var(
            name=f"{param_name}@GRAD",
            dtype=send_info["var"].dtype,
            persistable=False,
        )
        self.dist_context.set_tensor_dist_attr_for_program(
            new_param_grad,
            self.dist_context.get_tensor_dist_attr_for_program(
                send_info["var"]
            ),
        )
        assign_op = main_block._insert_op_without_sync(
            index=send_info["op_idx"] + 1,
            type='assign',
            inputs={'X': [send_info["var_name"]]},
            outputs={'Out': [new_param_grad]},
            attrs={
                OP_ROLE_KEY: OpRole.Backward,
            },
        )
        assign_op_dist_attr = send_info["dist_attr"]
        assign_op_dist_attr.set_input_dist_attr(
            send_info["var_name"],
            send_info["dist_attr"].get_input_dist_attr(send_info["var_name"]),
        )
        assign_op_dist_attr.set_output_dist_attr(
            new_param_grad.name,
            send_info["dist_attr"].get_input_dist_attr(send_info["var_name"]),
        )
        self.dist_context.set_op_dist_attr_for_program(
            assign_op, assign_op_dist_attr
        )

        # add new (param, grad) to dist_params_grads
        self.dist_params_grads.append((param, new_param_grad))

        # 3.2 insert allreduce_sum
        allreduce_op = main_block._insert_op_without_sync(
            index=send_info["op_idx"] + 2,
            type="c_allreduce_sum",
            inputs={"X": new_param_grad},
            outputs={"Out": new_param_grad},
            attrs={
                'ring_id': sync_group.id,
                'use_calc_stream': True,
                OP_ROLE_KEY: OpRole.Backward,
            },
        )
        allreduce_op_dist_attr = send_info["dist_attr"]
        allreduce_op_dist_attr.set_input_dist_attr(
            new_param_grad.name,
            send_info["dist_attr"].get_input_dist_attr(send_info["var_name"]),
        )
        allreduce_op_dist_attr.set_output_dist_attr(
            new_param_grad.name,
            send_info["dist_attr"].get_input_dist_attr(send_info["var_name"]),
        )
        self.dist_context.set_op_dist_attr_for_program(
            allreduce_op, allreduce_op_dist_attr
        )

        # 3.3 remove send_v2
        main_block._remove_op(send_info["op_idx"])

        # 3.4 insert assign
        assign_op = main_block._insert_op_without_sync(
            index=recv_info["op_idx"] + 1,
            type='assign',
            inputs={'X': [param]},
            outputs={'Out': [recv_info["var_name"]]},
        )
        assign_op_dist_attr = recv_info["dist_attr"]
        assign_op_dist_attr.set_input_dist_attr(
            param.name,
            recv_info["dist_attr"].get_output_dist_attr(recv_info["var_name"]),
        )
        assign_op_dist_attr.set_output_dist_attr(
            recv_info["var_name"],
            recv_info["dist_attr"].get_output_dist_attr(recv_info["var_name"]),
        )
        self.dist_context.set_op_dist_attr_for_program(
            assign_op, assign_op_dist_attr
        )

        # 3.5 remove recv_v2
        main_block._remove_op(recv_info["op_idx"])

        # 4. insert opt
        first_opt_idx = None
        for idx, op in enumerate(main_block.ops):
            if op._is_optimize_op():
                first_opt_idx = idx
                break
        ori_opt_op = param_info["opt_op"]
        inputs = {}
        outputs = {}
        for name in ori_opt_op.input_names:
            inputs[name] = ori_opt_op.input(name)
        for name in ori_opt_op.output_names:
            outputs[name] = ori_opt_op.output(name)
        new_opt_op = main_block._insert_op_without_sync(
            index=first_opt_idx,
            type=ori_opt_op.type,
            inputs=inputs,
            outputs=outputs,
            attrs=ori_opt_op.all_attrs(),
        )
        ref_dist_attr = self.dist_context.get_tensor_dist_attr_for_program(
            send_info["var"]
        )
        ref_process_mesh = ref_dist_attr.process_mesh
        ref_dims_mapping = ref_dist_attr.dims_mapping
        naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
            new_opt_op,
            ref_process_mesh,
            ref_dims_mapping,
            self.dist_context,
            chunk_id=ref_dist_attr.chunk_id,
        )

        main_block._sync_with_cpp()
        startup_program.global_block()._sync_with_cpp()

    def _param_from_embedding(self, block, param_name):
        for op in block.ops:
            if op.type == "lookup_table_v2":
                if op.input("W")[0] == param_name:
                    return True
        return False
