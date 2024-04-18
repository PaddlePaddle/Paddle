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
# limitations under the License

from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_KEY, OpRole

from ..completion import get_phi_spmd_rule
from ..process_group import new_process_group
from ..utils import _get_comm_group, get_dist_tensor_spec
from .common import (
    DistributedOperatorImpl,
    DistributedOperatorImplContainer,
    register_distributed_operator_impl,
    register_distributed_operator_impl_container,
    set_comm_op_dist_attr_for_program,
    update_op_dims_mapping,
)
from .dist_default import DistributedDefaultImpl0


class DistributedScatter(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)

    @staticmethod
    def update_dims_mapping(dist_op):
        # step1: prepare inputs need for rule (order args as PHI definition and filter out unnecessary args)
        op_desc = dist_op.serial_op.desc
        assert (
            dist_op.serial_op.type == "scatter"
        ), f"{dist_op.serial_op.type} is not supported by dist reshape yet."

        input_names = []
        output_names = []
        input_names.append(op_desc.input('X')[0])
        input_names.append(op_desc.input('Ids')[0])
        input_names.append(op_desc.input('Updates')[0])
        output_names.append(op_desc.output('Out')[0])
        overwrite = op_desc.attr('overwrite')

        input_specs = []
        for name in input_names:
            input_specs.append(get_dist_tensor_spec(dist_op, name))
        output_spec = get_dist_tensor_spec(dist_op, output_names[0], False)

        # step2: infer spmd
        rule = get_phi_spmd_rule("scatter")
        # tensor order following order in PHI definition
        fw_results = rule.infer_forward(
            input_specs[0], input_specs[1], input_specs[2], overwrite
        )
        bw_results = rule.infer_backward(
            input_specs[0],
            input_specs[1],
            input_specs[2],
            output_spec,
            overwrite,
        )

        # step3: update dist_attr
        # tensor order following order in PHI definition
        changed = update_op_dims_mapping(
            dist_op,
            input_names,
            output_names,
            fw_results,
            bw_results,
        )

        return changed

    @staticmethod
    def mapping_to_dist_operator_impl(dist_op, original_op_dist_attr):
        op_dist_attr = dist_op.dist_attr
        op_dist_attr.impl_type = dist_op.serial_op.type
        op_dist_attr.impl_idx = 0

        return False


register_distributed_operator_impl_container(DistributedScatter("scatter"))


class DistributedScatterImpl0(DistributedOperatorImpl):
    def __init__(self, name):
        super().__init__(name)
        self._forward_implemented = True
        self._backward_implemented = True

    def is_input_compatible(self, dist_op):
        return True

    def is_output_compatible(self, dist_op):
        return True

    def is_auto_compatible(self, dist_op):
        return True

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        kwargs: inputname_mapping & outputname_mapping
        """
        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        startup_block = dist_op_context.startup_block
        src_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)

        DistributedDefaultImpl0.forward(ctx, *args, **kwargs)

        dst_op = main_block.ops[-1]
        assert dst_op.type == "scatter"

        # HACK clear the partial state of output
        # only use all_reduce_sum now.
        for out_varname in dst_op.desc.output_arg_names():
            op_out_dist_attr = dst_op.dist_attr.get_output_dist_attr(
                out_varname
            )
            if op_out_dist_attr._is_partial():
                partial_dims = op_out_dist_attr._partial_dims()
                partial_status = op_out_dist_attr._partial_status()
                print("===== handle partial in dist_op =====")
                print(
                    "op:",
                    dst_op.type,
                    "out_var:",
                    out_varname,
                    "partial_dims:",
                    partial_dims,
                    "partial_status:",
                    partial_status,
                )
                for partial_dim, reduce_type in partial_status.items():
                    mesh = op_out_dist_attr.process_mesh
                    group_ranks = _get_comm_group(
                        mesh.process_ids,
                        mesh.shape,
                        partial_dim,
                        rank_id,
                    )
                    sync_group = new_process_group(group_ranks)

                    c_allreduce_sum_op = main_block.append_op(
                        type='c_allreduce_sum',
                        inputs={'X': [out_varname]},
                        outputs={'Out': [out_varname]},
                        attrs={
                            'ring_id': sync_group.id,
                            'use_calc_stream': True,
                            OP_ROLE_KEY: OpRole.Forward,
                        },
                    )
                    op_out_dist_attr._clean_partial_status()
                    print("allreduce_type:", c_allreduce_sum_op.type)

                    set_comm_op_dist_attr_for_program(
                        c_allreduce_sum_op,
                        mesh,
                        op_out_dist_attr,
                        ctx,
                        chunk_id=op_out_dist_attr.chunk_id,
                    )

    @staticmethod
    def backward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)


register_distributed_operator_impl(
    "scatter", DistributedScatterImpl0("scatter")
)
