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
# limitations under the License

import copy

from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_KEY, OpRole

from ..completion import get_phi_spmd_rule
from ..dist_attribute import OperatorDistAttr
from ..process_group import new_process_group
from ..utils import (
    _get_comm_group,
    get_dist_tensor_spec,
    is_dim_shard,
)
from .common import (
    DistributedOperatorImpl,
    DistributedOperatorImplContainer,
    copy_op_without_infer_shape,
    register_distributed_operator_impl,
    register_distributed_operator_impl_container,
    set_comm_op_dist_attr_for_program,
    update_op_dims_mapping,
)
from .dist_default import DistributedDefaultImpl0


class DistributedReduce(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)

    @staticmethod
    def update_dims_mapping(dist_op, reduce_type=0):
        # step1: prepare inputs need for rule (order args as PHI definition and filter out unnecessary args)

        op_desc = dist_op.serial_op.desc
        assert (
            len(op_desc.input_arg_names()) == 1
        ), "reduce_sum op [{}] has [{}] inputs".format(
            op_desc.type, len(op_desc.input_arg_names())
        )
        input_arg_name = op_desc.input_arg_names()[0]
        assert (
            len(op_desc.output_arg_names()) == 1
        ), "reduce_sum op [{}] has [{}] outputs".format(
            op_desc.type, len(op_desc.output_arg_names())
        )
        output_arg_name = op_desc.output_arg_names()[0]
        keep_dim = op_desc.attr('keep_dim')
        dims = op_desc.attr('dim')

        # TODO (zhangyichen) replace dist tensor spece by dist tensor in future.
        input_spec = get_dist_tensor_spec(dist_op, input_arg_name)
        output_spec = get_dist_tensor_spec(dist_op, output_arg_name, False)
        # len(dims) == 0 means reduce_all
        if len(dims) == 0:
            dims = list(range(len(input_spec.shape)))

        # step2: infer spmd
        rule = get_phi_spmd_rule("reduce_base")
        fw_results = rule.infer_forward(input_spec, dims, keep_dim, reduce_type)
        bw_results = rule.infer_backward(
            input_spec, output_spec, dims, keep_dim
        )

        # if (op_desc.type() == "reduce_mean"):
        #     print("op dist attr before:", dist_op.dist_attr)
        # step3: update dist_attr
        # tensor order following order in PHI definition
        changed = update_op_dims_mapping(
            dist_op, [input_arg_name], [output_arg_name], fw_results, bw_results
        )
        if op_desc.type() == "reduce_mean":
            print(fw_results[0][0])
            print(fw_results[1][0])
            print(dist_op.dist_attr)
            # breakpoint()

        return changed

    # NOTE this function will be remove once we use local reshard to replace distopimpls
    @staticmethod
    def mapping_to_dist_operator_impl(dist_op, original_op_dist_attr):
        op_dist_attr = dist_op.dist_attr
        op_desc = dist_op.serial_op.desc
        input_name = op_desc.input_arg_names()[0]
        input_dims_mapping = copy.deepcopy(
            op_dist_attr.get_input_dims_mapping(input_name)
        )
        axes = op_desc.attr('dim')

        op_dist_attr = dist_op.dist_attr
        reverted = False

        def is_partial_reduce(axes, dims_mapping):
            # FIXME(ljz) Hack for performance:
            # if the reduce result is a scalar, it is the loss reduce in GPT case,
            # and if any axis of reduce input is sharded, the result loss would be partial.
            # BUT we keep the loss as partial instead of allreduce it for performance, since it would effect the backward.
            # we should use an optimization pass for the Hack in future.
            if len(axes) != 0 and (len(axes) < len(dims_mapping)):
                for axis in axes:
                    if is_dim_shard(dims_mapping[axis]):
                        return True  # reverted
            return False

        # if reduce_axis is sharded, the output is partial and need to be allreduce
        if is_partial_reduce(axes, input_dims_mapping):
            # TODO (ljz) support reduce where the reduce_axis is sharded
            op_dist_attr = dist_op.dist_attr
            op_dist_attr.impl_type = dist_op.serial_op.type
            op_dist_attr.impl_idx = 0
            # dist_op.dist_attr = original_op_dist_attr
        # if reduce_axis is unsharded, NO extra operator need.
        else:
            op_dist_attr = dist_op.dist_attr
            op_dist_attr.impl_type = dist_op.serial_op.type
            op_dist_attr.impl_idx = 0
            # default_impl = get_default_distributed_operator_impl()
            # op_dist_attr.impl_type = default_impl.type
            # op_dist_attr.impl_idx = default_impl.idx
        return reverted


class DistributedReduceSum(DistributedReduce):
    def __init__(self, op_type):
        super().__init__(op_type)

    @staticmethod
    def update_dims_mapping(dist_op):
        return DistributedReduce.update_dims_mapping(dist_op, 0)


class DistributedReduceMean(DistributedReduce):
    def __init__(self, op_type):
        super().__init__(op_type)

    @staticmethod
    def update_dims_mapping(dist_op):
        return DistributedReduce.update_dims_mapping(dist_op, 4)


# register_distributed_operator_impl_container(DistributedReduceSum("reduce_sum"))
register_distributed_operator_impl_container(
    DistributedReduceMean("reduce_mean")
)


class DistributedReduceMeanImpl0(DistributedOperatorImpl):
    def __init__(self, name):
        super().__init__(name)
        self._forward_implemented = True
        self._backward_implemented = True
        scale_degree = 1.0

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
        assert dst_op.type == "reduce_mean"

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

                    DistributedReduceMeanImpl0.scale_degree = 1.0 / len(
                        mesh.process_ids
                    )
                    set_comm_op_dist_attr_for_program(
                        c_allreduce_sum_op,
                        mesh,
                        op_out_dist_attr,
                        ctx,
                        chunk_id=op_out_dist_attr.chunk_id,
                    )
                    scale_op = main_block.append_op(
                        type='scale',
                        inputs={'X': out_varname},
                        outputs={'Out': out_varname},
                        attrs={
                            'scale': 1.0 / len(mesh.process_ids),
                            OP_ROLE_KEY: OpRole.Forward,
                        },
                    )
                    dims_mapping = op_dist_attr.get_output_dims_mapping(
                        out_varname
                    )
                    scale_op_attr = OperatorDistAttr()
                    scale_op_attr.process_mesh = mesh
                    scale_op_attr.chunk_id = op_dist_attr.chunk_id
                    scale_op_attr.set_output_dims_mapping(
                        out_varname, dims_mapping
                    )
                    scale_op_attr.set_input_dims_mapping(
                        out_varname, dims_mapping
                    )
                    ctx.set_op_dist_attr_for_program(scale_op, scale_op_attr)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        startup_block = dist_op_context.startup_block
        src_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)

        assert 'X' in kwargs, "input [{}] is not given".format('X')
        assert 'Out@GRAD' in kwargs, "input [{}] is not given".format(
            'Out@GRAD'
        )
        assert 'X@GRAD' in kwargs, "output [{}] is not given".format('X@GRAD')

        copy_op_without_infer_shape(src_op, main_block, ctx, kwargs)

        out_grad_dist_attr = op_dist_attr.get_input_dist_attr(
            kwargs['Out@GRAD'][0]
        )
        x_grad_dist_attr = op_dist_attr.get_output_dist_attr(
            kwargs['X@GRAD'][0]
        )
        x_dist_attr = op_dist_attr.get_input_dist_attr(kwargs['X'][0])
        out_grad_dims_mapping = out_grad_dist_attr.dims_mapping
        x_grad_dims_mapping = x_grad_dist_attr.dims_mapping
        x_dims_mapping = x_dist_attr.dims_mapping

        process_mesh = op_dist_attr.process_mesh
        degree = 1.0
        for i in range(len(x_dims_mapping)):
            if x_dims_mapping[i] != -1:
                degree *= process_mesh.shape[x_dims_mapping[i]]

        if degree > 1.0:
            x_grad_varname = kwargs['X@GRAD'][0]
            scale_op = main_block.append_op(
                type='scale',
                inputs={'X': x_grad_varname},
                outputs={'Out': x_grad_varname},
                attrs={
                    'scale': 1.0 / degree,
                    OP_ROLE_KEY: OpRole.Backward,
                },
            )
            x_grad_dims_mapping = op_dist_attr.get_output_dims_mapping(
                x_grad_varname
            )
            scale_op_attr = OperatorDistAttr()
            scale_op_attr.process_mesh = op_dist_attr.process_mesh
            scale_op_attr.chunk_id = op_dist_attr.chunk_id
            scale_op_attr.set_output_dims_mapping(
                x_grad_varname, x_grad_dims_mapping
            )
            scale_op_attr.set_input_dims_mapping(
                x_grad_varname, x_grad_dims_mapping
            )
            ctx.set_op_dist_attr_for_program(scale_op, scale_op_attr)


register_distributed_operator_impl(
    "reduce_mean", DistributedReduceMeanImpl0("reduce_mean")
)
