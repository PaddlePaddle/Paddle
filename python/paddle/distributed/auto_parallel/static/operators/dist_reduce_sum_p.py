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
    get_dist_tensor_spec,
    is_dim_shard,
    set_dist_op_desc_original_id,
)
from .common import (
    DistributedOperatorImpl,
    DistributedOperatorImplContainer,
    get_default_distributed_operator_impl,
    register_distributed_operator_impl,
    register_distributed_operator_impl_container,
    update_op_dims_mapping,
)


class DistributedReduceSum(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)

    @staticmethod
    def update_dims_mapping(dist_op):
        # step1: prepare inputs need for rule (order args as PHI definition and filter out unnecessary args)

        op_desc = dist_op.serial_op.desc
        assert (
            len(op_desc.input_arg_names()) == 1
        ), f"reduce_sum op [{op_desc.type}] has [{len(op_desc.input_arg_names())}] inputs"
        input_arg_name = op_desc.input_arg_names()[0]
        assert (
            len(op_desc.output_arg_names()) == 1
        ), f"reduce_sum op [{op_desc.type}] has [{len(op_desc.output_arg_names())}] outputs"
        output_arg_name = op_desc.output_arg_names()[0]
        keep_dim = op_desc.attr('keep_dim')
        dims = op_desc.attr('dim')

        # TODO (zhangyichen) replace dist tensor spec by dist tensor in future.
        input_spec = get_dist_tensor_spec(dist_op, input_arg_name)
        output_spec = get_dist_tensor_spec(dist_op, output_arg_name, False)
        # len(dims) == 0 means reduce_all
        if len(dims) == 0:
            dims = list(range(len(input_spec.shape)))

        # step2: infer spmd
        rule = get_phi_spmd_rule("reduce_sum")
        fw_results = rule.infer_forward(input_spec, dims, keep_dim)
        bw_results = rule.infer_backward(
            input_spec, output_spec, dims, keep_dim
        )

        # step3: update dist_attr
        # tensor order following order in PHI definition
        changed = update_op_dims_mapping(
            dist_op, [input_arg_name], [output_arg_name], fw_results, bw_results
        )

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
            dist_op.dist_attr = original_op_dist_attr
            reverted = True
        # if reduce_axis is unsharded, NO extra operator need.
        else:
            default_impl = get_default_distributed_operator_impl()
            op_dist_attr.impl_type = default_impl.type
            op_dist_attr.impl_idx = default_impl.idx
        return reverted


register_distributed_operator_impl_container(DistributedReduceSum("reduce_sum"))


class DistributedReduceSumPrimitive(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)


register_distributed_operator_impl_container(
    DistributedReduceSumPrimitive("reduce_sum_p")
)


# Batch Dimension ReduceSum Primitive
class DistributedReduceSumPrimitiveImpl0(DistributedOperatorImpl):
    def __init__(self, name):
        super().__init__(name)
        self._forward_implemented = True
        self._backward_implemented = True

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr

        return len(op_desc.input_arg_names()) == 1

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        outputs = op_desc.output_arg_names()

        if len(outputs) != 1:
            return False

        output_name = outputs[0]
        output_var = dist_op.serial_op.block._var_recursive(output_name)
        if output_var.shape != ():
            return False

        return True

    def is_auto_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr

        return self.is_input_compatible(dist_op) and self.is_output_compatible(
            dist_op
        )

    def update_dims_mapping(self, dist_op):
        changed = False

        return changed

    @staticmethod
    def forward(ctx, *args, **kwargs):
        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        startup_block = dist_op_context.startup_block
        src_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id

        # check validation of inputs / outputs
        for input_name in src_op.desc.input_names():
            assert input_name in kwargs, f"input [{input_name}] is not given"
            assert len(kwargs[input_name]) == len(
                src_op.desc.input(input_name)
            ), f"number of tensor for input [{input_name}] is not match"
        for output_name in src_op.desc.output_names():
            assert output_name in kwargs, f"input [{output_name}] is not given"
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), f"number of tensor for input [{output_name}] is not match"

        # replicate op in dist program
        dist_op = main_block.append_op(type='nop')
        dist_op_desc = dist_op.desc
        dist_op_desc.copy_from(src_op.desc)
        set_dist_op_desc_original_id(dist_op_desc, src_op.desc, ctx)
        for input_name in src_op.desc.input_names():
            dist_op_desc.set_input(input_name, kwargs[input_name])
        for output_name in src_op.desc.output_names():
            dist_op_desc.set_output(output_name, kwargs[output_name])
        # TODO: should we add a new dist attr for the new op here?

        # batch dimension synchronization
        var_name = src_op.output_arg_names[0]
        sync_group = new_process_group(ctx.data_parallel_group)
        allreduce_op = main_block.append_op(
            type='c_allreduce_sum',
            inputs={'X': [var_name]},
            outputs={'Out': [var_name]},
            attrs={
                'ring_id': sync_group.id,
                'use_calc_stream': True,
                OP_ROLE_KEY: OpRole.Forward,
            },
        )

        # dist attr
        var = main_block._var_recursive(var_name)
        tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(var)
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
        new_op_attr = OperatorDistAttr()
        new_op_attr.process_mesh = op_dist_attr.process_mesh
        new_op_attr.set_output_dims_mapping(
            var.name, tensor_dist_attr.dims_mapping
        )
        new_op_attr.set_input_dims_mapping(
            var.name, tensor_dist_attr.dims_mapping
        )
        ctx.set_op_dist_attr_for_program(allreduce_op, new_op_attr)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        raise RuntimeError("primitive operator does NOT have backward function")


register_distributed_operator_impl(
    "reduce_sum_p",
    DistributedReduceSumPrimitiveImpl0("batch_dimension_reduce_sum_p"),
)
