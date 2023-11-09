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

import copy
import logging

from paddle.base.log_helper import get_logger
from paddle.common_ops_import import check_variable_and_dtype
from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_KEY, OpRole

from ..completion import get_phi_spmd_rule
from ..dist_attribute import OperatorDistAttr
from ..process_group import new_process_group
from ..utils import (
    _get_comm_group,
    _get_corresponding_rank,
    get_dist_tensor_spec,
    is_dim_shard,
    set_dist_op_desc_original_id,
)
from .common import (
    DistributedOperatorImpl,
    DistributedOperatorImplContainer,
    ParallelMode,
    copy_op_without_infer_shape,
    get_data_parallel_group,
    infer_shape,
    naive_copy_op_dist_attr_for_program,
    register_distributed_operator_impl,
    register_distributed_operator_impl_container,
    update_op_dims_mapping,
)

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


class DistributedCrossEntropy(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)

    @staticmethod
    def update_dims_mapping(dist_op):
        # step1: prepare inputs need for rule (order args as PHI definition and filter out unnecessary args)
        op_desc = dist_op.serial_op.desc

        logits_name = op_desc.input('Logits')[0]
        label_name = op_desc.input('Label')[0]
        loss_name = op_desc.output('Loss')[0]
        softmax_name = op_desc.output('Softmax')[0]

        soft_label = op_desc.attr('soft_label')
        ignore_index = op_desc.attr('ignore_index')
        numeric_stable_mode = op_desc.attr('numeric_stable_mode')
        axis = op_desc.attr('axis')

        logits_spec = get_dist_tensor_spec(dist_op, logits_name)
        label_spec = get_dist_tensor_spec(dist_op, label_name)
        loss_spec = get_dist_tensor_spec(dist_op, loss_name, False)
        softmax_spec = get_dist_tensor_spec(dist_op, softmax_name, False)

        # step2: infer spmd
        rule = get_phi_spmd_rule("softmax_with_cross_entropy")
        # tensor order following order in PHI defition
        fw_results = rule.infer_forward(
            logits_spec,
            label_spec,
            soft_label,
            True,
            numeric_stable_mode,
            ignore_index,
            axis,
        )
        bw_results = rule.infer_backward(
            logits_spec,
            label_spec,
            softmax_spec,
            loss_spec,
            soft_label,
            True,
            numeric_stable_mode,
            ignore_index,
            axis,
        )

        # step3: update dist_attr
        # tensor order following order in PHI defition
        changed = update_op_dims_mapping(
            dist_op,
            [logits_name, label_name],
            [softmax_name, loss_name],
            fw_results,
            bw_results,
        )

        return changed

    @staticmethod
    def mapping_to_dist_operator_impl(dist_op, original_op_dist_attr):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        op_dist_attr.impl_type = op_desc.type()

        logits_name = op_desc.input('Logits')[0]

        soft_label = op_desc.attr('soft_label')
        axis = op_desc.attr('axis')

        logits_dims_mapping = copy.deepcopy(
            op_dist_attr.get_input_dims_mapping(logits_name)
        )
        logits_ndim = len(logits_dims_mapping)
        axis = axis + logits_ndim if axis < 0 else axis

        if is_dim_shard(logits_dims_mapping[axis]):
            assert (
                soft_label is False
            ), "parallel_cross_entropy does not support soft_label now."
            assert (
                axis == logits_ndim - 1
            ), "parallel_cross_entropy can only support shard on the last dim now."
            op_dist_attr.impl_idx = 1
        else:
            op_dist_attr.impl_idx = 0

        return False


register_distributed_operator_impl_container(
    DistributedCrossEntropy("softmax_with_cross_entropy")
)


# The softmax_norm axis is not sharded
class DistributedCrossEntropyImpl0(DistributedOperatorImpl):
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
        assert (
            op_dist_attr is not None
        ), f"forward op [{str(src_op)}] don't have dist attribute !"

        # check validation of inputs / outputs
        assert 'Logits' in kwargs, "input [Logits] is not given"
        assert 'Label' in kwargs, "input [Label] is not given"
        assert 'Loss' in kwargs, "output [Loss] is not given"
        assert 'Softmax' in kwargs, "output [Softmax] is not given"

        assert (
            len(kwargs['Logits']) == 1
        ), "input [Logits] take 1 variable but got {}".format(kwargs['Logits'])
        assert (
            len(kwargs['Label']) == 1
        ), "input [Label] take 1 variable but got {}".format(kwargs['Label'])

        logits_var = main_block._var_recursive(kwargs['Logits'][0])
        label_var = main_block._var_recursive(kwargs['Label'][0])
        loss_var = main_block._var_recursive(kwargs['Loss'][0])
        softmax_var = main_block._var_recursive(kwargs['Softmax'][0])

        # got dist attribute info
        process_mesh_shape = op_dist_attr.process_mesh.shape
        process_mesh_group = op_dist_attr.process_mesh.process_ids

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in process_mesh_group:
            rank_id = _get_corresponding_rank(
                ctx, op_dist_attr.process_mesh, rank_id
            )

        check_variable_and_dtype(
            logits_var,
            'input',
            ['float32', 'float64'],
            'cross_entropy_with_softmax',
        )
        check_variable_and_dtype(
            label_var,
            'input',
            ['int32', 'int64', 'float32', 'float64'],
            'cross_entropy_with_softmax',
        )
        check_variable_and_dtype(
            loss_var,
            'output',
            ['float32', 'float64'],
            'cross_entropy_with_softmax',
        )
        check_variable_and_dtype(
            softmax_var,
            'output',
            ['float32', 'float64'],
            'cross_entropy_with_softmax',
        )

        cross_entropy_op = copy_op_without_infer_shape(
            src_op, main_block, ctx, kwargs
        )

        # set dist op's dist_attr with serial op's dist_attr
        copied_op_dist_attr = OperatorDistAttr()
        copied_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        copied_op_dist_attr.impl_type = op_dist_attr.impl_type
        copied_op_dist_attr.impl_idx = op_dist_attr.impl_idx
        for input_varname in cross_entropy_op.desc.input_arg_names():
            input_dist_attr = op_dist_attr.get_input_dist_attr(input_varname)
            assert input_dist_attr is not None, f"dist_attr is {op_dist_attr}"
            copied_op_dist_attr.set_input_dist_attr(
                input_varname, input_dist_attr
            )
        for output_varname in cross_entropy_op.desc.output_arg_names():
            output_dist_attr = op_dist_attr.get_output_dist_attr(output_varname)
            assert output_dist_attr is not None, f"dist_attr is {op_dist_attr}"
            copied_op_dist_attr.set_output_dist_attr(
                output_varname, output_dist_attr
            )
        ctx.set_op_dist_attr_for_program(cross_entropy_op, copied_op_dist_attr)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        backward_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        op_dist_attr = ctx.get_op_dist_attr_for_program(backward_op)

        assert (
            op_dist_attr is not None
        ), f"backward op [{str(backward_op)}] don't have dist attribute !"

        # check validation of inputs / outputs
        assert 'Softmax' in kwargs, "input [Logits] is not given"
        assert 'Label' in kwargs, "input [Label] is not given"
        assert 'Loss@GRAD' in kwargs, "input [Loss@GRAD] is not given"
        assert 'Logits@GRAD' in kwargs, "output [Logits@GRAD] is not given"

        assert (
            len(kwargs['Softmax']) == 1
        ), "input [Softmax] take 1 variable but got {}".format(
            kwargs['Softmax']
        )
        assert (
            len(kwargs['Label']) == 1
        ), "input [Label] take 1 variable but got {}".format(kwargs['Label'])
        assert (
            len(kwargs['Loss@GRAD']) == 1
        ), "input [Loss@GRAD] take 1 variable but got {}".format(kwargs['Out'])
        assert (
            len(kwargs['Logits@GRAD']) == 1
        ), "output [Logits@GRAD] take 1 variable but got {}".format(
            kwargs['Logits@GRAD']
        )

        # replicate op in dist program
        dist_op_desc = main_block.append_op(type='nop').desc
        dist_op_desc.copy_from(backward_op.desc)
        # Refer to the related dist op
        set_dist_op_desc_original_id(dist_op_desc, backward_op.desc, ctx)
        for input_name in backward_op.desc.input_names():
            dist_op_desc.set_input(input_name, kwargs[input_name])
        for output_name in backward_op.desc.output_names():
            dist_op_desc.set_output(output_name, kwargs[output_name])

        # data parallel gradient synchronization
        # act_grad_names = []
        # for input_name in backward_op.desc.input_names():
        #     for varname in backward_op.desc.input(input_name):
        #         if "@GRAD" not in varname and not is_parameter_related(
        #             varname, main_block
        #         ):
        #             act_grad_names.append(varname)

        # out_grad_names = []
        # for output_name in backward_op.desc.output_names():
        #     for varname in backward_op.desc.output(output_name):
        #         if varname in kwargs["grad_var_to_var"]:
        #             fwd_name = kwargs["grad_var_to_var"][varname]
        #             if not main_block._find_var_recursive(fwd_name):
        #                 continue
        #             if is_parameter_related(fwd_name, main_block):
        #                 out_grad_names.append(varname)

        # gradient_synchronization(
        #     ctx, backward_op, act_grad_names, out_grad_names, rank_id
        # )


class DistributedCrossEntropyImpl1(DistributedOperatorImpl):
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
        assert (
            op_dist_attr is not None
        ), f"forward op [{str(src_op)}] don't have dist attribute !"

        # check validation of inputs / outputs
        assert 'Logits' in kwargs, "input [Logits] is not given"
        assert 'Label' in kwargs, "input [Label] is not given"
        assert 'Loss' in kwargs, "output [Loss] is not given"
        assert 'Softmax' in kwargs, "output [Softmax] is not given"

        assert (
            len(kwargs['Logits']) == 1
        ), "input [Logits] take 1 variable but got {}".format(kwargs['Logits'])
        assert (
            len(kwargs['Label']) == 1
        ), "input [Label] take 1 variable but got {}".format(kwargs['Label'])

        logits_var = main_block._var_recursive(kwargs['Logits'][0])
        label_var = main_block._var_recursive(kwargs['Label'][0])
        loss_var = main_block._var_recursive(kwargs['Loss'][0])
        softmax_var = main_block._var_recursive(kwargs['Softmax'][0])

        # got dist attribute info
        process_mesh_shape = op_dist_attr.process_mesh.shape
        process_mesh_group = op_dist_attr.process_mesh.process_ids

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in process_mesh_group:
            rank_id = _get_corresponding_rank(
                ctx, op_dist_attr.process_mesh, rank_id
            )

        check_variable_and_dtype(
            logits_var,
            'input',
            ['float16', 'float32', 'float64'],
            'c_softmax_with_cross_entropy',
        )
        check_variable_and_dtype(
            label_var,
            'input',
            ['int32', 'int64', 'float32', 'float64'],
            'c_softmax_with_cross_entropy',
        )
        check_variable_and_dtype(
            loss_var,
            'output',
            ['float16', 'float32', 'float64'],
            'c_softmax_with_cross_entropy',
        )
        check_variable_and_dtype(
            softmax_var,
            'output',
            ['float16', 'float32', 'float64'],
            'c_softmax_with_cross_entropy',
        )

        # infer new var shape with op dist attr
        # the dims mapping in dist_op may be different from that in tensor
        # so we should
        loss_dist_attr = ctx.get_tensor_dist_attr_for_program(loss_var)
        assert loss_dist_attr is not None
        softmax_dist_attr = ctx.get_tensor_dist_attr_for_program(softmax_var)
        assert softmax_dist_attr is not None
        op_dist_attr_loss = op_dist_attr.get_output_dist_attr(loss_var.name)
        assert op_dist_attr_loss is not None
        op_dist_attr_softmax = op_dist_attr.get_output_dist_attr(
            softmax_var.name
        )
        assert op_dist_attr_softmax is not None
        loss_ref_shape = infer_shape(
            main_block, loss_var, loss_dist_attr, op_dist_attr_loss
        )
        softmax_ref_shape = infer_shape(
            main_block, softmax_var, softmax_dist_attr, op_dist_attr_softmax
        )
        loss_var.desc.set_shape(loss_ref_shape)
        softmax_var.desc.set_shape(softmax_ref_shape)

        # TODO calculate ring id
        softmax_axis = src_op.desc.attr('axis')
        logits_dims_mapping = op_dist_attr.get_input_dims_mapping(
            logits_var.name
        )
        parallel_axis = logits_dims_mapping[softmax_axis]
        group_ranks = _get_comm_group(
            process_mesh_group, process_mesh_shape, parallel_axis, rank_id
        )
        print("group_ranks:", group_ranks)
        group = new_process_group(group_ranks)
        print("rank:", rank_id, "group:", group)

        c_cross_entropy_op = main_block.append_op(
            type='c_softmax_with_cross_entropy',
            inputs={
                'Logits': logits_var,
                'Label': label_var,
            },
            outputs={
                'Loss': loss_var,
                'Softmax': softmax_var,
            },
            attrs={
                'ring_id': group.id,
                'rank': group.local_rank(rank_id),
                'nranks': group.nranks,
                'ignore_index': src_op.desc.attr('ignore_index'),
                OP_ROLE_KEY: src_op.attr('op_role'),
            },
        )

        # set dist op's dist_attr with serial op's dist_attr
        copied_op_dist_attr = OperatorDistAttr()
        copied_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        copied_op_dist_attr.impl_type = op_dist_attr.impl_type
        copied_op_dist_attr.impl_idx = op_dist_attr.impl_idx
        for input_varname in c_cross_entropy_op.desc.input_arg_names():
            input_dist_attr = op_dist_attr.get_input_dist_attr(input_varname)
            assert input_dist_attr is not None, f"dist_attr is {op_dist_attr}"
            copied_op_dist_attr.set_input_dist_attr(
                input_varname, input_dist_attr
            )
        for output_varname in c_cross_entropy_op.desc.output_arg_names():
            output_dist_attr = op_dist_attr.get_output_dist_attr(output_varname)
            assert output_dist_attr is not None, f"dist_attr is {op_dist_attr}"
            copied_op_dist_attr.set_output_dist_attr(
                output_varname, output_dist_attr
            )
        ctx.set_op_dist_attr_for_program(
            c_cross_entropy_op, copied_op_dist_attr
        )

    @staticmethod
    def backward(ctx, *args, **kwargs):
        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        backward_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        op_dist_attr = ctx.get_op_dist_attr_for_program(backward_op)

        assert (
            op_dist_attr is not None
        ), f"backward op [{str(backward_op)}] don't have dist attribute !"

        # check validation of inputs / outputs
        assert 'Softmax' in kwargs, "input [Softmax] is not given"
        assert 'Label' in kwargs, "input [Label] is not given"
        assert 'Loss@GRAD' in kwargs, "input [Loss@GRAD] is not given"
        assert 'Logits@GRAD' in kwargs, "output [Logits@GRAD] is not given"

        assert (
            len(kwargs['Softmax']) == 1
        ), "input [Softmax] take 1 variable but got {}".format(
            kwargs['Softmax']
        )
        assert (
            len(kwargs['Label']) == 1
        ), "input [Label] take 1 variable but got {}".format(kwargs['Label'])
        assert (
            len(kwargs['Loss@GRAD']) == 1
        ), "input [Loss@GRAD] take 1 variable but got {}".format(
            kwargs['Loss@GRAD']
        )
        assert (
            len(kwargs['Logits@GRAD']) == 1
        ), "output [Logits@GRAD] take 1 variable but got {}".format(
            kwargs['Logits@GRAD']
        )

        # got dist attribute info
        process_mesh_shape = op_dist_attr.process_mesh.shape
        process_mesh_group = op_dist_attr.process_mesh.process_ids

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in process_mesh_group:
            rank_id = _get_corresponding_rank(
                ctx, op_dist_attr.process_mesh, rank_id
            )

        # reduce_mean = False
        for op in main_block.ops:
            # print("op_type:", op.type, "input_names:", op.input_names, "input_arg_names:", op.input_arg_names)
            if (
                op.type == "reduce_mean_grad"
                and kwargs['Loss@GRAD'][0] in op.output_arg_names
            ):
                loss_grad_var = main_block._var_recursive(
                    kwargs['Loss@GRAD'][0]
                )
                dp_group = get_data_parallel_group(
                    ctx, backward_op, kwargs['Loss@GRAD'], rank_id
                )
                dp_degree = len(dp_group.ranks)
                scale_op = main_block.append_op(
                    type='scale',
                    inputs={'X': loss_grad_var},
                    outputs={'Out': loss_grad_var},
                    attrs={
                        'scale': 1.0 / dp_degree,
                        OP_ROLE_KEY: OpRole.Backward,
                    },
                )
                scale_op._set_attr(
                    'op_namescope', '/' + ParallelMode.DataParallel
                )
                dims_mapping = op_dist_attr.get_input_dims_mapping(
                    loss_grad_var.name
                )
                scale_op_attr = OperatorDistAttr()
                scale_op_attr.process_mesh = op_dist_attr.process_mesh
                scale_op_attr.set_output_dims_mapping(
                    loss_grad_var.name, dims_mapping
                )
                scale_op_attr.set_input_dims_mapping(
                    loss_grad_var.name, dims_mapping
                )
                ctx.set_op_dist_attr_for_program(scale_op, scale_op_attr)

        # TODO calculate ring id
        softmax_axis = backward_op.desc.attr('axis')
        # softmax_dims_mapping is the same as logits_dims_mapping
        softmax_dims_mapping = op_dist_attr.get_input_dims_mapping(
            kwargs['Softmax'][0]
        )
        parallel_axis = softmax_dims_mapping[softmax_axis]
        group_ranks = _get_comm_group(
            process_mesh_group, process_mesh_shape, parallel_axis, rank_id
        )
        group = new_process_group(group_ranks)

        cross_entropy_grad_op_desc = main_block.append_op(type='nop').desc
        cross_entropy_grad_op_desc.set_type("c_softmax_with_cross_entropy_grad")
        cross_entropy_grad_op_desc.set_input('Softmax', [kwargs['Softmax'][0]])
        cross_entropy_grad_op_desc.set_input('Label', [kwargs['Label'][0]])
        cross_entropy_grad_op_desc.set_input(
            'Loss@GRAD', [kwargs['Loss@GRAD'][0]]
        )
        cross_entropy_grad_op_desc.set_output(
            'Logits@GRAD', [kwargs['Logits@GRAD'][0]]
        )

        ignore_index = backward_op.desc.attr('ignore_index')
        # the ignore_index attribute in c_cross_entropy_grad kernel
        # is int64_t type, so we should set this attribute with
        # _set_int64_attr. Otherwise ignore_index will be int32 type,
        # causing an error.
        cross_entropy_grad_op_desc._set_int64_attr('ignore_index', ignore_index)
        cross_entropy_grad_op_desc._set_attr('ring_id', group.id)
        cross_entropy_grad_op_desc._set_attr('rank', group.local_rank(rank_id))
        cross_entropy_grad_op_desc._set_attr('nranks', group.nranks)
        cross_entropy_grad_op_desc._set_attr(OP_ROLE_KEY, OpRole.Backward)

        cross_entropy_grad_op = main_block.ops[-1]
        naive_copy_op_dist_attr_for_program(
            cross_entropy_grad_op, backward_op, ctx
        )

        # data parallel gradient synchronization
        # act_grad_names = []
        # for input_name in backward_op.desc.input_names():
        #     for varname in backward_op.desc.input(input_name):
        #         if "@GRAD" not in varname and not is_parameter_related(
        #             varname, main_block
        #         ):
        #             act_grad_names.append(varname)

        # out_grad_names = []
        # for output_name in backward_op.desc.output_names():
        #     for varname in backward_op.desc.output(output_name):
        #         if varname in kwargs["grad_var_to_var"]:
        #             fwd_name = kwargs["grad_var_to_var"][varname]
        #             if not main_block._find_var_recursive(fwd_name):
        #                 continue
        #             if is_parameter_related(fwd_name, main_block):
        #                 out_grad_names.append(varname)

        # print(act_grad_names)
        # gradient_synchronization(
        #     ctx, backward_op, act_grad_names, out_grad_names, rank_id
        # )


register_distributed_operator_impl(
    "softmax_with_cross_entropy", DistributedCrossEntropyImpl0("cross_entropy")
)
register_distributed_operator_impl(
    "softmax_with_cross_entropy",
    DistributedCrossEntropyImpl1("c_cross_entropy"),
)
