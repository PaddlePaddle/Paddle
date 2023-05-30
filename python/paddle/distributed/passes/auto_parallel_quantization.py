# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
from paddle.framework import IrGraph, core
from paddle.static.quantization import (
    AddQuantDequantForInferencePass,
    AddQuantDequantPassV2,
    OutScaleForTrainingPass,
    QuantizationTransformPassV2,
    quant_config,
)

from ..auto_parallel.static.converter import Converter
from ..auto_parallel.static.dist_attribute import (
    OperatorDistAttr,
    TensorDistAttr,
)
from .pass_base import PassBase, register_pass

TRANSFORM_PASS_OP_TYPES = list(
    quant_config.SUPPORT_WEIGHT_QUANTIZATION_OP_DICT.keys()
)
QUANT_DEQUANT_PASS_OP_TYPES = list(
    quant_config.SUPPORT_ACT_QUANTIZATION_OP_DICT.keys()
)


def _node_id(node):
    return (node.node.graph_id(), node.node.id())


@register_pass("auto_parallel_quantization")
class QuantizationPass(PassBase):
    def __init__(self):
        super().__init__()
        self.set_attr("dist_context", None)
        self.set_attr("params_grads", None)
        self.set_attr("mode", "train")
        self.set_attr("loss", None)

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False
        if self.get_attr("params_grads") is None:
            return False
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):

        dist_context = self.get_attr("dist_context")
        params_grads = self.get_attr("params_grads")
        mode = self.get_attr("mode")
        loss = self.get_attr("loss")

        # TODO: scope and place will be removed,
        # cause params should be initialized by engine module.
        scope = paddle.static.global_scope()
        place = paddle.framework.CUDAPlace(
            paddle.distributed.ParallelEnv().dev_id
        )

        # 0. record the relation among blocks
        parent_idx_dict = {}
        for block in main_program.blocks:
            parent_idx_dict[block.idx] = block.parent_idx

        is_test = True if mode != "train" else False
        # 1. Program convert to Graph, and this pass is only for train mode
        main_graph = IrGraph(
            core.Graph(main_program.desc), for_test=mode != "train"
        )

        # 2. Prepare inputs
        transform_pass_ops = []
        quant_dequant_ops = []
        quantize_op_types = [
            'conv2d',
            'depthwise_conv2d',
            'mul',
            'matmul',
            'matmul_v2',
        ]
        for op_type in quantize_op_types:
            if op_type in TRANSFORM_PASS_OP_TYPES:
                transform_pass_ops.append(op_type)
            elif op_type in QUANT_DEQUANT_PASS_OP_TYPES:
                quant_dequant_ops.append(op_type)

        weight_quantize_type = (
            "channel_wise_abs_max"
            if self.get_attr('channel_wise_abs_max')
            else "abs_max"
        )

        # 3. Add quant op for ops which have parameters
        if len(transform_pass_ops) > 0:
            transform_pass = QuantizationTransformPassV2(
                scope=scope,
                place=place,
                weight_bits=self.get_attr('weight_bits'),
                activation_bits=self.get_attr('activation_bits'),
                skip_pattern=self.get_attr('not_quant_pattern'),
                activation_quantize_type="moving_average_abs_max",
                quantizable_op_type=transform_pass_ops,
                weight_quantize_type=weight_quantize_type,
                weight_quantize_func=None,
                act_quantize_func=None,
                weight_preprocess_func=None,
                act_preprocess_func=None,
                optimizer_func=None,
                executor=None,
                is_test=is_test,
            )
            for sub_graph in main_graph.all_sub_graphs():
                transform_pass.apply(sub_graph)

        # 4. Add quant op for ops which don't have parameter
        if len(quant_dequant_ops) > 0:
            quant_dequant_pass = AddQuantDequantPassV2(
                scope=scope,
                place=place,
                quant_bits=self.get_attr('activation_bits'),
                skip_pattern=self.get_attr('not_quant_pattern'),
                quantizable_op_type=quant_dequant_ops,
                is_test=is_test,
            )
            for sub_graph in main_graph.all_sub_graphs():
                quant_dequant_pass.apply(sub_graph)

        # 5. Gather quantitative information for the output
        out_scale_training_pass = OutScaleForTrainingPass(
            scope=scope, place=place, is_test=is_test
        )
        for sub_graph in main_graph.all_sub_graphs():
            out_scale_training_pass.apply(sub_graph)

        # 6. When export quant model, traverse to find the output of each op, and insert the quant/dequant op after it.
        if mode != "train" and self.get_attr('onnx_format'):
            try:
                out_scale_infer_pass = AddQuantDequantForInferencePass(
                    scope=scope,
                    place=place,
                    quant_bits=self.get_attr('activation_bits'),
                )
                # for sub_graph in main_graph.all_sub_graphs():
                #     out_scale_infer_pass.apply(sub_graph)
            except:
                logging.warning(
                    "Unable to convert quant model with onnx_format=True, please update PaddlePaddle >= 2.4.0"
                )

        # 7. Convert Graph back to Program
        quant_program = main_graph.to_program()
        quant_program = self.move_presist_var_to_global_block(quant_program)

        # 8.1 get new prams_grads from quant_program
        new_params_grads = []
        for param, grad in params_grads:
            if param.name not in quant_program.global_block().vars:
                continue

            new_param = quant_program.global_block().vars[param.name]
            new_grad = quant_program.global_block().vars[grad.name]
            new_params_grads.append((new_param, new_grad))

        # 8.2 get new loss var
        new_loss = None
        if loss:
            new_loss = quant_program.global_block().vars[loss.name]

        # 8.3 recover the relation among blocks
        for block in quant_program.blocks:
            block.desc._set_forward_block_idx(parent_idx_dict[block.idx])

        # 9. complete distributed attribution
        self.set_dist_attr_for_qat_program(
            quant_program, main_program, dist_context
        )

        # 10. reset scale var value with dist_attr
        self.reset_scope_var(quant_program, dist_context, scope, place)

        context.set_attr("main_program", quant_program)
        context.set_attr("startup_program", startup_program)
        context.set_attr("params_grads", new_params_grads)
        context.set_attr("loss", new_loss)

    def move_presist_var_to_global_block(self, program):
        global_block = program.global_block()
        for _op in global_block.ops:
            if _op.type == "while":
                _block_id = _op.attr("sub_block").id
                _block = program.block(_block_id)
                persistables = []
                for _name, _var in _block.vars.items():
                    if _var.persistable:
                        global_block._clone_variable(_var)
                        persistables.append(_name)
                for _name in persistables:
                    _block._remove_var(_name)
                persistables.extend(_op.input('X'))
                _op.desc.set_input("X", persistables)
        return program

    def reset_scope_var(self, quant_program, dist_context, scope, place):
        # The var_value, created by qatization_passes, should has same shape with the value after parallel.
        for var in quant_program.list_vars():
            scope_var = scope.find_var(var.name)
            if not (scope_var and scope_var.get_tensor()._is_initialized()):
                continue
            tensor = scope_var.get_tensor()
            if var.shape == tensor.shape:
                continue

            var_dist_attr = dist_context.get_tensor_dist_attr_for_program(var)
            dist_attr = {
                "dims_mapping": var_dist_attr.dims_mapping,
                "process_shape": var_dist_attr.process_mesh.shape,
                "process_group": var_dist_attr.process_mesh.process_ids,
            }

            # slice tensor_value with dist_attr
            sliced_tensor = Converter.slice_with_dist_attr(
                np.array(tensor), dist_attr
            )
            tensor._clear()
            tensor.set(sliced_tensor, place)

    def set_dist_attr_for_qat_program(
        self, quant_program, main_program, dist_context
    ):
        # NOTE: hack implement, upgrading soon
        for ib, block in enumerate(quant_program.blocks):
            # recover origin ops' dist_attr and set quant ops' dist_attr
            qat_offset = 0
            for ip, quant_op in enumerate(block.ops):
                quant_op_dist_attr = OperatorDistAttr()

                if (
                    "quantize" in quant_op.type
                    or quant_op.type == "moving_average_abs_max_scale"
                ):
                    # set all quantization ops' dist_attr by quantified op
                    input_name = quant_op.desc.input('X')[0]
                    if "quantize" in input_name:
                        input_name = input_name[
                            : input_name.index(".quantized")
                        ]

                    if (
                        quant_op.type == "moving_average_abs_max_scale"
                        or ip - qat_offset >= len(main_program.blocks[ib].ops)
                    ):
                        consume_op = (
                            main_program.blocks[ib]
                            ._var_recursive(input_name)
                            .op
                        )
                    else:
                        consume_op = main_program.blocks[ib].ops[
                            ip - qat_offset
                        ]
                    consume_op_dist_attr = dist_context.get_dist_op_for_program(
                        consume_op
                    ).dist_attr
                    ref_process_mesh = consume_op_dist_attr.process_mesh

                    if input_name in consume_op_dist_attr.outputs_dist_attrs:
                        consume_input_dist_attr = (
                            consume_op_dist_attr.outputs_dist_attrs[input_name]
                        )
                    else:
                        consume_input_dist_attr = (
                            consume_op_dist_attr.inputs_dist_attrs[input_name]
                        )

                    quant_op_dist_attr.impl_idx = 0
                    quant_op_dist_attr.impl_type = "default"
                    quant_op_dist_attr.process_mesh = ref_process_mesh
                    quant_op_dist_attr.set_input_dist_attr(
                        quant_op.desc.input('X')[0], consume_input_dist_attr
                    )

                    for slot_name in quant_op.desc.input_names():
                        in_name = quant_op.desc.input(slot_name)[0]
                        input_var = block._var_recursive(in_name)
                        ref_dims_mapping = [-1 for i in input_var.shape]
                        if slot_name == "X":
                            continue
                        elif slot_name in ['Scale', 'ZeroPoint']:
                            if (
                                quant_op.has_attr('quant_axis')
                                and quant_op.attr('quant_axis') != -1
                            ):
                                x_name = quant_op.desc.input('X')[0]
                                x_var = block._var_recursive(x_name)
                                x_dist_attr = (
                                    quant_op_dist_attr.get_input_dist_attr(
                                        x_name
                                    )
                                )
                                quant_axis = quant_op.attr('quant_axis')
                                ref_dims_mapping = [
                                    x_dist_attr.dims_mapping[quant_axis]
                                ]

                        tensor_dist_attr = TensorDistAttr()
                        tensor_dist_attr.process_mesh = ref_process_mesh
                        tensor_dist_attr.dims_mapping = ref_dims_mapping
                        dist_context.set_tensor_dist_attr_for_program(
                            input_var, tensor_dist_attr
                        )
                        quant_op_dist_attr.set_input_dist_attr(
                            in_name, tensor_dist_attr
                        )

                    for slot_name in quant_op.desc.output_names():
                        output_name = quant_op.desc.output(slot_name)[0]
                        output_var = block._var_recursive(output_name)
                        ref_dims_mapping = [-1 for i in output_var.shape]
                        if slot_name == "Y":
                            dist_context.set_tensor_dist_attr_for_program(
                                output_var, consume_input_dist_attr
                            )
                            quant_op_dist_attr.set_output_dist_attr(
                                output_name, consume_input_dist_attr
                            )
                            continue
                        elif slot_name == "OutScale":
                            if (
                                quant_op.has_attr('quant_axis')
                                and quant_op.attr('quant_axis') != -1
                            ):
                                x_name = quant_op.desc.input('X')[0]
                                x_var = block._var_recursive(x_name)
                                x_dist_attr = (
                                    quant_op_dist_attr.get_input_dist_attr(
                                        x_name
                                    )
                                )
                                quant_axis = quant_op.attr('quant_axis')
                                ref_dims_mapping = [
                                    x_dist_attr.dims_mapping[quant_axis]
                                ]

                        tensor_dist_attr = TensorDistAttr()
                        tensor_dist_attr.process_mesh = ref_process_mesh
                        tensor_dist_attr.dims_mapping = ref_dims_mapping
                        dist_context.set_tensor_dist_attr_for_program(
                            output_var, tensor_dist_attr
                        )
                        quant_op_dist_attr.set_output_dist_attr(
                            output_name, tensor_dist_attr
                        )

                    quant_op._set_attr("op_device", "")
                    qat_offset += 1

                else:
                    # recover origin ops' dist_attr
                    origin_op = main_program.blocks[ib].ops[ip - qat_offset]
                    quant_op.desc.set_original_id(origin_op.desc.original_id())
                    dist_origin_op = dist_context.get_dist_op_for_program(
                        origin_op
                    )
                    assert (
                        dist_origin_op is not None
                    ), "origin op must have dist attr."

                    origin_op_dist_attr = dist_origin_op.dist_attr
                    quant_op_dist_attr.impl_idx = origin_op_dist_attr.impl_idx
                    quant_op_dist_attr.impl_type = origin_op_dist_attr.impl_type
                    quant_op_dist_attr.process_mesh = (
                        origin_op_dist_attr.process_mesh
                    )

                    scale_offset = 0
                    for idx, input_name in enumerate(quant_op.input_arg_names):
                        if (
                            origin_op.type == "while"
                            and input_name not in origin_op.input_arg_names
                        ):
                            assert (
                                "@scale" in input_name
                                or "@zero_point" in input_name
                            )
                            scale_offset += 1
                            continue

                        idx -= scale_offset
                        origin_input_name = origin_op.input_arg_names[idx]
                        origin_input_dist_attr = (
                            origin_op_dist_attr.inputs_dist_attrs[
                                origin_input_name
                            ]
                        )
                        quant_op_dist_attr.set_input_dist_attr(
                            input_name, origin_input_dist_attr
                        )

                    for idx, output_name in enumerate(
                        quant_op.output_arg_names
                    ):
                        origin_output_name = origin_op.output_arg_names[idx]
                        origin_output_dist_attr = (
                            origin_op_dist_attr.outputs_dist_attrs[
                                origin_output_name
                            ]
                        )
                        quant_op_dist_attr.set_output_dist_attr(
                            output_name, origin_output_dist_attr
                        )

                        if not main_program.blocks[ib]._find_var_recursive(
                            output_name
                        ):
                            origin_output_var = main_program.blocks[
                                ib
                            ]._var_recursive(origin_output_name)
                            origin_out_tensor_dist_attr = (
                                dist_context.get_dist_tensor_for_program(
                                    origin_output_var
                                ).dist_attr
                            )
                            quant_output_var = block._var_recursive(output_name)
                            dist_context.set_tensor_dist_attr_for_program(
                                quant_output_var, origin_out_tensor_dist_attr
                            )

                dist_context.set_op_dist_attr_for_program(
                    quant_op, quant_op_dist_attr
                )

            # recover vars' dist_attr
            for name, dst_var in block.vars.items():
                if name in main_program.blocks[ib].vars:
                    src_var = main_program.blocks[ib].vars[name]
                    dist_tensor = dist_context.get_dist_tensor_for_program(
                        src_var
                    )
                    if not dist_tensor:
                        continue
                    dist_context.set_tensor_dist_attr_for_program(
                        dst_var, dist_tensor.dist_attr
                    )
