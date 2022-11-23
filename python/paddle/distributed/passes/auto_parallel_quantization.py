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

import paddle

from paddle.fluid import core, framework
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.fluid.contrib.slim.quantization import utils
from paddle.fluid.contrib.slim.quantization import QuantizationTransformPassV2
from paddle.fluid.contrib.slim.quantization import AddQuantDequantPassV2
from paddle.fluid.contrib.slim.quantization import OutScaleForTrainingPass
from paddle.distributed.auto_parallel.dist_attribute import OperatorDistributedAttribute, TensorDistributedAttribute

from .pass_base import PassBase, register_pass

TRANSFORM_PASS_OP_TYPES = utils._weight_supported_quantizable_op_type
QUANT_DEQUANT_PASS_OP_TYPES = utils._act_supported_quantizable_op_type


def _node_id(node):
    return (node.node.graph_id(), node.node.id())


@register_pass("auto_parallel_quantization")
class QuantizationPass(PassBase):

    def __init__(self):
        super(QuantizationPass, self).__init__()
        self.set_attr("dist_context", None)
        self.set_attr("params_grads", None)

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

        # TODO: scope and place will be removed,
        # cause params should be initialized by engine module.
        scope = paddle.static.global_scope()
        place = paddle.fluid.CUDAPlace(ParallelEnv().dev_id)

        # 1. Program convert to Graph, and this pass is only for train mode
        main_graph = framework.IrGraph(core.Graph(main_program.desc),
                                       for_test=False)

        # 2. Prepare inputs
        transform_pass_ops = []
        quant_dequant_ops = []
        quantize_op_types = [
            'conv2d', 'depthwise_conv2d', 'mul', 'matmul', 'matmul_v2'
        ]
        for op_type in quantize_op_types:
            if op_type in TRANSFORM_PASS_OP_TYPES:
                transform_pass_ops.append(op_type)
            elif op_type in QUANT_DEQUANT_PASS_OP_TYPES:
                quant_dequant_ops.append(op_type)

        weight_quantize_type = "channel_wise_abs_max" if self.get_attr(
            'channel_wise_abs_max') else "abs_max"

        # 3. Add quant op for ops which have parameters
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
            executor=None)
        transform_pass.apply(main_graph)

        # 4. Add quant op for ops which don't have parameter
        quant_dequant_pass = AddQuantDequantPassV2(
            scope=scope,
            place=place,
            quant_bits=self.get_attr('activation_bits'),
            skip_pattern=self.get_attr('not_quant_pattern'),
            quantizable_op_type=quant_dequant_ops)
        quant_dequant_pass.apply(main_graph)

        # 5. Gather quantitative information for the output
        out_scale_training_pass = OutScaleForTrainingPass(scope=scope,
                                                          place=place)
        out_scale_training_pass.apply(main_graph)

        # 6. Convert Graph back to Program
        quant_program = main_graph.to_program()

        # 7. get new prams_grads from quant_program
        new_params_grads = []
        for param, grad in params_grads:
            if param.name not in quant_program.global_block().vars:
                continue

            new_param = quant_program.global_block().vars[param.name]
            new_grad = quant_program.global_block().vars[grad.name]
            new_params_grads.append((new_param, new_grad))

        # 8. complete distributed attribution
        # NOTE: hack implement, upgrading soon
        for ib, block in enumerate(quant_program.blocks):
            # recover origin ops' dist_attr and set quant ops' dist_attr
            qat_offset = 0
            for ip, quant_op in enumerate(block.ops):
                quant_op_dist_attr = OperatorDistributedAttribute()

                if "quantize" in quant_op.type or \
                    quant_op.type == "moving_average_abs_max_scale":

                    input_name = quant_op.desc.input('X')[0]
                    if "quantize" in input_name:
                        input_name = input_name[:input_name.index(".quantized")]

                    if quant_op.type == "moving_average_abs_max_scale":
                        consume_op = main_program.blocks[ib].vars[input_name].op
                    else:
                        consume_op = main_program.blocks[ib].ops[ip -
                                                                 qat_offset]
                    consume_op_dist_attr = dist_context.get_dist_op_for_program(
                        consume_op).dist_attr
                    ref_process_mesh = consume_op_dist_attr.process_mesh

                    if input_name in consume_op_dist_attr.outputs_dist_attrs:
                        consume_input_dist_attr = consume_op_dist_attr.outputs_dist_attrs[
                            input_name]
                    else:
                        consume_input_dist_attr = consume_op_dist_attr.inputs_dist_attrs[
                            input_name]

                    quant_op_dist_attr.impl_idx = 0
                    quant_op_dist_attr.impl_type = "default"
                    quant_op_dist_attr.process_mesh = ref_process_mesh
                    quant_op_dist_attr.set_input_dist_attr(
                        quant_op.desc.input('X')[0], consume_input_dist_attr)

                    for slot_name in quant_op.desc.input_names():
                        if slot_name == "X":
                            continue
                        for in_name in quant_op.desc.input(slot_name):
                            input_var = block.vars[in_name]
                            tensor_dist_attr = TensorDistributedAttribute()
                            tensor_dist_attr.process_mesh = ref_process_mesh
                            tensor_dist_attr.dims_mapping = [-1]
                            dist_context.set_tensor_dist_attr_for_program(
                                input_var, tensor_dist_attr)
                            quant_op_dist_attr.set_input_dist_attr(
                                in_name, tensor_dist_attr)

                    for slot_name in quant_op.desc.output_names():
                        output_name = quant_op.desc.output(slot_name)[0]
                        output_var = block.vars[output_name]
                        if slot_name == "Y":
                            dist_context.set_tensor_dist_attr_for_program(
                                output_var, consume_input_dist_attr)
                            quant_op_dist_attr.set_output_dist_attr(
                                output_name, consume_input_dist_attr)
                        else:
                            tensor_dist_attr = TensorDistributedAttribute()
                            tensor_dist_attr.process_mesh = ref_process_mesh
                            tensor_dist_attr.dims_mapping = [-1]
                            dist_context.set_tensor_dist_attr_for_program(
                                output_var, tensor_dist_attr)
                            quant_op_dist_attr.set_output_dist_attr(
                                output_name, tensor_dist_attr)

                    quant_op._set_attr("op_device", "")
                    qat_offset += 1

                else:

                    origin_op = main_program.blocks[ib].ops[ip - qat_offset]
                    quant_op.desc.set_original_id(origin_op.desc.original_id())
                    dist_origin_op = dist_context.get_dist_op_for_program(
                        origin_op)
                    assert dist_origin_op is not None, "origin op must have dist attr."

                    origin_op_dist_attr = dist_origin_op.dist_attr
                    quant_op_dist_attr.impl_idx = origin_op_dist_attr.impl_idx
                    quant_op_dist_attr.impl_type = origin_op_dist_attr.impl_type
                    quant_op_dist_attr.process_mesh = origin_op_dist_attr.process_mesh
                    for idx, input_name in enumerate(quant_op.input_arg_names):
                        origin_input_name = origin_op.input_arg_names[idx]
                        origin_input_dist_attr = origin_op_dist_attr.inputs_dist_attrs[
                            origin_input_name]
                        quant_op_dist_attr.set_input_dist_attr(
                            input_name, origin_input_dist_attr)

                        if input_name not in main_program.blocks[ib].vars:
                            origin_input_var = main_program.blocks[ib].vars[
                                origin_input_name]
                            origin_in_tensor_dist_attr = dist_context.get_dist_tensor_for_program(
                                origin_input_var).dist_attr
                            quant_input_var = block.vars[input_name]
                            dist_context.set_tensor_dist_attr_for_program(
                                quant_input_var, origin_in_tensor_dist_attr)

                    for idx, output_name in enumerate(
                            quant_op.output_arg_names):
                        origin_output_name = origin_op.output_arg_names[idx]
                        origin_output_dist_attr = origin_op_dist_attr.outputs_dist_attrs[
                            origin_output_name]
                        quant_op_dist_attr.set_output_dist_attr(
                            output_name, origin_output_dist_attr)

                        if output_name not in main_program.blocks[ib].vars:
                            origin_output_var = main_program.blocks[ib].vars[
                                origin_output_name]
                            origin_out_tensor_dist_attr = dist_context.get_dist_tensor_for_program(
                                origin_output_var).dist_attr
                            quant_output_var = block.vars[output_name]
                            dist_context.set_tensor_dist_attr_for_program(
                                quant_output_var, origin_out_tensor_dist_attr)

                dist_context.set_op_dist_attr_for_program(
                    quant_op, quant_op_dist_attr)

            # recover vars' dist_attr
            for name, dst_var in block.vars.items():
                if name in main_program.blocks[ib].vars:
                    src_var = main_program.blocks[ib].vars[name]
                    dist_tensor = dist_context.get_dist_tensor_for_program(
                        src_var)
                    if not dist_tensor:
                        continue
                    dist_context.set_tensor_dist_attr_for_program(
                        dst_var, dist_tensor.dist_attr)

        context.set_attr("main_program", quant_program)
        context.set_attr("startup_program", startup_program)
        context.set_attr("params_grads", new_params_grads)
