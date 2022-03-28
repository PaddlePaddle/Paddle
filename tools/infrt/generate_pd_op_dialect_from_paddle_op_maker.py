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

import paddle.fluid.framework as framework
from paddle.fluid import core
from paddle import compat as cpt


# collect original ops: op which has both inference and grid defination
def get_original_ops():
    all_ops, _, _ = core.op_supported_infos('CPU', core.VarDesc.VarType.FP16)
    grad_ops = []
    original_ops = []
    necessary_ops = ["scale"]

    for op in all_ops:
        if op.endswith("_grad"):
            if op.endswith("_grad_grad"):
                continue
            grad_ops.append(op)
    for op in all_ops:
        if str(op + "_grad") in grad_ops:
            original_ops.append(op)
        elif op in necessary_ops:
            original_ops.append(op)

    print("Grad ops num: " + str(len(grad_ops)))
    print("Responded original ops num: " + str(len(original_ops)))
    return original_ops


# functions of parsing Paddle Proto
INPUTS = "Inputs"
OUTPUTS = "Outputs"
ATTRS = "Attrs"
COMMENT = "Comment"

DUPLICABLE = "duplicable"
INTERMEDIATE = "intermediate"
DISPENSABLE = "dispensable"

TYPE = "type"
GENERATED = "generated"
DEFAULT_VALUE = "default_value"

EXTRA = "extra"
QUANT = "quant"


def get_attr_default_value(op_name):
    return core.get_op_attrs_default_value(cpt.to_bytes(op_name))


def get_vars_info(op_vars_proto):
    vars_info = {}
    for var_proto in op_vars_proto:
        name = str(var_proto.name)
        vars_info[name] = {}
        vars_info[name][DUPLICABLE] = var_proto.duplicable
        vars_info[name][DISPENSABLE] = var_proto.dispensable
        vars_info[name][INTERMEDIATE] = var_proto.intermediate
        vars_info[name][EXTRA] = var_proto.extra
        vars_info[name][QUANT] = var_proto.quant
    return vars_info


def get_attrs_info(op_proto, op_attrs_proto):
    attrs_info = {}
    attrs_default_values = get_attr_default_value(op_proto.type)
    for attr_proto in op_attrs_proto:
        attr_name = str(attr_proto.name)
        attrs_info[attr_name] = {}
        attrs_info[attr_name][TYPE] = attr_proto.type
        attrs_info[attr_name][GENERATED] = attr_proto.generated
        attrs_info[attr_name][DEFAULT_VALUE] = attrs_default_values[
            attr_name] if attr_name in attrs_default_values else None
        attrs_info[attr_name][EXTRA] = attr_proto.extra
        attrs_info[attr_name][QUANT] = attr_proto.quant
    return attrs_info


def get_op_desc(op_proto):
    op_info = {}
    op_info[INPUTS] = get_vars_info(op_proto.inputs)
    op_info[OUTPUTS] = get_vars_info(op_proto.outputs)
    op_info[ATTRS] = get_attrs_info(op_proto, op_proto.attrs)
    op_info[COMMENT] = op_proto.comment
    return op_info


def get_all_ops_desc():
    all_op_protos_dict = {}
    all_op_protos = framework.get_all_op_protos()
    for op_proto in all_op_protos:
        op_type = str(op_proto.type)
        all_op_protos_dict[op_type] = get_op_desc(op_proto)
    return all_op_protos_dict


def generate_all_ops_inputs_outputs_map(op_descs):
    # 1. Collect input and output name information of each Op
    original_ops_ = get_original_ops()
    ops_inputs_map = {}
    ops_outputs_map = {}
    for op_type, op_proto in op_descs.items():
        if op_type not in original_ops_:
            continue
        inputs = list()
        outpus = list()
        for input_ in op_proto[INPUTS]:
            if op_proto[INPUTS][input_][EXTRA] != True and op_proto[INPUTS][
                    input_][INTERMEDIATE] != True:
                inputs.append(input_)
        for output_ in op_proto[OUTPUTS]:
            if op_proto[OUTPUTS][output_][EXTRA] != True and op_proto[OUTPUTS][
                    output_][INTERMEDIATE] != True:
                outpus.append(output_)
        ops_inputs_map[op_type] = inputs
        ops_outputs_map[op_type] = outpus

    # 2. Generate Cpp style map str
    cpp_style_ops_inputs_map_str = ""
    start_ = "#include <unordered_map>\n#include <vector>\n#include <string>\n" + \
            "const std::unordered_map<std::string, std::unordered_map<std::string, uint8_t>> pd_dialect_inputs_info_map_  = {\n"
    ops_inputs_str = ""
    for ele in ops_inputs_map.items():
        op_name = ele[0]
        op_inputs = ele[1]
        op_inputs_str = "{"
        input_idx = 0
        for op_input in op_inputs:
            op_input_str = '{left_brace}"{op_input}", {input_idx}{right_brace}, '.format(
                left_brace="{",
                op_input=op_input,
                input_idx=input_idx,
                right_brace="}")
            input_idx = input_idx + 1
            op_inputs_str = op_inputs_str + op_input_str
        op_inputs_str = op_inputs_str[:-2] + "}"
        pair = '{left_brace}"{op_name}", {op_inputs}{right_brace},\n'.format(
            left_brace="{",
            op_name=op_name,
            op_inputs=op_inputs_str,
            right_brace="}")
        ops_inputs_str = ops_inputs_str + "    " + pair
    ops_inputs_str = ops_inputs_str[:-2]
    cpp_style_ops_inputs_map_str = start_ + ops_inputs_str + "\n};"

    cpp_style_ops_outputs_map_str = ""
    start_ = "const std::unordered_map<std::string, std::unordered_map<std::string, uint8_t>> pd_dialect_outputs_info_map_  = {\n"
    ops_outputs_str = ""
    for ele in ops_outputs_map.items():
        op_name = ele[0]
        op_outputs = ele[1]
        op_outputs_str = "{"
        output_idx = 0
        for op_output in op_outputs:
            op_output_str = '{left_brace}"{op_output}", {output_idx}{right_brace}, '.format(
                left_brace="{",
                op_output=op_output,
                output_idx=output_idx,
                right_brace="}")
            output_idx = output_idx + 1
            op_outputs_str = op_outputs_str + op_output_str
        op_outputs_str = op_outputs_str[:-2] + "}"
        pair = '{left_brace}"{op_name}", {op_outputs}{right_brace},\n'.format(
            left_brace="{",
            op_name=op_name,
            op_outputs=op_outputs_str,
            right_brace="}")
        ops_outputs_str = ops_outputs_str + "    " + pair
    ops_outputs_str = ops_outputs_str[:-2]
    cpp_style_ops_outputs_map_str = start_ + ops_outputs_str + "\n};"

    # 3. Write to header file
    dst_head_file = "../../paddle/infrt/dialect/pd/common/pd_ops_info.h"
    with open(dst_head_file, 'w') as ops_inputs_outputs_head_file:
        ops_inputs_outputs_head_file.write(cpp_style_ops_inputs_map_str)
        ops_inputs_outputs_head_file.write("\n\n")
        ops_inputs_outputs_head_file.write(cpp_style_ops_outputs_map_str)


def get_constraint(op_type, op_proto):
    # 2.3.1 inputs
    constraint = "NoSideEffect"

    optional_input_num_ = 0
    for input_ in op_proto[INPUTS]:
        if op_proto[INPUTS][input_][EXTRA] != True and op_proto[INPUTS][input_][
                INTERMEDIATE] != True and op_proto[INPUTS][input_][
                    DISPENSABLE] == True:
            optional_input_num_ += 1
    if optional_input_num_ > 1:
        constraint += ", AttrSizedOperandSegments"
    return constraint


# funtion to generate paddle op dialect file
def convert_op_proto_into_mlir(op_descs):
    dst_dialect_file = "../../paddle/infrt/dialect/pd/ir/pd_ops.td"
    custom_dialect_file = "custom_pdop.td"

    # 1. Head files
    comment_ = "/*===- TableGen'source file -----------------------------------------------===*\\\n\
|*                                                                            *|\n\
|* Op Definitions                                                             *|\n\
|*                                                                            *|\n\
|* Automatically generated file, do not edit!                                 *|\n\
|* Generated by tools/infrt/generate_pd_op_dialect_from_paddle_op_maker.py    *|\n\
|*                                                                            *|\n\
\*===----------------------------------------------------------------------===*/\n"

    lines = [
        "#ifndef PD_OPS",
        "#define PD_OPS",
        "include \"mlir/Interfaces/InferTypeOpInterface.td\"",
        "include \"mlir/Interfaces/LoopLikeInterface.td\"",
        "include \"mlir/IR/OpBase.td\"",
        "include \"paddle/infrt/dialect/pd/ir/pd_op_base.td\"",
        "",
    ]

    start_ = comment_ + "\n".join(lines)

    with open(dst_dialect_file, 'w') as ops_mlir_file:
        ops_mlir_file.write(start_)

    # 2. Op dialect
    # skip list ( ops whose dialect can not be generated automatically will be recorded here)
    skipped_op_list = [
        "cos_sim", "fused_embedding_seq_pool", "cosh", "kron", "recurrent",
        "while", "conditional_block", "set_value", "run_program"
    ]
    skipped_attr_list = [
        "trainable_statistics", "use_global_stats", "is_test", "use_quantizer"
    ]

    original_ops_ = get_original_ops()
    automatically_generated_op_dialect = []
    for op_type, op_proto in op_descs.items():
        if (op_type in skipped_op_list) or (op_type not in original_ops_):
            continue
        automatically_generated_op_dialect.append(op_type)
        constraint_ = get_constraint(op_type, op_proto)
        # 2.1 OpDef
        HEAD = 'def PD_{op_type_capitalize}Op : PD_Op<"{op_type}", [{constraint}]> {left_brace}\n'.format(
            op_type_capitalize=op_type.capitalize(),
            constraint=constraint_,
            op_type=op_type,
            left_brace="{")
        SUMMARY = '  let summary = "{} op";\n'.format(op_type)

        # 2.2 Description
        contents = ""
        origin_contents = (op_proto[COMMENT]).split("\n")
        for line_ in origin_contents:
            contents = contents + "    {}\n".format(line_)
        DESCRIPTION = "  let description = [{left_brace}\n{description}  {right_brace}];\n".format(
            left_brace="{", description=contents, right_brace="}")

        # 2.3 arguments info
        ARGUMENTS = ""
        if (len(op_proto[INPUTS]) > 0 or len(op_proto[ATTRS]) > 0):
            ARGUMENTS = "  let arguments = (ins "

            # 2.3.1 inputs
            for input_ in op_proto[INPUTS]:
                if op_proto[INPUTS][input_][EXTRA] != True and op_proto[INPUTS][
                        input_][INTERMEDIATE] != True:
                    if op_proto[INPUTS][input_][DISPENSABLE] != True:
                        if op_proto[INPUTS][input_][DUPLICABLE] != True:
                            ARGUMENTS = ARGUMENTS + " PD_Tensor:$" + input_ + ","
                        else:
                            ARGUMENTS = ARGUMENTS + " PD_Tensor_Array:$" + input_ + ","
                    else:
                        if op_proto[INPUTS][input_][DUPLICABLE] != True:
                            ARGUMENTS = ARGUMENTS + " Optional<PD_Tensor>:$" + input_ + ","
                        else:
                            ARGUMENTS = ARGUMENTS + " Optional<PD_Tensor_Array>:$" + input_ + ","

            # unsupported:   BLOCK = 8;  BLOCKS = 10;
            attr_mlir_converter = {
                0: 'SI32Attr',
                1: 'F32Attr',
                2: 'StrAttr',
                3: 'I32ArrayAttr',
                4: 'F32ArrayAttr',
                5: 'StrArrayAttr',
                6: 'BoolAttr',
                7: 'BoolArrayAttr',
                9: 'SI64Attr',
                11: 'I64ArrayAttr'
            }

            # 2.3.2 attributes
            for attr in op_proto[ATTRS]:
                if (op_proto[ATTRS][attr][EXTRA] == True) or (
                        attr in skipped_attr_list):
                    continue
                if op_proto[ATTRS][attr][DEFAULT_VALUE] != None:
                    if op_proto[ATTRS][attr][TYPE] in attr_mlir_converter:
                        default_value = str(op_proto[ATTRS][attr][
                            DEFAULT_VALUE])
                        if (attr_mlir_converter[op_proto[ATTRS][attr][TYPE]] in
                            [
                                'I32ArrayAttr', 'F32ArrayAttr', 'StrArrayAttr',
                                'BoolArrayAttr', 'I64ArrayAttr'
                            ]):
                            default_value = default_value.replace(
                                '[', '{').replace(']', '}')
                        if (attr_mlir_converter[op_proto[ATTRS][attr][TYPE]] in
                            ['BoolAttr', 'BoolArrayAttr']):
                            default_value = default_value.lower()
                        elif (attr_mlir_converter[op_proto[ATTRS][attr][TYPE]]
                              in ['StrAttr', 'StrArrayAttr']):
                            default_value = default_value.replace('\'', '\\\"')
                            if attr_mlir_converter[op_proto[ATTRS][attr][
                                    TYPE]] == "StrAttr":
                                default_value = '\\\"' + default_value + '\\\"'
                        attr_list = " DefaultValuedAttr<" + attr_mlir_converter[
                            op_proto[ATTRS][attr]
                            [TYPE]] + ", \"" + default_value + "\">:$" + attr + ","
                        ARGUMENTS += attr_list
                    else:
                        print("Error:" + op_type + ":" + attr + ":" + str(
                            op_proto[ATTRS][attr][TYPE]))
                else:
                    if op_proto[ATTRS][attr][TYPE] in attr_mlir_converter:
                        attr_type_ = attr_mlir_converter[op_proto[ATTRS][attr][
                            TYPE]]
                        if (attr_type_ in [
                                'StrAttr', 'I32ArrayAttr', 'F32ArrayAttr',
                                'StrArrayAttr', 'BoolArrayAttr', 'I64ArrayAttr'
                        ]):
                            attr_list = attr_type_ + ":$" + attr + ","
                            ARGUMENTS += attr_list
                    else:
                        print(" ouch Error:" + op_type + ":" + attr + ":" + str(
                            op_proto[ATTRS][attr][TYPE]))
            ARGUMENTS = ARGUMENTS[:-1] + ");\n"

        # 2.4 results info
        RESULTS = ""
        if (len(op_proto[OUTPUTS]) > 0):
            outputs = ""
            for output_ in op_proto[OUTPUTS]:
                if op_proto[OUTPUTS][output_][EXTRA] != True and op_proto[
                        OUTPUTS][output_][INTERMEDIATE] != True:
                    if op_proto[OUTPUTS][output_][DUPLICABLE] != True:
                        outputs = outputs + "PD_Tensor:${},".format(output_)
                    else:
                        outputs = outputs + "PD_Tensor_Array:${},".format(
                            output_)
            RESULTS = "\n  let results = (outs {});\n".format(outputs[:-1])

        with open(dst_dialect_file, 'a') as ops_mlir_file:
            ops_mlir_file.write(HEAD)
            ops_mlir_file.write(SUMMARY)
            ops_mlir_file.write(DESCRIPTION)
            ops_mlir_file.write(ARGUMENTS)
            ops_mlir_file.write(RESULTS)
            ops_mlir_file.write("}\n")

    print("Skipped ops num: " + str(len(skipped_op_list)))
    print("Automatically generated op dialects num: " + str(
        len(automatically_generated_op_dialect)))

    # 3. custom op dialect and end of file
    with open(dst_dialect_file, 'a') as ops_mlir_file:
        with open(custom_dialect_file, 'r') as custom_ops_file:
            custom_ops = custom_ops_file.readlines()
            ops_mlir_file.writelines(custom_ops)

        end_ = "\n#endif  // PD_OPS"
        ops_mlir_file.write(end_)


if __name__ == "__main__":
    all_op_protos_dict = get_all_ops_desc()
    generate_all_ops_inputs_outputs_map(all_op_protos_dict)
    convert_op_proto_into_mlir(all_op_protos_dict)
