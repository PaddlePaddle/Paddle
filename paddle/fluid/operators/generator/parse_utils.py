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

import re
from copy import copy
from typing import Dict, Any, List, Tuple
from tests import is_attr, is_input, is_output, is_vec


def to_named_dict(items: List[Dict]) -> Dict[str, Dict]:
    named_dict = {}
    for item in items:
        if "name" not in item:
            raise KeyError(f"name not in {item}")
        name = item["name"]
        named_dict[name] = item
    return named_dict


def parse_arg(op_name: str, s: str) -> Dict[str, str]:
    """parse an argument in following formats:
    1. typename name
    2. typename name = default_value
    """
    typename, rest = [item.strip() for item in s.split(" ", 1)]
    assert (
        len(typename) > 0
    ), f"The arg typename should not be empty. Please check the args of {op_name} in yaml."

    assert (
        rest.count("=") <= 1
    ), f"There is more than 1 = in an arg in {op_name}"
    if rest.count("=") == 1:
        name, default_value = [item.strip() for item in rest.split("=", 1)]
        assert (
            len(name) > 0
        ), f"The arg name should not be empty. Please check the args of {op_name} in yaml."
        assert (
            len(default_value) > 0
        ), f"The default value should not be empty. Please check the args of {op_name} in yaml."
        return {
            "typename": typename,
            "name": name,
            "default_value": default_value,
        }
    else:
        name = rest.strip()
        assert (
            len(name) > 0
        ), f"The arg name should not be empty. Please check the args of {op_name} in yaml."
        return {"typename": typename, "name": name}


def parse_input_and_attr(
    op_name: str, arguments: str
) -> Tuple[List, List, Dict, Dict]:
    args_str = arguments.strip()
    assert args_str.startswith('(') and args_str.endswith(')'), (
        f"Args declaration should start with '(' and end with ')', "
        f"please check the args of {op_name} in yaml."
    )
    args_str = args_str[1:-1]
    args = parse_plain_list(args_str)

    inputs = []
    attrs = []

    met_attr_with_default_value = False

    for arg in args:
        item = parse_arg(op_name, arg)
        typename = item["typename"]
        name = item["name"]
        if is_input(typename):
            assert len(attrs) == 0, (
                f"The input Tensor should appear before attributes. "
                f"please check the position of {op_name}:input({name}) "
                f"in yaml."
            )
            inputs.append(item)
        elif is_attr(typename):
            if met_attr_with_default_value:
                assert (
                    "default_value" in item
                ), f"{op_name}: Arguments with default value should not precede those without default value"
            elif "default_value" in item:
                met_attr_with_default_value = True
            attrs.append(item)
        else:
            raise KeyError(f"{op_name}: Invalid argument type {typename}.")
    return inputs, attrs


def parse_output(op_name: str, s: str) -> Dict[str, str]:
    """parse an output, typename or typename(name)."""
    match = re.search(
        r"(?P<out_type>[a-zA-Z0-9_[\]]+)\s*(?P<name>\([a-zA-Z0-9_@]+\))?\s*(?P<expr>\{[^\}]+\})?",
        s,
    )
    typename = match.group("out_type")
    name = match.group("name")
    size_expr = match.group("expr")

    name = name[1:-1] if name is not None else 'out'
    size_expr = size_expr[1:-1] if size_expr is not None else None

    assert is_output(typename), (
        f"Invalid output type: {typename} in op : {op_name}."
        f"Supported types are Tensor and Tensor[]"
    )
    if size_expr is not None:
        assert is_vec(typename), (
            f"Invalid output size: output {name} in op : {op_name} is "
            f"not a vector but has size expr"
        )
        return {"typename": typename, "name": name, "size": size_expr}
    else:
        return {"typename": typename, "name": name}


def parse_outputs(op_name: str, outputs: str) -> List[Dict]:
    outputs = parse_plain_list(outputs, sep=",")
    output_items = []
    for output in outputs:
        output_items.append(parse_output(op_name, output))
    return output_items


def parse_infer_meta(infer_meta: Dict[str, Any]) -> Dict[str, Any]:
    infer_meta = copy(infer_meta)  # to prevent mutating the input
    if "param" not in infer_meta:
        infer_meta["param"] = None
    return infer_meta


def parse_candidates(s: str) -> Dict[str, Any]:
    "parse candidates joined by either '>'(ordered) or ','(unordered)"
    delimiter = ">" if ">" in s else ","
    ordered = delimiter == ">"
    candidates = parse_plain_list(s, delimiter)
    return {"ordered": ordered, "candidates": candidates}


def parse_plain_list(s: str, sep=",") -> List[str]:
    items = [item.strip() for item in s.strip().split(sep)]
    return items


def parse_kernel(op_name: str, kernel_config: Dict[str, Any]) -> Dict[str, Any]:
    # kernel :
    #    func : [], Kernel functions (example: scale, scale_sr)
    #    param : [], Input params of kernel
    #    backend : str, the names of param to choose the kernel backend, default is None
    #    layout : str, the names of param to choose the kernel layout, default is None
    #    data_type : str, the names of param to choose the kernel data_type, default is None
    #    dispatch : {}, the key is kernel_func, the value is type of inputs and outputs for kernel (example: {kernel_name : (['dense','sparse_coo']#input,['sparse_coo']#output)})
    kernel = {
        'func': [],  # up to 2 function names
        'param': None,
        'backend': None,
        'layout': None,
        'data_type': None,
        'dispatch': {},
    }
    if 'param' in kernel_config:
        kernel['param'] = kernel_config['param']

    if 'backend' in kernel_config:
        kernel['backend'] = parse_candidates(kernel_config["backend"])

    if 'layout' in kernel_config:
        kernel['layout'] = parse_candidates(kernel_config["layout"])

    if 'data_type' in kernel_config:
        kernel['data_type'] = parse_candidates(kernel_config["data_type"])

    kernel_funcs = re.compile(r'([a-zA-Z0-9_]+)\s*({[^}]+})?').findall(
        kernel_config['func']
    )

    def parse_kernel_in_out_type(in_out_str):
        if len(in_out_str) == 0:
            return None
        tmp_in_out_list = in_out_str[1:-1].split('->')
        inputs = [item.strip() for item in tmp_in_out_list[0].split(',')]
        outputs = [item.strip() for item in tmp_in_out_list[1].split(',')]

        # check the tensor type
        for item in inputs:
            assert item in [
                'dense',
                'selected_rows',
                'sparse_coo',
                'sparse_csr',
            ], f"{op_name} : Invalid input tensor type ('{item}'), here we only support 'dense', 'selected_rows', 'sparse_coo' and 'sparse_csr'."
        for item in outputs:
            assert item in [
                'dense',
                'selected_rows',
                'sparse_coo',
                'sparse_csr',
            ], f"{op_name} : Invalid output tensor type ('{item}'), here we only support 'dense', 'selected_rows', 'sparse_coo' and 'sparse_csr'."

        return (inputs, outputs)

    for func_item in kernel_funcs:
        kernel['func'].append(func_item[0])
        kernel['dispatch'][func_item[0]] = parse_kernel_in_out_type(
            func_item[1]
        )

    return kernel


def parse_inplace(op_name: str, inplace_cfg: str) -> Dict[str, str]:
    inplace_map = {}
    inplace_cfg = inplace_cfg.lstrip("(").rstrip(")")
    pairs = parse_plain_list(inplace_cfg)
    for pair in pairs:
        in_name, out_name = parse_plain_list(pair, sep="->")
        inplace_map[out_name] = in_name
    return inplace_map


def parse_invoke(op_name: str, invoke_config: str) -> Dict[str, Any]:
    invoke_config = invoke_config.strip()
    func, rest = invoke_config.split("(", 1)
    func = func.strip()
    args = rest.rstrip(")").strip()
    invocation = {"func": func, "args": args}
    return invocation


def extract_type_and_name(records: List[Dict]) -> List[Dict]:
    """extract type and name from forward call, it is simpler than forward op ."""
    extracted = [
        {"name": item["name"], "typename": item["typename"]} for item in records
    ]
    return extracted


def parse_forward(op_name: str, forward_config: str) -> Dict[str, Any]:
    # op_name (const Tensor& input, ... , int attr, ...) -> Tensor(out)
    result = re.search(
        r"(?P<op>[a-z][a-z0-9_]+)\s*(?P<args>\([^\)]+\))\s*->\s*(?P<outputs>.+)",
        forward_config,
    )
    op = result.group("op")
    outputs = parse_outputs(op_name, result.group("outputs"))
    outputs = extract_type_and_name(outputs)

    inputs, attrs = parse_input_and_attr(op_name, result.group("args"))
    inputs = extract_type_and_name(inputs)
    attrs = extract_type_and_name(attrs)
    forward_cfg = {
        "name": op,
        "inputs": inputs,
        "attrs": attrs,
        "outputs": outputs,
    }
    return forward_cfg


def parse_op_entry(op_entry: Dict[str, Any], name_field="op"):
    op_name = op_entry[name_field]
    inputs, attrs = parse_input_and_attr(op_name, op_entry["args"])
    outputs = parse_outputs(op_name, op_entry["output"])

    # validate default value of DataType and DataLayout
    for attr in attrs:
        if "default_value" in attr:
            typename = attr["typename"]
            default_value = attr["default_value"]
            if typename == "DataType":
                assert (
                    "DataType" in default_value
                ), f"invalid DataType default value in {op_name}"
                # remove namespace
                default_value = default_value[default_value.find("DataType") :]
                attr["default_value"] = default_value
            elif typename == "DataLayout":
                assert (
                    "DataLayout" in default_value
                ), f"invalid DataLayout default value in {op_name}"
                default_value = default_value[
                    default_value.find("DataLayout") :
                ]
                attr["default_value"] = default_value

    input_names = [item["name"] for item in inputs]
    attr_names = [item["name"] for item in attrs]
    output_names = [item["name"] for item in outputs]

    # add optional tag for every input
    for input in inputs:
        input["optional"] = False
    if "optional" in op_entry:
        optional_args = parse_plain_list(op_entry["optional"])
        for name in optional_args:
            assert (
                name in input_names
            ), f"{op_name} has an optional input: '{name}' which is not an input."
        for input in inputs:
            if input["name"] in optional_args:
                input["optional"] = True

    # add intermediate tag for every output
    for output in outputs:
        output["intermediate"] = False
    if "intermediate" in op_entry:
        intermediate_outs = parse_plain_list(op_entry["intermediate"])
        for name in intermediate_outs:
            assert (
                name in output_names
            ), f"{op_name} has an intermediate output: '{name}' which is not an output."
        for output in outputs:
            if output["name"] in intermediate_outs:
                output["intermediate"] = True

    # add no_need_buffer for every input
    for input in inputs:
        input["no_need_buffer"] = False
    if "no_need_buffer" in op_entry:
        no_buffer_args = parse_plain_list(op_entry["no_need_buffer"])
        for name in no_buffer_args:
            assert (
                name in input_names
            ), f"{op_name} has an no buffer input: '{name}' which is not an input."
        for input in inputs:
            if input["name"] in no_buffer_args:
                input["no_need_buffer"] = True
    else:
        no_buffer_args = None

    # TODO(chenfeiyu): data_transform

    op = {
        "name": op_name,
        "inputs": inputs,
        "attrs": attrs,
        "outputs": outputs,
        "no_need_buffer": no_buffer_args,
    }

    # invokes another op ?
    is_base_op = "invoke" not in op_entry

    if is_base_op:
        # kernel
        kernel = parse_kernel(op_name, op_entry["kernel"])
        if kernel["param"] is None:
            kernel["param"] = input_names + attr_names

        # infer meta
        infer_meta = parse_infer_meta(op_entry["infer_meta"])
        if infer_meta["param"] is None:
            infer_meta["param"] = copy(kernel["param"])

        # inplace
        if "inplace" in op_entry:
            inplace_pairs = parse_inplace(op_name, op_entry["inplace"])
        else:
            inplace_pairs = None
        op.update(
            {
                "infer_meta": infer_meta,
                "kernel": kernel,
                "inplace": inplace_pairs,
            }
        )
    else:
        # invoke
        invoke = parse_invoke(op_name, op_entry["invoke"])
        op["invoke"] = invoke

    # backward
    if "backward" in op_entry:
        backward = op_entry["backward"]
    else:
        backward = None
    op["backward"] = backward

    # forward for backward_ops
    is_backward_op = name_field == "backward_op"
    if is_backward_op:
        if "forward" in op_entry:
            forward = parse_forward(op_name, op_entry["forward"])
            # validate_fb
            validate_backward_inputs(
                op_name, forward["inputs"], forward["outputs"], inputs
            )
            validate_backward_attrs(op_name, forward["attrs"], attrs)
            validate_backward_outputs(op_name, forward["inputs"], outputs)
        else:
            forward = None
        op["forward"] = forward
    return op


def validate_backward_attrs(op, forward_attrs, backward_attrs):
    if len(forward_attrs) >= len(backward_attrs):
        return
    num_exceptional_attrs = len(backward_attrs) - len(forward_attrs)
    # this is a not-that-clean trick to allow backward op to has more attrs
    # than the forward op , as long as they all have default value
    for i in range(-num_exceptional_attrs, 0):
        assert (
            "default_value" in backward_attrs[i]
        ), f"{op } has exceptional attr without default value"


def validate_backward_inputs(
    op, forward_inputs, forward_outputs, backward_inputs
):
    foward_input_names = [item["name"] for item in forward_inputs]
    forward_output_names = [item["name"] for item in forward_outputs]
    backward_input_names = [item["name"] for item in backward_inputs]

    assert len(backward_input_names) <= len(foward_input_names) + 2 * len(
        forward_output_names
    ), f"{op } has too many inputs."


def validate_backward_outputs(op, forward_inputs, backward_outputs):
    assert len(backward_outputs) <= len(
        forward_inputs
    ), f"{op } has too many outputs"


def cross_validate(ops):
    for name, op in ops.items():
        if "forward" in op:
            fw_call = op["forward"]
            fw_name = fw_call["name"]
            if fw_name not in ops:
                print(
                    f"Something Wrong here, this backward op ({name})'s forward op ({fw_name}) does not exist."
                )
            else:
                fw_op = ops[fw_name]
                if "backward" not in fw_op or fw_op["backward"] is None:
                    print(
                        f"Something Wrong here, {name}'s forward op ({fw_name}) does not claim {name} as its backward."
                    )
                else:
                    assert (
                        fw_op["backward"] == name
                    ), f"{name}: backward and forward name mismatch"

                assert len(fw_call["inputs"]) <= len(
                    fw_op["inputs"]
                ), f"{name}: forward call has more inputs than the op "
                for (input, input_) in zip(fw_call["inputs"], fw_op["inputs"]):
                    assert (
                        input["typename"] == input_["typename"]
                    ), f"type mismatch in {name} and {fw_name}"

                assert len(fw_call["attrs"]) <= len(
                    fw_op["attrs"]
                ), f"{name}: forward call has more attrs than the op "
                for (attr, attr_) in zip(fw_call["attrs"], fw_op["attrs"]):
                    if attr["typename"] == "Scalar":
                        # special case for Scalar, fw_call can omit the type
                        assert re.match(
                            r"Scalar(\(\w+\))*", attr_["typename"]
                        ), f"type mismatch in {name} and {fw_name}"
                    else:
                        assert (
                            attr["typename"] == attr_["typename"]
                        ), f"type mismatch in {name} and {fw_name}"

                assert len(fw_call["outputs"]) == len(
                    fw_op["outputs"]
                ), f"{name}: forward call has more outputs than the op "
                for (output, output_) in zip(
                    fw_call["outputs"], fw_op["outputs"]
                ):
                    assert (
                        output["typename"] == output_["typename"]
                    ), f"type mismatch in {name} and {fw_name}"
