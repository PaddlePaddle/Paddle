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

import itertools
import re

from type_mapping import (
    attr_types_map,
    dense_input_types_map,
    dense_optional_input_types_map,
    dense_output_types_map,
    input_types_map,
    opmaker_attr_types_map,
    optional_input_types_map,
    output_type_map,
    phi_attr_types_map,
    sr_output_types_map,
)


def quote(s):
    return '"{}"'.format(s)


# ------------------------------ attr -------------------------------------
def to_phi_attr_type(s):
    return phi_attr_types_map[s]


def to_op_attr_type(s):
    return opmaker_attr_types_map[s]


def to_paddle_attr_type(s):
    "Convert type tag for attributes in yaml to c++ types"
    return attr_types_map[s]


# ------------------------------ input ----------------------------------
def to_paddle_input_type(s, optional=False):
    "Convert type tag for inputs in yaml to c++ types"
    if optional:
        return optional_input_types_map[s]
    else:
        return input_types_map[s]


def to_dense_input_type(s, optional=False):
    "Convert types in yaml to dense tensor type in phi"
    if optional:
        return dense_input_types_map[s]
    else:
        return dense_optional_input_types_map[s]


# ------------------------------ output  ----------------------------------
def to_paddle_output_type(s):
    return output_type_map[s]


def to_dense_output_type(s):
    "Convert types in yaml to dense tensor type in phi"
    return dense_output_types_map[s]


def to_sr_output_type(s):
    "Convert types in yaml to selected rows type in phi"
    return sr_output_types_map[s]


# -------------- transform argument names from yaml to opmaker ------------
def to_opmaker_name(s):
    if s.endswith("_grad"):
        return 'GradVarName("{}")'.format(s[:-5])
    else:
        return '"{}"'.format(s)


def to_opmaker_name_cstr(s):
    if s.endswith("_grad"):
        return '"{}@GRAD"'.format(s[:-5])
    else:
        return '"{}"'.format(s)


def to_pascal_case(s):
    words = s.split("_")
    return "".join([word.capitalize() for word in words])


def to_input_name(s):
    """find input variable name in op yaml for higher order backward op .
    x -> dx
    x -> d2x
    x -> d3x

    NOTE: for first order backward op
    x -> x_grad
    is more common.
    """
    match = re.match(r"(d\d*)(\w+)", s)
    assert match.group(1) != "", "it should be a grad style name."
    return match.group(2)


def to_scalar_tensor_name(attr):
    if 'tensor_name' in attr:
        return attr['tensor_name']
    return to_pascal_case(attr['name']) + 'Tensor'


def to_int_array_tensor_name(attr):
    if 'tensor_name' in attr:
        return attr['tensor_name']
    return to_pascal_case(attr['name']) + 'Tensor'


def to_int_array_tensors_name(attr):
    if 'tensors_name' in attr:
        return attr['tensors_name']
    return to_pascal_case(attr['name']) + 'TensorList'


def cartesian_prod_attrs(attrs):
    items = []
    for attr in attrs:
        type_name = attr["typename"]
        name = attr["name"]
        if type_name == "Scalar":
            items.append((name, to_scalar_tensor_name(attr)))
        elif type_name == "IntArray":
            if 'tensor_name' not in attr and 'manual_flag' in attr:
                items.append((name, to_int_array_tensors_name(attr)))
            elif 'tensors_name' not in attr and 'manual_flag' in attr:
                items.append((name, to_int_array_tensor_name(attr)))
            else:
                items.append(
                    (
                        name,
                        to_int_array_tensor_name(attr),
                        to_int_array_tensors_name(attr),
                    )
                )
        else:
            items.append((name,))

    _combinations = itertools.product(*items)
    combinations = []
    for x in _combinations:
        combinations.append('{' + ", ".join(quote(t) for t in x) + '}')
    return combinations


def cartesian_prod_mapping(op):
    kernels = op["kernel"]["func"]
    inputs = [
        x["name"] for x in op["inputs"] if x["name"] in op["kernel"]["param"]
    ]
    inputs = [to_opmaker_name_cstr(input) for input in inputs]
    attrs = cartesian_prod_attrs(op["attrs"])
    outputs = [to_opmaker_name_cstr(output["name"]) for output in op["outputs"]]

    def vec(items):
        return "{" + ', '.join(items) + "}"

    inputs = [vec(inputs)]
    outputs = [vec(outputs)]
    kernels = [quote(x) for x in kernels]
    mappings = itertools.product(kernels, inputs, attrs, outputs)

    outs = []
    for spec in mappings:
        outs.append("return KernelSignature({});".format(", ".join(spec)))
    return "\n".join(outs)
