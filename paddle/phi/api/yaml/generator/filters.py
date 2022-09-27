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

from typing import List, Dict
import itertools
import re

from jinja2.filters import do_xmlattr
from type_mapping import (input_types_map, optional_input_types_map,
                          attr_types_map, opmaker_attr_types_map,
                          output_type_map)
from type_mapping import (dense_input_types_map, dense_optional_input_types_map,
                          dense_output_types_map, sr_input_types_map,
                          sr_optional_input_types_map, sr_output_types_map,
                          phi_attr_types_map)


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
    """find input variable name in api yaml for higher order backward api.
    x -> dx
    x -> d2x
    x -> d3x

    NOTE: for first order backward api
    x -> x_grad
    is more common.
    """
    match = re.match(r"(d\d*)(\w+)", s)
    assert (match.group(1) != ""), "it should be a grad style name."
    return match.group(2)


def cartesian_prod_attrs(attrs):
    items = []
    for attr in attrs:
        type_name = attr["typename"]
        name = attr["name"]
        if type_name == "Scalar":
            items.append((name, "{}Tensor".format(name)))
        elif type_name == "IntArray":
            items.append(
                (name, "{}Tensor".format(name), "{}TensorList".format(name)))
        else:
            items.append((name, ))

    _combinations = itertools.product(*items)
    combinations = []
    for x in _combinations:
        combinations.append('{' + ", ".join(quote(t) for t in x) + '}')
    return combinations


def cartesian_prod_mapping(api):
    kernels = api["kernel"]["func"]
    inputs = [
        x["name"] for x in api["inputs"] if x["name"] in api["kernel"]["param"]
    ]
    inputs = [to_opmaker_name_cstr(input) for input in inputs]
    attrs = cartesian_prod_attrs(api["attrs"])
    outputs = [
        to_opmaker_name_cstr(output["name"]) for output in api["outputs"]
    ]

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
