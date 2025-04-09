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
from __future__ import annotations

import itertools
import re
from typing import TYPE_CHECKING

from type_mapping import (
    attr_types_map,
    dense_input_types_map,
    dense_optional_input_types_map,
    dense_output_types_map,
    input_types_map,
    opmaker_attr_types_map,
    optional_input_types_map,
    optional_output_type_map,
    output_type_map,
    phi_attr_types_map,
    sr_output_types_map,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def get_infer_var_type_func(op_name):
    if op_name == "assign":
        return f"""
class {to_pascal_case(op_name)}InferVarType : public framework::VarTypeInference {{
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {{
    ctx->SyncTypeAndDataType("X", "Out");
  }}
}};
"""
    elif op_name == "lookup_table_v2_grad":
        return f"""
class {to_pascal_case(op_name)}InferVarType : public framework::VarTypeInference {{
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {{
    auto out_var_name = framework::GradVarName("W");
    auto attr = ctx->GetAttr("is_sparse");
    bool is_sparse = PADDLE_GET(bool, attr);
    if (is_sparse) {{
      VLOG(3) << "lookup_table_v2_grad op " << framework::GradVarName("W")
              << " is set to SelectedRows";
      ctx->SetOutputType(out_var_name,
                         framework::proto::VarType::SELECTED_ROWS);
    }} else {{
      VLOG(3) << "lookup_table_v2_grad op " << framework::GradVarName("W")
              << " is set to phi::DenseTensor";
      ctx->SetOutputType(out_var_name, framework::proto::VarType::DENSE_TENSOR);
    }}
    ctx->SetOutputDataType(out_var_name, ctx->GetInputDataType("W"));
  }}
}};
"""
    elif op_name == "merge_selected_rows":
        return f"""
class {to_pascal_case(op_name)}InferVarType : public framework::PassInDtypeAndVarTypeToOutput {{
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType() const override {{
      static std::unordered_map<std::string, std::string> m{{{{"X", /*->*/ "Out"}}}};
      return m;
  }}
}};
"""
    elif op_name == "strided_slice":
        return f"""
class {to_pascal_case(op_name)}InferVarType : public framework::VarTypeInference {{
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {{
    ctx->SetOutputType("Out", ctx->GetInputType("Input"));
    ctx->SetOutputDataType("Out", ctx->GetInputDataType("Input"));
  }}
}};
"""
    elif op_name == "strided_slice_grad":
        return f"""
class {to_pascal_case(op_name)}InferVarType : public framework::VarTypeInference {{
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {{
    ctx->SetOutputType(framework::GradVarName("Input"),
                       ctx->GetInputType(framework::GradVarName("Out")));
    ctx->SetOutputDataType(
        framework::GradVarName("Input"),
        ctx->GetInputDataType(framework::GradVarName("Out")));
  }}
}};
"""
    else:
        return None


def quote(s):
    return f'"{s}"'


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


def assert_dense_or_sr(input_type):
    return (
        "ctx.IsSelectedRowsInput"
        if input_type == "selected_rows"
        else "ctx.IsDenseTensorInput"
    )


def find_optional_inputs_name(inputs):
    optional_inputs_name = [
        input["fluid_name"] for input in inputs if input["optional"] is True
    ]
    return optional_inputs_name


def delete_last_underline(op_name):
    return op_name if op_name[-1] != '_' else op_name[:-1]


# ------------------------------ output  ----------------------------------
def to_paddle_output_type(s, optional=False):
    if optional:
        return optional_output_type_map[s]
    return output_type_map[s]


def to_dense_output_type(s):
    "Convert types in yaml to dense tensor type in phi"
    return dense_output_types_map[s]


def to_sr_output_type(s):
    "Convert types in yaml to selected rows type in phi"
    return sr_output_types_map[s]


def filter_intermediate(items: Sequence):
    return tuple([item for item in items if not item.get('intermediate')])


# -------------- transform argument names from yaml to opmaker ------------
def to_opmaker_name(s):
    if s.endswith("_grad"):
        return f'GradVarName("{s[:-5]}")'
    else:
        return f'"{s}"'


def to_opmaker_name_cstr(s):
    if s.endswith("_grad"):
        return f'"{s[:-5]}@GRAD"'
    else:
        return f'"{s}"'


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


def to_composite_grad_opmaker_name(backward_op_name):
    words = backward_op_name.split("_")
    for i in range(len(words)):
        words[i] = words[i].strip()
        words[i] = words[i].capitalize()
    composite_grad_opmaker_name = "".join(word for word in words[:-1])
    composite_grad_opmaker_name += "CompositeGradOpMaker"
    return composite_grad_opmaker_name


def to_variable_names(dict_list: list[dict], key: str) -> list[str]:
    names = []
    for var in dict_list:
        names.append(var[key])
    return names


def cartesian_prod_attrs(attrs):
    items = []
    for attr in attrs:
        type_name = attr["typename"]
        name = attr["fluid_name"]
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
        x["fluid_name"]
        for x in op["inputs"]
        if x["fluid_name"] in op["kernel"]["param"]
    ]
    inputs = [to_opmaker_name_cstr(input) for input in inputs]
    attrs = cartesian_prod_attrs(op["attrs"])
    outputs = [
        to_opmaker_name_cstr(output["fluid_name"]) for output in op["outputs"]
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
