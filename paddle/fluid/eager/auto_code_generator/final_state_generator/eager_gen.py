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

import yaml
import re
import argparse
import os

# For API dispatch used at python-level
# { op_name : [arg_name, ...] }
core_ops_returns_info = {}
core_ops_args_info = {}
core_ops_args_type_info = {}


def ParseArguments():
    parser = argparse.ArgumentParser(
        description='Eager Code Generator Args Parser')
    parser.add_argument('--nodes_h_path', type=str)
    parser.add_argument('--nodes_cc_path', type=str)
    parser.add_argument('--forwards_h_path', type=str)
    parser.add_argument('--forwards_cc_path', type=str)
    parser.add_argument('--api_yaml_path', type=str)
    parser.add_argument('--backward_yaml_path', type=str)

    args = parser.parse_args()
    return args


#################
###  Helpers  ###
#################
def FindGradName(string):
    return string + "_grad"


def FindForwardName(string):
    if not string.endswith("_grad"):
        return None
    return string[:-5]


def IsPlainTensorType(string):
    plain_tensor_types = ['Tensor&', 'Tensor', 'const Tensor&', 'const Tensor']
    if string in plain_tensor_types:
        return True
    return False


def IsVectorTensorType(string):
    vector_tensor_types = ['list(Tensor)']
    if string in vector_tensor_types:
        return True
    return False


def GetSavedName(string):
    return string + "_"


def GetConstReference(string):
    ret = string
    if not string.startswith("const "):
        ret = "const " + string
    if not string.endswith("&"):
        ret += "&"
    return ret


def RemoveConstAndReference(string):
    ret = string
    if string.startswith("const "):
        ret = ret[6:]
    if string.endswith("&"):
        ret = ret[:-1]

    return ret


def GetGradNodeName(string):
    return f"FinalGradNode{string}"


def GetForwardFunctionName(string):
    return f"{string}_final_state_dygraph_function"


def GetAutoGradMetaName(string):
    return f"{string}_autograd_meta"


def GetAutoGradMetaVectorName(string):
    return f"{string}_autograd_meta_vec"


######################
###  File Readers  ###
######################
def ReadFwdFile(filepath):
    f = open(filepath, 'r')
    contents = yaml.load(f, Loader=yaml.FullLoader)
    return contents


def ReadBwdFile(filepath):
    f = open(filepath, 'r')
    contents = yaml.load(f, Loader=yaml.FullLoader)
    ret = {}
    for content in contents:
        assert 'backward_api' in content.keys()
        api_name = content['backward_api']
        ret[api_name] = content
    return ret


######################
###  Yaml Parsers  ###
######################
def ParseYamlArgs(string):
    # Example: const Tensor& x, const Tensor& y, bool transpose_x, bool transpose_y

    # inputs_list = [ [arg_name, arg_type, orig_position], ...]
    inputs_list = []
    # attrs_list = [ [arg_name, arg_type, default_value, orig_position], ...]
    attrs_list = []

    args = [x.strip() for x in string.strip().split(",")]
    atype = r'((const )?\S+) '
    aname = r'(.*)'
    pattern = f'{atype}{aname}'
    for i in range(len(args)):
        arg = args[i]
        m = re.search(pattern, arg)
        arg_type = m.group(1).strip()
        arg_name = m.group(3).split("=")[0].strip()
        default_value = m.group(3).split("=")[1].strip() if len(
            m.group(3).split("=")) > 1 else None
        if "Tensor" in arg_type:
            assert default_value is None
            inputs_list.append([arg_name, arg_type, i])
        else:
            attrs_list.append([arg_name, arg_type, default_value, i])

    return inputs_list, attrs_list


def ParseYamlReturns(string):
    # Example: Tensor, Tensor

    # list = [ ["", ret_type, orig_position], ...]
    returns_list = []

    returns = [x.strip() for x in string.strip().split(",")]
    for i in range(len(returns)):
        ret = returns[i]
        returns_list.append(["", ret, i])

    return returns_list


def ParseYamlReturnsWithName(string):
    # Example: Tensor(out), Tensor(out1)

    # list = [ [ret_name, ret_type, orig_position], ...]
    returns_list = []

    returns = [x.strip() for x in string.strip().split(",")]

    atype = r'(.*?)'
    aname = r'(.*?)'
    pattern = f'{atype}\({aname}\)'
    for i in range(len(returns)):
        ret = returns[i]
        m = re.search(pattern, ret)
        ret_type = m.group(1)
        ret_name = m.group(2)
        assert "Tensor" in ret_type
        returns_list.append([ret_name, ret_type, i])

    return returns_list


def ParseYamlForwardFromBackward(string):
    # Example: matmul (const Tensor& x, const Tensor& y, bool transpose_x, bool transpose_y) -> Tensor(out)

    fname = r'(.*?)'
    wspace = r'\s*'
    fargs = r'(.*?)'
    frets = r'(.*)'
    pattern = f'{fname}{wspace}\({wspace}{fargs}{wspace}\){wspace}->{wspace}{frets}'

    m = re.search(pattern, string)
    function_name = m.group(1)
    function_args = m.group(2)
    function_returns = m.group(3)

    forward_inputs_list, forward_attrs_list = ParseYamlArgs(function_args)
    forward_returns_list = ParseYamlReturnsWithName(function_returns)

    return forward_inputs_list, forward_attrs_list, forward_returns_list


def ParseYamlForward(args_str, returns_str):
    # args Example: (const Tensor& x, const Tensor& y, bool transpose_x = false, bool transpose_y = false)
    # returns Example: Tensor, Tensor

    fargs = r'(.*?)'
    wspace = r'\s*'
    args_pattern = f'\({fargs}\)'
    args_str = re.search(args_pattern, args_str).group(1)

    inputs_list, attrs_list = ParseYamlArgs(args_str)
    returns_list = ParseYamlReturns(returns_str)

    return inputs_list, attrs_list, returns_list


def ParseYamlBackward(args_str, returns_str):
    # args Example: (const Tensor& x, const Tensor& y, const Tensor& out_grad, bool transpose_x=false, bool transpose_y=false)
    # returns Example: Tensor(x_grad), Tensor(y_grad)

    fargs = r'(.*?)'
    wspace = r'\s*'
    args_pattern = f'\({fargs}\)'
    args_str = re.search(args_pattern, args_str).group(1)

    inputs_list, attrs_list = ParseYamlArgs(args_str)
    returns_list = ParseYamlReturnsWithName(returns_str)

    return inputs_list, attrs_list, returns_list


#######################
###  Preprocessing  ###
#######################
def ForwardsValidationCheck(forward_inputs_list, forward_attrs_list,
                            forward_returns_list, orig_forward_inputs_list,
                            orig_forward_attrs_list, orig_forward_returns_list):
    for i in range(len(forward_inputs_list)):
        forward_input_name = forward_inputs_list[i][0]
        forward_input_type = forward_inputs_list[i][1]
        forward_input_pos = forward_inputs_list[i][2]
        orig_input_name = orig_forward_inputs_list[i][0]
        orig_input_type = orig_forward_inputs_list[i][1]
        orig_input_pos = orig_forward_inputs_list[i][2]

        assert forward_input_type == orig_input_type
        assert forward_input_pos == orig_input_pos

    for i in range(len(forward_attrs_list)):
        orig_attr_name = orig_forward_attrs_list[i][0]
        orig_attr_type = orig_forward_attrs_list[i][1]
        orig_attr_default = orig_forward_attrs_list[i][2]
        orig_attr_pos = orig_forward_attrs_list[i][3]
        forward_attr_name = forward_attrs_list[i][0]
        forward_attr_type = forward_attrs_list[i][1]
        forward_attr_default = forward_attrs_list[i][2]
        forward_attr_pos = forward_attrs_list[i][3]
        assert orig_attr_type == forward_attr_type
        assert orig_attr_default == forward_attr_default
        assert orig_attr_pos == forward_attr_pos

    for i in range(len(forward_returns_list)):
        orig_return_type = orig_forward_returns_list[i][1]
        orig_return_pos = orig_forward_returns_list[i][2]
        forward_return_type = forward_returns_list[i][1]
        forward_return_pos = forward_returns_list[i][2]

        assert orig_return_type == forward_return_type
        assert orig_return_pos == forward_return_pos

    # Check Order: Inputs, Attributes
    max_input_position = -1
    for _, _, pos in forward_inputs_list:
        max_input_position = max(max_input_position, pos)

    max_attr_position = -1
    for _, _, _, pos in forward_attrs_list:
        assert pos > max_input_position
        max_attr_position = max(max_attr_position, pos)


def BackwardValidationCheck(backward_fwd_input_map, backward_grad_input_map,
                            backward_attrs_list):

    # Check Order: TensorWrappers, GradTensors, Attributes
    max_fwd_input_position = -1
    for _, (_, _, pos) in backward_fwd_input_map.items():
        max_fwd_input_position = max(max_fwd_input_position, pos)

    max_grad_tensor_position = -1
    for _, (_, _, pos) in backward_grad_input_map.items():
        assert pos > max_fwd_input_position
        max_grad_tensor_position = max(max_grad_tensor_position, pos)

    max_attr_position = -1
    for _, _, _, pos in backward_attrs_list:
        assert pos > max_grad_tensor_position
        max_attr_position = max(max_attr_position, pos)


def DetermineForwardPositionMap(forward_inputs_list, forward_returns_list):
    forward_inputs_position_map = {}
    forward_outputs_position_map = {}
    for i in range(len(forward_inputs_list)):
        forward_input = forward_inputs_list[i]
        input_name = forward_input[0]
        input_type = forward_input[1]
        input_pos = forward_input[2]

        forward_inputs_position_map[input_name] = [input_type, input_pos]

    for i in range(len(forward_returns_list)):
        forward_return = forward_returns_list[i]
        return_name = forward_return[0]
        return_type = forward_return[1]
        return_pos = forward_return[2]

        forward_outputs_position_map[return_name] = [return_type, return_pos]

    return forward_inputs_position_map, forward_outputs_position_map


def SlotNameMatching(backward_inputs_list, backward_returns_list,
                     forward_inputs_position_map, forward_outputs_position_map):

    backward_fwd_input_map = {}
    backward_grad_input_map = {}
    backward_grad_output_map = {}

    for backward_input in backward_inputs_list:
        backward_input_name = backward_input[0]
        backward_input_type = backward_input[1]
        backward_input_pos = backward_input[2]

        backward_fwd_name = FindForwardName(backward_input_name)
        if backward_fwd_name:
            # Grad Input
            assert backward_fwd_name in forward_outputs_position_map.keys()
            matched_forward_output_type = forward_outputs_position_map[
                backward_fwd_name][0]
            matched_forward_output_pos = forward_outputs_position_map[
                backward_fwd_name][1]

            backward_grad_input_map[backward_input_name] = [
                backward_input_type, matched_forward_output_pos,
                backward_input_pos
            ]
        else:
            # TensorWrapper Input
            if backward_input_name in forward_inputs_position_map.keys():
                tensor_wrapper_type = forward_inputs_position_map[
                    backward_input_name][0]
                backward_fwd_input_map[backward_input_name] = [
                    backward_input_type, True, backward_input_pos
                ]

            elif backward_input_name in forward_outputs_position_map.keys():
                tensor_wrapper_type = forward_outputs_position_map[
                    backward_input_name][0]
                backward_fwd_input_map[backward_input_name] = [
                    backward_input_type, False, backward_input_pos
                ]
            else:
                assert False

    for backward_output in backward_returns_list:
        backward_output_name = backward_output[0]
        backward_output_type = backward_output[1]
        backward_output_pos = backward_output[2]

        backward_fwd_name = FindForwardName(backward_output_name)
        assert backward_fwd_name is not None
        assert backward_fwd_name in forward_inputs_position_map.keys()

        matched_forward_input_type = forward_inputs_position_map[
            backward_fwd_name][0]
        matched_forward_input_pos = forward_inputs_position_map[
            backward_fwd_name][1]

        backward_grad_output_map[backward_output_name] = [
            backward_output_type, matched_forward_input_pos, backward_output_pos
        ]

    return backward_fwd_input_map, backward_grad_input_map, backward_grad_output_map


def GenerateNodeDeclaration(fwd_api_name, backward_fwd_input_map,
                            backward_attrs_list):
    # Inputs:
    # fwd_api_name = ""
    # backward_fwd_input_map   = { "name" : [type, is_fwd_input, orig_position] ...}
    # backward_attrs_list = [ [attr_name, attr_type, default_value, orig_position], ...]

    # Determine Node Name
    forward_op_name = fwd_api_name

    # SetTensorWrapper Methods & TensorWrapper Members
    set_tensor_wrapper_methods_str = ""
    tensor_wrapper_members_str = ""
    for tname, (ttype, is_fwd_input, _) in backward_fwd_input_map.items():
        tensor_wrapper_name = GetSavedName(tname)
        if IsPlainTensorType(ttype):
            SET_PLAIN_TENSOR_WRAPPER_TEMPLATE = """
   void SetTensorWrapper{}(const paddle::experimental::Tensor& {}, bool full_reserved) {{     
     {} = egr::TensorWrapper({}, full_reserved);
   }}
"""
            set_tensor_wrapper_methods_str += SET_PLAIN_TENSOR_WRAPPER_TEMPLATE.format(
                tname, tname, tensor_wrapper_name, tname)

            PLAIN_TENSOR_MEMBER_TEMPLATE = """
   egr::TensorWrapper {};
"""
            tensor_wrapper_members_str += PLAIN_TENSOR_MEMBER_TEMPLATE.format(
                tensor_wrapper_name)
        else:
            assert IsVectorTensorType(ttype)
            SET_VECTOR_TENSOR_WRAPPER_TEMPLATE = """
   void SetTensorWrapper{}(const std::vector<paddle::experimental::Tensor>& {}, bool full_reserved) {{
     for(const auto& eager_tensor : {}) {{
        {}.emplace_back( egr::TensorWrapper(eager_tensor, full_reserved) );
     }};
   }}
"""
            set_tensor_wrapper_methods_str += SET_VECTOR_TENSOR_WRAPPER_TEMPLATE.format(
                tname, tname, tname, tensor_wrapper_name)

            VECTOR_TENSOR_MEMBER_TEMPLATE = """
   std::vector<egr::TensorWrapper> {};
"""
            tensor_wrapper_members_str += VECTOR_TENSOR_MEMBER_TEMPLATE.format(
                tensor_wrapper_name)
    # End: SetTensorWrapper Methods & TensorWrapper Members

    # SetAttributes & Attribute Members
    set_attribute_methods_str = ""
    attribute_members_str = ""
    for aname, atype, default_val, _ in backward_attrs_list:
        saved_attr_name = GetSavedName(aname)
        SET_ATTR_METHOD_TEMPLATE = """
   void SetAttribute{}({} {}) {{     
     {} = {};
   }}
"""
        set_attribute_methods_str += SET_ATTR_METHOD_TEMPLATE.format(
            aname, GetConstReference(atype), aname, saved_attr_name, aname)

        ATTRIBUTE_MEMBER_TEMPLATE = """
   {} {} = {};
"""
        attribute_members_str += ATTRIBUTE_MEMBER_TEMPLATE.format(
            RemoveConstAndReference(atype), saved_attr_name, default_val)
    # End: SetAttributes & Attribute Members

    grad_node_name = GetGradNodeName(fwd_api_name)
    NODE_DECLARATION_TEMPLATE = """
class {} : public egr::GradNodeBase {{
 public:
  {}() : egr::GradNodeBase() {{}}
  {}(size_t bwd_in_slot_num, size_t bwd_out_slot_num) : 
      egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {{}}
  ~{}() override = default;

  virtual std::vector<std::vector<paddle::experimental::Tensor>> operator()(
      const std::vector<std::vector<paddle::experimental::Tensor>>& grads) override;
  
  // SetTensorWrapperX, SetTensorWrapperY, ...
  {}
  // SetAttributes
  {}
 private:
  // TensorWrappers
  {}

  // Attributes
  {}
}};
"""
    node_declaration_str = NODE_DECLARATION_TEMPLATE.format(
        grad_node_name, grad_node_name, grad_node_name, grad_node_name,
        set_tensor_wrapper_methods_str, set_attribute_methods_str,
        tensor_wrapper_members_str, attribute_members_str)

    return node_declaration_str


def GenerateNodeDefinition(fwd_api_name, bwd_api_name, backward_fwd_input_map,
                           backward_grad_input_map, backward_grad_output_map,
                           backward_attrs_list):
    # fwd_api_name = ""
    # backward_fwd_input_map   = { "name" : [type, is_fwd_input, orig_position] ...}
    # backward_grad_input_map  = { "name" : [type, fwd_position, orig_position] ...}
    # backward_grad_output_map = { "name" : [type, fwd_position, orig_position] ...}
    # backward_attrs_list = [ [attr_name, attr_type, default_value, orig_position], ...]

    # Construct grad_api function args
    # Order: TensorWrappers, GradTensors, Attributes
    grad_api_args_len = len(backward_fwd_input_map.keys()) + len(
        backward_grad_input_map.keys()) + len(backward_attrs_list)
    grad_api_args = ["" for i in range(grad_api_args_len)]
    for name, (_, is_fwd_input,
               grad_api_position), in backward_fwd_input_map.items():
        tensor_wrapper_name = GetSavedName(name)
        grad_api_args[
            grad_api_position] = f"egr::EagerUtils::RecoverTensorWrapper(&this->{tensor_wrapper_name}, nullptr)"

    for _, (ttype, fwd_position,
            grad_api_position) in backward_grad_input_map.items():
        if IsPlainTensorType(ttype):
            grad_api_args[grad_api_position] = f"grads[{fwd_position}][0]"
        else:
            assert IsVectorTensorType(ttype)
            grad_api_args[grad_api_position] = f"grads[{fwd_position}]"

    for name, _, _, grad_api_position in backward_attrs_list:
        saved_attribute_name = GetSavedName(name)
        grad_api_args[grad_api_position] = f"this->{saved_attribute_name}"
    grad_api_args_str = ", ".join(grad_api_args)

    # Construct grad_api returns
    num_bwd_outputs = len(backward_grad_output_map.keys())
    returns_str = f"std::vector<std::vector<paddle::experimental::Tensor>> returns({num_bwd_outputs});\n"
    for _, (ttype, fwd_position,
            grad_api_position) in backward_grad_output_map.items():
        # Infer Grad API Return Type
        if num_bwd_outputs == 1:
            # Single tensor output, return as is
            if IsPlainTensorType(ttype):
                returns_str += "returns[0] = { grad_api_returns };\n"
            else:
                assert IsVectorTensorType(ttype)
                returns_str += "returns[0] = grad_api_returns;\n"
        else:
            # Rearrange output order accordingly
            returns_str += f"returns[{fwd_position}] =  grad_api_returns[{grad_api_position}];\n"
    returns_str += f"return returns;\n"

    grad_node_name = GetGradNodeName(fwd_api_name)
    FUNCTION_TEMPLATE = """
std::vector<std::vector<paddle::experimental::Tensor>> {}::operator()(const std::vector<std::vector<paddle::experimental::Tensor>>& grads) {{
    // Call grad_api function
    auto grad_api_returns = paddle::experimental::{}({});
    {}
}}
  """

    node_definition_str = FUNCTION_TEMPLATE.format(
        grad_node_name, bwd_api_name, grad_api_args_str, returns_str)

    return node_definition_str


def GenerateNodeCreationCodes(fwd_api_name, bwd_api_name,
                              forward_inputs_position_map,
                              forward_outputs_position_map, forward_attrs_list,
                              backward_fwd_input_map, backward_grad_input_map,
                              backward_grad_output_map, backward_attrs_list):
    # fwd_api_name = ""
    # forward_inputs_position_map = { "name" : [type, fwd_position] }
    # forward_outputs_position_map = { "name" : [type, fwd_position] }
    # forward_attrs_list = [ [attr_name, attr_type, default_value, orig_position], ...]
    # backward_fwd_input_map   = { "name" : [type, is_fwd_input, orig_position] ...}
    # backward_grad_input_map  = { "name" : [type, fwd_position, orig_position] ...}
    # backward_grad_output_map = { "name" : [type, fwd_position, orig_position] ...}
    # backward_attrs_list = [ [attr_name, attr_type, default_value, orig_position], ...]

    # Get Input AutoGradMeta
    inputs_autograd_meta_list = []
    compute_require_grad_args_list = ["trace_backward"]
    for name, (ttype, pos) in forward_inputs_position_map.items():
        input_autograd_meta_name = GetAutoGradMetaName(name)
        if IsPlainTensorType(ttype):
            input_autograd_meta = f"    egr::AutogradMeta* {input_autograd_meta_name} = egr::EagerUtils::nullable_autograd_meta({name});"
        else:
            assert IsVectorTensorType(ttype)
            input_autograd_meta_vec_name = GetAutoGradMetaVectorName(name)
            input_autograd_meta = f"    std::vector<egr::AutogradMeta*> {input_autograd_meta_vec_name} = egr::EagerUtils::nullable_autograd_meta({name});\n"
            input_autograd_meta += f"    std::vector<egr::AutogradMeta*>* {input_autograd_meta_name} = &{input_autograd_meta_vec_name};"

        inputs_autograd_meta_list.append(input_autograd_meta)
        compute_require_grad_args_list.append(input_autograd_meta_name)
    inputs_autograd_meta_str = "\n".join(inputs_autograd_meta_list)
    compute_require_grad_args_str = ",".join(compute_require_grad_args_list)

    # Get Output AutoGradMeta
    outputs_autograd_meta_list = []
    pass_stop_gradient_args_list = ["false"]
    num_fwd_outputs = len(forward_outputs_position_map.keys())
    for name, (rtype, pos) in forward_outputs_position_map.items():
        output_autograd_meta_name = GetAutoGradMetaName(name)
        output_autograd_meta_vec_name = GetAutoGradMetaVectorName(name)
        if num_fwd_outputs == 1:
            if IsPlainTensorType(rtype):
                output_autograd_meta = f"    egr::AutogradMeta* {output_autograd_meta_name} = egr::EagerUtils::autograd_meta(&api_result);"
            else:
                assert IsVectorTensorType(rtype)
                output_autograd_meta = f"    std::vector<egr::AutogradMeta*> {output_autograd_meta_vec_name} = egr::EagerUtils::autograd_meta(&api_result);\n"
                output_autograd_meta += f"    std::vector<egr::AutogradMeta*>* {output_autograd_meta_name} = &{output_autograd_meta_vec_name};"
        else:
            # Tuple api_result
            if IsPlainTensorType(rtype):
                outputs_autograd_meta = f"    egr::AutogradMeta* {output_autograd_meta_name} = egr::EagerUtils::autograd_meta(&api_result[{pos}]);"
            else:
                assert IsVectorTensorType(rtype)
                output_autograd_meta = f"    std::vector<egr::AutogradMeta*> {output_autograd_meta_vec_name} = egr::EagerUtils::autograd_meta(&api_result[{pos}]);\n"
                output_autograd_meta += f"    std::vector<egr::AutogradMeta*>* {output_autograd_meta_name} = &{output_autograd_meta_vec_name};"

        outputs_autograd_meta_list.append(output_autograd_meta)
        pass_stop_gradient_args_list.append(output_autograd_meta_name)

    # ComputeRequireGrad & PassStopGradient
    outputs_autograd_meta_str = "\n".join(outputs_autograd_meta_list)
    pass_stop_gradient_args_str = ",".join(pass_stop_gradient_args_list)

    # Node Construction
    num_bwd_inputs = len(backward_grad_input_map.keys())
    num_bwd_outputs = len(backward_grad_output_map.keys())
    grad_node_name = GetGradNodeName(fwd_api_name)
    node_construction_str = f"        auto grad_node = std::make_shared<{grad_node_name}>({num_bwd_inputs}, {num_bwd_outputs});"

    # SetAttributes
    set_attributes_list = []
    for name, _, _, _ in backward_attrs_list:
        set_attributes = f"        grad_node->SetAttribute{name}({name});"
        set_attributes_list.append(set_attributes)
    set_attributes_str = "\n".join(set_attributes_list)

    # SetTensorWrappers
    set_tensor_wrappers_list = []
    for name, (_, is_fwd_input, _) in backward_fwd_input_map.items():
        if is_fwd_input:
            set_tensor_wrappers = f"        grad_node->SetTensorWrapper{name}({name}, true);"
        else:
            set_tensor_wrappers = f"        grad_node->SetTensorWrapper{name}({name}, false);"
        set_tensor_wrappers_list.append(set_tensor_wrappers)
    set_tensor_wrappers_str = "\n".join(set_tensor_wrappers_list)

    # SetGradOutMeta & SetEdges
    set_grad_out_meta_list = []
    set_edges_list = []
    for name, (_, pos) in forward_inputs_position_map.items():
        input_autograd_meta_name = GetAutoGradMetaName(name)
        set_grad_out_meta = f"        grad_node->SetGradOutMeta({input_autograd_meta_name}, {pos});"
        set_edges = f"        grad_node->AddEdges({input_autograd_meta_name}, {pos});"
        set_grad_out_meta_list.append(set_grad_out_meta)
        set_edges_list.append(set_edges)
    set_grad_out_meta_str = "\n".join(set_grad_out_meta_list)
    set_edges_str = "\n".join(set_edges_list)

    # SetOutRank & SetHistory & SetGradInMeta
    set_out_rank_list = []
    set_history_list = []
    set_grad_in_meta_list = []
    set_retain_grad_list = []
    num_outputs = len(forward_outputs_position_map.keys())
    for name, (_, pos) in forward_outputs_position_map.items():
        output_autograd_meta_name = GetAutoGradMetaName(name)
        set_out_rank = f"        egr::EagerUtils::SetOutRankWithSlot({output_autograd_meta_name}, {pos});"
        set_history = f"        egr::EagerUtils::SetHistory({output_autograd_meta_name}, grad_node);"
        set_grad_in_meta = f"        grad_node->SetGradInMeta({output_autograd_meta_name}, {pos});"

        set_out_rank_list.append(set_out_rank)
        set_history_list.append(set_history)
        set_grad_in_meta_list.append(set_grad_in_meta)

        if num_outputs == 1:
            set_retain_grad = f"        egr::EagerUtils::CheckAndRetainGrad(api_result);"
        else:
            set_retain_grad = f"        egr::EagerUtils::CheckAndRetainGrad(api_result[{pos}]);"
        set_retain_grad_list.append(set_retain_grad)
    set_out_rank_str = "\n".join(set_out_rank_list)
    set_history_str = "\n".join(set_history_list)
    set_grad_in_meta_str = "\n".join(set_grad_in_meta_list)
    set_retain_grad_str = "\n".join(set_retain_grad_list)

    NODE_CREATION_TEMPLATE = """

    // Get AutoGradMeta
{}
{}
    bool trace_backward = egr::Controller::Instance().HasGrad();

    bool require_any_grad = egr::EagerUtils::ComputeRequireGrad({});
    if(require_any_grad) {{
        egr::EagerUtils::PassStopGradient({});
        
        // Node Construction
{}

        // SetAttributes
{}

        // SetTensorWrappers
{}

        // SetGradOutMeta & SetEdges
{}
{}

        // SetOutRank & SetHistory & SetGradInMeta & RetainGrad
{}
{}
{}
{}

    }}

"""
    node_creation_str = NODE_CREATION_TEMPLATE.format(
        inputs_autograd_meta_str, outputs_autograd_meta_str,
        compute_require_grad_args_str, pass_stop_gradient_args_str,
        node_construction_str, set_attributes_str, set_tensor_wrappers_str,
        set_grad_out_meta_str, set_edges_str, set_out_rank_str, set_history_str,
        set_grad_in_meta_str, set_retain_grad_str)

    return node_creation_str


def GenerateForwardDefinition(fwd_api_name, bwd_api_name,
                              forward_inputs_position_map,
                              forward_outputs_position_map, forward_attrs_list,
                              backward_fwd_input_map, backward_grad_input_map,
                              backward_grad_output_map, backward_attrs_list):
    # fwd_api_name = ""
    # forward_inputs_position_map = { "name" : [type, fwd_position] }
    # forward_outputs_position_map = { "name" : [type, fwd_position] }
    # forward_attrs_list = [ [attr_name, attr_type, default_value, orig_position], ...]
    # backward_fwd_input_map   = { "name" : [type, is_fwd_input, orig_position] ...}
    # backward_grad_input_map  = { "name" : [type, fwd_position, orig_position] ...}
    # backward_grad_output_map = { "name" : [type, fwd_position, orig_position] ...}
    # backward_attrs_list = [ [attr_name, attr_type, default_value, orig_position], ...]

    # Get Function Args
    num_inputs = len(forward_attrs_list) + len(forward_inputs_position_map.keys(
    ))
    inputs_args_definition_list = ["" for i in range(num_inputs)]
    inputs_args_declaration_list = ["" for i in range(num_inputs)]
    inputs_call_list = ["" for i in range(num_inputs)]
    for name, (ttype, pos) in forward_inputs_position_map.items():
        inputs_call_list[pos] = f"{name}"
        if IsPlainTensorType(ttype):
            inputs_args_definition_list[
                pos] = f"const paddle::experimental::Tensor& {name}"
            inputs_args_declaration_list[
                pos] = f"const paddle::experimental::Tensor& {name}"
        else:
            assert IsVectorTensorType(ttype)
            inputs_args_definition_list[
                pos] = f"const std::vector<paddle::experimental::Tensor>& {name}"
            inputs_args_declaration_list[
                pos] = f"const std::vector<paddle::experimental::Tensor>& {name}"

    for name, atype, default_val, pos in forward_attrs_list:
        inputs_call_list[pos] = name
        if default_val is not None:
            inputs_args_declaration_list[
                pos] = f"{atype} {name} = {default_val}"
        else:
            inputs_args_declaration_list[pos] = f"{atype} {name}"
        inputs_args_definition_list[pos] = f"{atype} {name}"

    inputs_args_declaration_str = ", ".join(inputs_args_declaration_list)
    inputs_args_definition_str = ", ".join(inputs_args_definition_list)
    inputs_call_args_str = ", ".join(inputs_call_list)

    # Forward Full Logic
    forward_call_str = f"auto api_result = paddle::experimental::{fwd_api_name}({inputs_call_args_str});"

    # Get return type list & outputs
    num_outputs = len(forward_outputs_position_map.keys())
    returns_type_list = ["" for i in range(num_outputs)]
    returns_list = ["" for i in range(num_outputs)]
    for name, (rtype, pos) in forward_outputs_position_map.items():
        if num_outputs == 1:
            returns_list[0] = f"api_result"
        else:
            # Tuple api_result
            returns_list[pos] = f"api_result[{pos}]"

        if IsPlainTensorType(rtype):
            returns_type_list[pos] = "paddle::experimental::Tensor"
        else:
            assert IsVectorTensorType(rtype)
            returns_type_list[pos] = "std::vector<paddle::experimental::Tensor>"

    if num_outputs == 1:
        returns_str = returns_list[0]
        returns_type_str = returns_type_list[0]
    else:
        returns_type_str = ", ".join(returns_type_list)
        returns_type_str = f"std::tuple<{returns_type_str}>"
        returns_str = ", ".join(returns_list)
        returns_str = f"std::make_tuple({returns_str})"

    node_creation_str = GenerateNodeCreationCodes(
        fwd_api_name, bwd_api_name, forward_inputs_position_map,
        forward_outputs_position_map, forward_attrs_list,
        backward_fwd_input_map, backward_grad_input_map,
        backward_grad_output_map, backward_attrs_list)

    FORWARD_FUNCTION_TEMPLATE = """
{} {}({}) {{
    // Forward API Call
    {}
    
{}

    // Returns
    return {};
}}
"""

    forward_function_name = GetForwardFunctionName(fwd_api_name)
    forward_function_str = FORWARD_FUNCTION_TEMPLATE.format(
        returns_type_str, forward_function_name, inputs_args_definition_str,
        forward_call_str, node_creation_str, returns_str)
    forward_function_declaration_str = f"{returns_type_str} {forward_function_name}({inputs_args_declaration_str});"

    return forward_function_str, forward_function_declaration_str


def CollectCoreOpsInformation(fwd_api_name, forward_inputs_position_map,
                              forward_outputs_position_map, forward_attrs_list):
    # fwd_api_name : ""
    # forward_inputs_position_map = { "name" : [type, fwd_position] }
    # forward_outputs_position_map = { "name" : [type, fwd_position] }
    # forward_attrs_list = [ [attr_name, attr_type, default_value, orig_position], ...]
    num_args = len(forward_inputs_position_map.keys()) + len(forward_attrs_list)
    num_returns = len(forward_outputs_position_map.keys())

    final_state_fwd_api_name = "final_state_" + fwd_api_name
    core_ops_returns_info[
        final_state_fwd_api_name] = ["" for i in range(num_returns)]
    core_ops_args_info[final_state_fwd_api_name] = ["" for i in range(num_args)]
    core_ops_args_type_info[
        final_state_fwd_api_name] = ["" for i in range(num_args)]
    for name, (ttype, pos) in forward_inputs_position_map.items():
        core_ops_args_info[final_state_fwd_api_name][pos] = name
        if IsPlainTensorType(ttype):
            core_ops_args_type_info[final_state_fwd_api_name][pos] = "tensor"
        else:
            assert IsVectorTensorType(ttype)
            core_ops_args_type_info[final_state_fwd_api_name][pos] = "list"

    for name, _, _, pos in forward_attrs_list:
        core_ops_args_info[final_state_fwd_api_name][pos] = name

    for name, (ttype, pos) in forward_outputs_position_map.items():
        core_ops_returns_info[final_state_fwd_api_name][pos] = name


def GenerateCoreOpInfoDeclaration():
    core_ops_declaration_str = """
    extern std::unordered_map<std::string, std::vector<std::string>> core_ops_final_state_args_info;
    extern std::unordered_map<std::string, std::vector<std::string>> core_ops_final_state_args_type_info;
    extern std::unordered_map<std::string, std::vector<std::string>> core_ops_final_state_returns_info;

"""
    return core_ops_declaration_str


def GenerateCoreOpInfoDefinition():

    CORE_OPS_INFO_TEMPLATE = """
std::unordered_map<std::string, std::vector<std::string>> core_ops_final_state_args_info = {{
    {}
}};
std::unordered_map<std::string, std::vector<std::string>> core_ops_final_state_args_type_info = {{
    {}
}};
std::unordered_map<std::string, std::vector<std::string>> core_ops_final_state_returns_info = {{
    {}
}};

"""
    op_args_info_list = []
    for op_name, arg_list in core_ops_args_info.items():
        arg_str = ",".join(["\"" + v + "\"" for v in arg_list])
        op_args_info = f"{{ \"{op_name}\", {{ {arg_str} }} }},"
        op_args_info_list.append(op_args_info)

    op_types_info_list = []
    for op_name, type_list in core_ops_args_type_info.items():
        type_str = ",".join(["\"" + v + "\"" for v in type_list])
        op_types_info = f"{{ \"{op_name}\", {{ {type_str} }} }},"
        op_types_info_list.append(op_types_info)

    op_returns_info_list = []
    for op_name, return_list in core_ops_returns_info.items():
        return_str = ",".join(["\"" + v + "\"" for v in return_list])
        return_types_info = f"{{ \"{op_name}\", {{ {return_str} }} }},"
        op_returns_info_list.append(return_types_info)

    op_args_info_str = "\n".join(op_args_info_list)
    op_types_info_str = "\n".join(op_types_info_list)
    op_returns_info_str = "\n".join(op_returns_info_list)

    core_ops_info_definition_str = CORE_OPS_INFO_TEMPLATE.format(
        op_args_info_str, op_types_info_str, op_returns_info_str)

    return core_ops_info_definition_str


def GenerateNodeCCFile(filepath, node_definition_str):
    file_contents = """
#include "glog/logging.h"
#include "paddle/pten/api/all.h"
#include "paddle/pten/api/backward/backward_api.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/api/generated/eager_generated/backwards/nodes.h"

"""
    file_contents += node_definition_str
    with open(filepath, 'a') as f:
        f.write(file_contents)


def GenerateNodeHFile(filepath, node_declaration_str):
    file_contents = """
#pragma once
#include "paddle/fluid/eager/tensor_wrapper.h"
#include "paddle/fluid/eager/grad_node_info.h"

"""
    file_contents += node_declaration_str
    with open(filepath, 'a') as f:
        f.write(file_contents)


def GenerateForwardCCFile(filepath, forward_definition_str):
    file_contents = """
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/api/generated/eager_generated/backwards/nodes.h"

#include "paddle/fluid/eager/api/utils/global_utils.h"

"""

    file_contents += GenerateCoreOpInfoDefinition()
    file_contents += forward_definition_str
    with open(filepath, 'a') as f:
        f.write(file_contents)


def GenerateForwardHFile(filepath, forward_function_declaration_str):
    file_contents = """
#pragma once
#include "glog/logging.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/pten/api/all.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/op_registry.h"

"""
    file_contents += GenerateCoreOpInfoDeclaration()
    file_contents += forward_function_declaration_str
    with open(filepath, 'a') as f:
        f.write(file_contents)


if __name__ == "__main__":
    args = ParseArguments()

    api_yaml_path = args.api_yaml_path
    backward_yaml_path = args.backward_yaml_path

    fwd_api_list = ReadFwdFile(api_yaml_path)
    grad_api_dict = ReadBwdFile(backward_yaml_path)

    # Generate per Dygraph API
    node_declaration_str = ""
    node_definition_str = ""
    forward_definition_str = ""
    forward_declaration_str = ""
    for fwd_api in fwd_api_list:
        # We only generate Ops with grad
        if 'backward' not in fwd_api.keys():
            continue

        assert 'api' in fwd_api.keys()
        assert 'args' in fwd_api.keys()
        assert 'output' in fwd_api.keys()
        assert 'backward' in fwd_api.keys()

        fwd_api_name = fwd_api['api']
        fwd_args_str = fwd_api['args']
        fwd_returns_str = fwd_api['output']

        bwd_api_name = fwd_api['backward']
        assert bwd_api_name in grad_api_dict.keys()
        bwd_api = grad_api_dict[bwd_api_name]

        assert 'args' in bwd_api.keys()
        assert 'output' in bwd_api.keys()
        assert 'forward' in bwd_api.keys()
        bwd_forward_str = bwd_api['forward']
        bwd_args_str = bwd_api['args']
        bwd_returns_str = bwd_api['output']

        # Collect Forward Inputs/Outputs
        forward_inputs_list, forward_attrs_list, forward_returns_list = ParseYamlForwardFromBackward(
            bwd_forward_str)
        print("Parsed Forward Inputs List: ", forward_inputs_list)
        print("Prased Forward Attrs List: ", forward_attrs_list)
        print("Parsed Forward Returns List: ", forward_returns_list)

        # Collect Original Forward Inputs/Outputs and then perform validation checks
        orig_forward_inputs_list, orig_forward_attrs_list, orig_forward_returns_list = ParseYamlForward(
            fwd_args_str, fwd_returns_str)
        print("Parsed Original Forward Inputs List: ", orig_forward_inputs_list)
        print("Prased Original Forward Attrs List: ", orig_forward_attrs_list)
        print("Parsed Original Forward Returns List: ",
              orig_forward_returns_list)

        # Forward Validation Checks
        ForwardsValidationCheck(forward_inputs_list, forward_attrs_list,
                                forward_returns_list, orig_forward_inputs_list,
                                orig_forward_attrs_list,
                                orig_forward_returns_list)

        # Parse Backward Inputs/Outputs
        backward_inputs_list, backward_attrs_list, backward_returns_list = ParseYamlBackward(
            bwd_args_str, bwd_returns_str)
        print("Parsed Backward Inputs List: ", backward_inputs_list)
        print("Prased Backward Attrs List: ", backward_attrs_list)
        print("Parsed Backward Returns List: ", backward_returns_list)

        # Determine Forward Inputs/Outputs Position
        forward_inputs_position_map, forward_outputs_position_map = DetermineForwardPositionMap(
            forward_inputs_list, forward_returns_list)
        print("Generated Forward Input Position Map: ",
              forward_inputs_position_map)
        print("Generated Forward Output Position Map: ",
              forward_outputs_position_map)

        # SlotName Matching
        backward_fwd_input_map, backward_grad_input_map, backward_grad_output_map = SlotNameMatching(
            backward_inputs_list, backward_returns_list,
            forward_inputs_position_map, forward_outputs_position_map)
        print("Generated Backward Fwd Input Map: ", backward_fwd_input_map)
        print("Generated Backward Grad Input Map: ", backward_grad_input_map)
        print("Generated Backward Grad Output Map: ", backward_grad_output_map)

        # Backward Validation Check
        BackwardValidationCheck(backward_fwd_input_map, backward_grad_input_map,
                                backward_attrs_list)

        # Node Declaration Generation
        node_declaration_str += GenerateNodeDeclaration(
            fwd_api_name, backward_fwd_input_map, backward_attrs_list)
        print("Generated Node Declaration: ", node_declaration_str)

        node_definition_str += GenerateNodeDefinition(
            fwd_api_name, bwd_api_name, backward_fwd_input_map,
            backward_grad_input_map, backward_grad_output_map,
            backward_attrs_list)
        print("Generated Node Definition: ", node_definition_str)

        # Node Definition Generation
        definition_declaration_pair = GenerateForwardDefinition(
            fwd_api_name, bwd_api_name, forward_inputs_position_map,
            forward_outputs_position_map, forward_attrs_list,
            backward_fwd_input_map, backward_grad_input_map,
            backward_grad_output_map, backward_attrs_list)
        print("Generated Forward Definition: ", forward_definition_str)
        print("Generated Forward Declaration: ", forward_declaration_str)
        forward_definition_str += definition_declaration_pair[0]
        forward_declaration_str += definition_declaration_pair[1]

        # For python-level API dispatch
        CollectCoreOpsInformation(fwd_api_name, forward_inputs_position_map,
                                  forward_outputs_position_map,
                                  forward_attrs_list)

    # Generate Files
    nodes_h_path = args.nodes_h_path
    nodes_cc_path = args.nodes_cc_path
    forwards_h_path = args.forwards_h_path
    forwards_cc_path = args.forwards_cc_path

    for path in [
            nodes_cc_path, nodes_h_path, forwards_h_path, forwards_cc_path
    ]:
        if os.path.exists(path):
            os.remove(path)

    GenerateNodeCCFile(nodes_cc_path, node_definition_str)
    GenerateNodeHFile(nodes_h_path, node_declaration_str)
    GenerateForwardCCFile(forwards_cc_path, forward_definition_str)
    GenerateForwardHFile(forwards_h_path, forward_declaration_str)
