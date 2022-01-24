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


######################
###  File Readers  ###
######################
def ReadFwdFile(filepath):
    f = open(filepath, 'r')
    contents = yaml.load(f)
    return contents


def ReadBwdFile(filepath):
    f = open(filepath, 'r')
    contents = yaml.load(f)
    ret = {}
    for content in contents:
        assert 'grad_api' in content.keys()
        api_name = content['grad_api']
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
    aname = r'(\S+)'
    pattern = f'{atype}{aname}'
    for i in range(len(args)):
        arg = args[i]
        m = re.search(pattern, arg)
        arg_type = m.group(1)
        arg_name = m.group(3).split("=")[0]
        default_value = m.group(3).split("=")[1] if len(m.group(3).split(
            "=")) > 1 else None
        if "Tensor" in arg_type:
            assert default_value is None
            inputs_list.append([arg_name, arg_type, i])
        else:
            attrs_list.append([arg_name, arg_type, default_value, i])

    return inputs_list, attrs_list


def ParseYamlReturns(string):
    # Example: Tensor, Tensor

    # list = [ [ret_type, orig_position], ...]
    returns_list = []

    returns = [x.strip() for x in string.strip().split(",")]
    for i in range(len(returns)):
        ret = returns[i]
        returns_list.append([ret, i])

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
    # inputs_list          = [ [input_name, input_type, orig_position], ...]
    # attrs_list           = [ [attr_name, attr_type, default_value, orig_position], ...]
    # forward_returns_list = [ [ret_name, ret_type, orig_position] ...]
    # orig_returns_list    = [ [ret_type, orig_position], ...]
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
        orig_return_type = orig_forward_returns_list[i][0]
        orig_return_pos = orig_forward_returns_list[i][1]
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
    # backward_fwd_input_map   = { "name" : [type, is_fwd_input, orig_position] ...}
    # backward_grad_input_map  = { "name" : [type, fwd_position, orig_position] ...}
    # backward_attrs_list = [ [attr_name, attr_type, default_value, orig_position], ...]

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
    # inputs_list          = [ [input_name, input_type, orig_position], ...]
    # forward_returns_list = [ [ret_name, ret_type, orig_position] ...]

    # forward_position_map = { "name" : [type, fwd_position] ...}
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

    # backward_inputs_list  = [ [input_name, input_type, orig_position], ...]
    # backward_returns_list = [ [ret_name, ret_type, orig_position], ...]
    # forward_inputs_position_map  = { "name" : [type, fwd_position] }
    # forward_outputs_position_map = { "name" : [type, fwd_position] }

    # backward_fwd_input_map   = { "name" : [type, is_fwd_input, orig_position] ...}
    # backward_grad_input_map  = { "name" : [type, fwd_position, orig_position] ...}
    # backward_grad_output_map = { "name" : [type, fwd_position, orig_position] ...}

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
   void SetTensorWrapper{}(const egr::EagerTensor& {}, bool full_reserved) {{     
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
   void SetTensorWrapper{}(const std::vector<egr::EagerTensor>& {}, bool full_reserved) {{
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
   {} {};
"""
        attribute_members_str += ATTRIBUTE_MEMBER_TEMPLATE.format(
            GetConstReference(atype), saved_attr_name)
    # End: SetAttributes & Attribute Members

    NODE_DECLARATION_TEMPLATE = """
class GradNode{} : public egr::GradNodeBase {{
 public:
  GradNode{}() : egr::GradNodeBase() {{}}
  GradNode{}(size_t bwd_in_slot_num, size_t bwd_out_slot_num) : 
      egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {{}}
  ~GradNode{}() override = default;

  virtual std::vector<std::vector<egr::EagerTensor>> operator()(
      const std::vector<std::vector<egr::EagerTensor>>& grads) override;
  
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
        forward_op_name, forward_op_name, forward_op_name, forward_op_name,
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
        if is_fwd_input:
            grad_api_args[
                grad_api_position] = f"egr::EagerUtils::RecoverTensorWrapper(&this->{tensor_wrapper_name}, true)"
        else:
            grad_api_args[
                grad_api_position] = f"egr::EagerUtils::RecoverTensorWrapper(&this->{tensor_wrapper_name}, false)"

    for _, (_, fwd_position,
            grad_api_position) in backward_grad_input_map.items():
        grad_api_args[
            grad_api_position] = f"*grads[{fwd_position}].Tensor().get()"

    for name, _, _, grad_api_position in backward_attrs_list:
        saved_attribute_name = GetSavedName(name)
        grad_api_args[grad_api_position] = f"this->{saved_attribute_name}"
    grad_api_args_str = ", ".join(grad_api_args)

    # Construct grad_api returns
    num_outputs = len(backward_grad_output_map.keys())
    returns_list = ["" for i in range(num_outputs)]
    for _, (ttype, fwd_position,
            grad_api_position) in backward_grad_output_map.items():
        # Infer Grad API Return Type
        if num_outputs == 1:
            # Single tensor output, return as is
            if IsPlainTensorType(ttype):
                returns_list[0] = "{grad_api_returns}"
            else:
                assert IsVectorTensorType(ttype)
                returns_list[0] = "grad_api_returns"
        else:
            # Rearrange output order accordingly
            if IsPlainTensorType(ttype):
                returns_list[
                    fwd_position] = f"{{ grad_api_returns[{grad_api_position}] }}"
            else:
                assert IsVectorTensorType(ttype)
                returns_list[
                    fwd_position] = f"grad_api_returns[{grad_api_position}]"
    returns_str = ", ".join(returns_list)
    returns_str = f"{{ {returns_str} }}"

    FUNCTION_TEMPLATE = """
std::vector<std::vector<egr::EagerTensor>> GradNode{}::operator()(const std::vector<std::vector<egr::EagerTensor>>& grads) {{
    // Call grad_api function
    auto grad_api_returns = {}({});
    return {};
}}
  """

    node_definition_str = FUNCTION_TEMPLATE.format(
        fwd_api_name, bwd_api_name, grad_api_args_str, returns_str)

    return node_definition_str


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
    inputs_args_list = ["" for i in range(num_inputs)]
    inputs_call_list = ["" for i in range(num_inputs)]
    for name, (ttype, pos) in forward_inputs_position_map.items():
        inputs_call_list[pos] = name
        if IsPlainTensorType(ttype):
            inputs_args_list[pos] = f"const egr::EagerTensor& {name}"
        else:
            assert IsVectorTensorType(ttype)
            inputs_args_list[
                pos] = f"const std::vector<egr::EagerTensor>& {name}"

    for name, atype, default_val, pos in forward_attrs_list:
        inputs_call_list[pos] = name
        if default_val is not None:
            inputs_args_list[pos] = f"{atype} {name} = {default_val}"
        else:
            inputs_args_list[pos] = f"{atype} {name}"

    inputs_args_str = ", ".join(inputs_args_list)
    inputs_call_str = ", ".join(inputs_call_list)

    # Forward Full Logic
    forward_call_str = f"auto api_result = {fwd_api_name}({inputs_call_str});"

    # Get return type list & outputs
    num_outputs = len(forward_outputs_position_map.keys())
    returns_type_list = ["" for i in range(num_outputs)]
    returns_list = ["" for i in range(num_outputs)]
    for name, (rtype, pos) in forward_outputs_position_map.items():
        if num_outputs == 1:
            returns_list[
                0] = f"egr::EagerUtils::CreateEagerTensorFromTensor(api_result)"
        else:
            # Tuple api_result
            returns_list[
                pos] = f"egr::EagerUtils::CreateEagerTensorFromTensor(api_result[{pos}])"

        if IsPlainTensorType(rtype):
            returns_type_list[pos] = "egr::EagerTensor"
        else:
            assert IsVectorTensorType(rtype)
            returns_type_list[pos] = "std::vector<egr::EagerTensor>"

    if num_outputs == 1:
        returns_str = returns_list[0]
        returns_type_str = returns_type_list[0]
    else:
        returns_type_str = ", ".join(returns_type_list)
        returns_type_str = f"std::tuple<{returns_type_str}>"
        returns_str = ", ".join(returns_list)
        returns_str = f"std::make_tuple({returns_str})"

    FORWARD_FUNCTION_TEMPLATE = """
    {} {} ({}) {{

    }}
"""


if __name__ == "__main__":
    filepath = "/workspace/PaddleRepos/Paddle4/python/paddle/utils/code_gen/api.yaml"
    fwd_api_list = ReadFwdFile(filepath)

    filepath = "/workspace/PaddleRepos/Paddle4/python/paddle/utils/code_gen/grad.yaml"
    grad_api_dict = ReadBwdFile(filepath)

    # Generate per Dygraph API
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
        node_declaration_str = GenerateNodeDeclaration(
            fwd_api_name, backward_fwd_input_map, backward_attrs_list)
        print("Generated Node Declaration: ", node_declaration_str)

        node_definition_str = GenerateNodeDefinition(
            fwd_api_name, bwd_api_name, backward_fwd_input_map,
            backward_grad_input_map, backward_grad_output_map,
            backward_attrs_list)
        print("Generated Node Definition: ", node_definition_str)

        # Node Definition Generation
        forward_definition_str = GenerateForwardDefinition(
            fwd_api_name, bwd_api_name, forward_inputs_position_map,
            forward_outputs_position_map, forward_attrs_list,
            backward_fwd_input_map, backward_grad_input_map,
            backward_grad_output_map, backward_attrs_list)
        print("Generated Forward Definition: ", forward_definition_str)
