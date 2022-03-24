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
from codegen_utils import core_ops_returns_info, core_ops_args_info, core_ops_args_type_info
from codegen_utils import yaml_types_mapping
from codegen_utils import ReadFwdFile, ReadBwdFile
from codegen_utils import FindGradName, FindForwardName, GetSavedName, GetGradNodeName
from codegen_utils import IsPlainTensorType, IsVectorTensorType
from codegen_utils import GetConstReference, RemoveConstAndReference
from codegen_utils import GetDygraphForwardFunctionName, GetIntermediateAPIFunctionName
from codegen_utils import GetAutoGradMetaName, GetAutoGradMetaVectorName
from codegen_utils import RemoveSpecialSymbolsInName, RecoverBaseNameOfInplaceFunction
from codegen_utils import GetInplacedFunctionName
from codegen_utils import ParseYamlArgs, ParseYamlReturns, ParseYamlForwardFromBackward
from codegen_utils import ParseYamlForward, ParseYamlBackward
from codegen_utils import FunctionGeneratorBase, YamlGeneratorBase
from codegen_utils import ops_to_fill_zero_for_empty_grads


###########
## Utils ##
###########
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


########################
## Code Gen Templates ##
########################
SET_PLAIN_TENSOR_WRAPPER_TEMPLATE = \
"""
   void SetTensorWrapper{}(const paddle::experimental::Tensor& {}, bool full_reserved) {{     
     {} = egr::TensorWrapper({}, full_reserved, {});
   }}
"""

PLAIN_TENSOR_MEMBER_TEMPLATE = \
"""
       egr::TensorWrapper {};
"""

CLEAR_TENSOR_WRAPPER_TEMPLATE = \
"""
       {}.clear();
"""

SET_VECTOR_TENSOR_WRAPPER_TEMPLATE = \
"""
       void SetTensorWrapper{}(const std::vector<paddle::experimental::Tensor>& {}, bool full_reserved) {{
         for(const auto& eager_tensor : {}) {{
            {}.emplace_back( egr::TensorWrapper(eager_tensor, full_reserved, {}) );
         }};
       }}
"""

VECTOR_TENSOR_MEMBER_TEMPLATE = \
"""
       std::vector<egr::TensorWrapper> {};
"""

CLEAR_VECTOR_TENSOR_WRAPPERS_TEMPLATE = \
"""
       for (auto tw: {}) {
         tw.clear();
       };
"""

SET_ATTR_METHOD_TEMPLATE = \
"""
       void SetAttribute{}({} {}) {{
         {} = {};
       }}
"""

ATTRIBUTE_MEMBER_WITH_DEFAULT_TEMPLATE = \
"""
       {} {} = {};
"""

ATTRIBUTE_MEMBER_TEMPLATE = \
"""
       {} {};
"""

NODE_DECLARATION_TEMPLATE = \
"""
    class {} : public egr::GradNodeBase {{
     public:
      {}() : egr::GradNodeBase() {{}}
      {}(size_t bwd_in_slot_num, size_t bwd_out_slot_num) : 
          egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {{}}
      ~{}() override = default;

      virtual std::vector<std::vector<paddle::experimental::Tensor>> operator()(
          std::vector<std::vector<paddle::experimental::Tensor>>& grads, bool create_graph = false) override;
      std::string name() override {{ return \" {} \"; }}
      
      void ClearTensorWrappers() override {{
          {}
        is_tensor_wrappers_cleared = true;
      }}
      
      // SetTensorWrapperX, SetTensorWrapperY, ...
      {}
      // SetAttributes
      {}

      bool IsTensorWrappersCleared() override {{
          return is_tensor_wrappers_cleared;  
      }}
     private:
      // TensorWrappers
      {}

      bool is_tensor_wrappers_cleared = false;

      // Attributes
      {}
    }};
"""

FUNCTION_TEMPLATE = \
"""
    std::vector<std::vector<paddle::experimental::Tensor>> {}::operator()(std::vector<std::vector<paddle::experimental::Tensor>>& grads, bool create_graph) {{
        {}
        auto hooked_grads = ApplyGradientHooks(grads);

        // Call grad_api function
        VLOG(3) << \"Final State Running: \" << \"{}\"; 
        auto grad_api_returns = {}{}({});
        {}
    }}
"""

FORWARD_FUNCTION_TEMPLATE = \
"""
    {} {}({}) {{
        {}
        
    {}

        // Returns
        return {};
    }}

"""

NODE_CREATION_TEMPLATE = \
"""
    // Get AutoGradMeta
{}
    bool trace_backward = egr::Controller::Instance().HasGrad();
    bool require_any_grad = egr::EagerUtils::ComputeRequireGrad({});
{}
    // Forward API Call
    {}
{}
    {{
{}
{}
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
    }}
"""

NAMESPACE_WRAPPER_TEMPLATE = \
"""
namespace {} {{
    {}
}}
"""

NODE_CC_FILE_TEMPLATE = \
"""
#include "glog/logging.h"
#include "paddle/phi/api/all.h"
#include "paddle/phi/api/backward/backward_api.h"
#include "paddle/phi/api/backward/sparse_bw_api.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/api/generated/eager_generated/backwards/nodes.h"
#include "paddle/fluid/eager/to_static/run_program_op_node.h"

#include "paddle/phi/api/include/sparse_api.h"

{}
"""

NODE_H_FILE_TEMPLATE = \
"""
#pragma once
#include "paddle/fluid/eager/tensor_wrapper.h"
#include "paddle/fluid/eager/grad_node_info.h"

{}
"""

FORWARD_CC_FILE_TEMPLATE = \
"""
#include "paddle/phi/api/lib/dygraph_api.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/api/generated/eager_generated/backwards/nodes.h"

#include "paddle/phi/api/include/sparse_api.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"

{}
{}
"""

FORWARD_H_FILE_TEMPLATE = \
"""
#pragma once
#include "glog/logging.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/phi/api/all.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/eager/to_static/run_program_op_func.h"

{}
{}
"""

CORE_OPS_INFO_TEMPLATE = \
"""
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

CORE_OPS_DECLARATION_TEMPLATE = \
"""
    extern std::unordered_map<std::string, std::vector<std::string>> core_ops_final_state_args_info;
    extern std::unordered_map<std::string, std::vector<std::string>> core_ops_final_state_args_type_info;
    extern std::unordered_map<std::string, std::vector<std::string>> core_ops_final_state_returns_info;

"""

CHECK_INPLACE_TEMPLATE = \
"""
    // Check Inplace
    egr::EagerUtils::CheckInplace({}, {}, require_any_grad);\n
"""

BUMP_INPLACE_VERSION_TEMPLATE = \
"""
    // Bump Inplace Version
    {}.bump_inplace_version();
    VLOG(3) << \"Tensor(\" << {}.name() << \") uses Inplace Strategy.\";\n
"""


#######################
## Generator Helpers ##
#######################
def GenerateCoreOpInfoDeclaration():
    return CORE_OPS_DECLARATION_TEMPLATE


def GenerateCoreOpInfoDefinition():

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


#####################
## Generator Class ##
#####################
class DygraphSingleFunctionGenerator(FunctionGeneratorBase):
    def __init__(self, forward_api_contents, grad_api_contents, namespace):
        self.forward_api_contents = forward_api_contents
        # Members from Parent:
        #self.namespace
        #self.forward_api_contents
        #self.forward_api_name
        #self.orig_forward_inputs_list
        #self.orig_forward_attrs_list
        #self.orig_forward_returns_list
        #self.forward_inputs_position_map
        #self.forward_outputs_position_map
        #self.optional_inputs
        #self.no_need_buffers
        #self.intermediate_outputs   
        #self.inplace_map
        FunctionGeneratorBase.__init__(self, forward_api_contents, namespace)

        self.grad_api_contents = grad_api_contents

        # Raw Contents
        self.backward_forward_str = ""
        self.backward_api_name = ""

        self.forward_attrs_list = [
        ]  #[ [attr_name, attr_type, default_value, orig_position], ...]
        self.forward_inputs_list = [
        ]  #[ [arg_name, arg_type, orig_position], ...]
        self.forward_returns_list = [
        ]  #[ [ret_name, ret_type, orig_position], ...]

        self.backward_inputs_list = [
        ]  #[ [attr_name, attr_type, default_value, orig_position], ...]
        self.backward_attrs_list = [
        ]  #[ [arg_name, arg_type, orig_position], ...]
        self.backward_returns_list = [
        ]  #[ [ret_name, ret_type, orig_position], ...]

        # SlotNameMatched Backward Data
        self.backward_forward_inputs_map = {
        }  #{ "name" : [type, is_fwd_input, orig_position] ...}
        self.backward_grad_inputs_map = {
        }  #{ "name" : [type, fwd_position, orig_position] ...}
        self.backward_grad_outputs_map = {
        }  #{ "name" : [type, fwd_position, orig_position] ...}

        # Generated Results
        self.forward_definition_str = ""
        self.forward_declaration_str = ""
        self.node_declaration_str = ""
        self.node_definition_str = ""

    def DygraphYamlValidationCheck(self):
        forward_api_contents = self.forward_api_contents
        grad_api_contents = self.grad_api_contents

        assert 'api' in forward_api_contents.keys()
        assert 'args' in forward_api_contents.keys()
        assert 'output' in forward_api_contents.keys()
        assert 'backward' in forward_api_contents.keys()

        assert 'args' in grad_api_contents.keys()
        assert 'output' in grad_api_contents.keys()
        assert 'forward' in grad_api_contents.keys()

    def ForwardsValidationCheck(self):
        forward_inputs_list = self.forward_inputs_list
        forward_attrs_list = self.forward_attrs_list
        forward_returns_list = self.forward_returns_list

        orig_forward_inputs_list = self.orig_forward_inputs_list
        orig_forward_attrs_list = self.orig_forward_attrs_list
        orig_forward_returns_list = self.orig_forward_returns_list

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

    def BackwardValidationCheck(self):
        backward_forward_inputs_map = self.backward_forward_inputs_map
        backward_grad_inputs_map = self.backward_grad_inputs_map
        backward_attrs_list = self.backward_attrs_list

        # Check Order: TensorWrappers, GradTensors, Attributes
        max_fwd_input_position = -1
        for _, (_, _, pos) in backward_forward_inputs_map.items():
            max_fwd_input_position = max(max_fwd_input_position, pos)

        max_grad_tensor_position = -1
        for _, (_, _, pos) in backward_grad_inputs_map.items():
            assert pos > max_fwd_input_position
            max_grad_tensor_position = max(max_grad_tensor_position, pos)

        max_attr_position = -1
        for _, _, _, pos in backward_attrs_list:
            assert pos > max_grad_tensor_position
            max_attr_position = max(max_attr_position, pos)

    def IntermediateValidationCheck(self):
        intermediate_outputs = self.intermediate_outputs
        forward_returns_list = self.forward_returns_list
        """
        Check whether intermediate_outputs are positioned
        at the very end of forward_returns_list
        """
        intermediate_positions = range(
            len(forward_returns_list) - len(intermediate_outputs),
            len(forward_returns_list))
        for ret_name, _, pos in forward_returns_list:
            if ret_name in intermediate_outputs:
                assert pos in intermediate_positions

    def CollectBackwardInfo(self):
        forward_api_contents = self.forward_api_contents
        grad_api_contents = self.grad_api_contents

        self.backward_api_name = forward_api_contents['backward']
        self.backward_forward_str = grad_api_contents['forward']

        backward_args_str = grad_api_contents['args']
        backward_returns_str = grad_api_contents['output']

        self.backward_inputs_list, self.backward_attrs_list, self.backward_returns_list = ParseYamlBackward(
            backward_args_str, backward_returns_str)
        print("Parsed Backward Inputs List: ", self.backward_inputs_list)
        print("Prased Backward Attrs List: ", self.backward_attrs_list)
        print("Parsed Backward Returns List: ", self.backward_returns_list)

    def CollectForwardInfoFromBackwardContents(self):

        backward_forward_str = self.backward_forward_str

        self.forward_inputs_list, self.forward_attrs_list, self.forward_returns_list = ParseYamlForwardFromBackward(
            backward_forward_str)

    def SlotNameMatching(self):
        backward_inputs_list = self.backward_inputs_list
        backward_returns_list = self.backward_returns_list
        forward_inputs_position_map = self.forward_inputs_position_map
        forward_outputs_position_map = self.forward_outputs_position_map

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

                self.backward_grad_inputs_map[backward_input_name] = [
                    backward_input_type, matched_forward_output_pos,
                    backward_input_pos
                ]
            else:
                # TensorWrapper Input
                if backward_input_name in forward_inputs_position_map.keys():
                    tensor_wrapper_type = forward_inputs_position_map[
                        backward_input_name][0]
                    self.backward_forward_inputs_map[backward_input_name] = [
                        backward_input_type, True, backward_input_pos
                    ]

                elif backward_input_name in forward_outputs_position_map.keys():
                    tensor_wrapper_type = forward_outputs_position_map[
                        backward_input_name][0]
                    self.backward_forward_inputs_map[backward_input_name] = [
                        backward_input_type, False, backward_input_pos
                    ]
                else:
                    assert False, backward_input_name

        for backward_output in backward_returns_list:
            backward_output_name = backward_output[0]
            backward_output_type = backward_output[1]
            backward_output_pos = backward_output[2]

            backward_fwd_name = FindForwardName(backward_output_name)
            assert backward_fwd_name is not None
            assert backward_fwd_name in forward_inputs_position_map.keys(
            ), f"Unable to find {backward_fwd_name} in forward inputs"

            matched_forward_input_type = forward_inputs_position_map[
                backward_fwd_name][0]
            matched_forward_input_pos = forward_inputs_position_map[
                backward_fwd_name][1]

            self.backward_grad_outputs_map[backward_output_name] = [
                backward_output_type, matched_forward_input_pos,
                backward_output_pos
            ]
        print("Generated Backward Fwd Input Map: ",
              self.backward_forward_inputs_map)
        print("Generated Backward Grad Input Map: ",
              self.backward_grad_inputs_map)
        print("Generated Backward Grad Output Map: ",
              self.backward_grad_outputs_map)

    def GenerateNodeDeclaration(self):
        forward_op_name = self.forward_api_name
        backward_forward_inputs_map = self.backward_forward_inputs_map
        backward_attrs_list = self.backward_attrs_list
        no_need_buffers = self.no_need_buffers

        # SetTensorWrapper Methods & TensorWrapper Members
        set_tensor_wrapper_methods_str = ""
        tensor_wrapper_members_str = ""
        clear_tensor_wrapper_str = ""
        for tname, (ttype, is_fwd_input,
                    _) in backward_forward_inputs_map.items():
            no_need_buffer = "true" if tname in no_need_buffers else "false"
            tensor_wrapper_name = GetSavedName(tname)
            if IsPlainTensorType(ttype):
                set_tensor_wrapper_methods_str += SET_PLAIN_TENSOR_WRAPPER_TEMPLATE.format(
                    tname, tname, tensor_wrapper_name, tname, no_need_buffer)

                tensor_wrapper_members_str += PLAIN_TENSOR_MEMBER_TEMPLATE.format(
                    tensor_wrapper_name)

                clear_tensor_wrapper_str += CLEAR_TENSOR_WRAPPER_TEMPLATE.format(
                    tensor_wrapper_name)

            else:
                assert IsVectorTensorType(ttype)
                set_tensor_wrapper_methods_str += SET_VECTOR_TENSOR_WRAPPER_TEMPLATE.format(
                    tname, tname, tname, tensor_wrapper_name, no_need_buffer)

                tensor_wrapper_members_str += VECTOR_TENSOR_MEMBER_TEMPLATE.format(
                    tensor_wrapper_name)

                clear_tensor_wrapper_str += CLEAR_VECTOR_TENSOR_WRAPPERS_TEMPLATE.format(
                    tensor_wrapper_name)

        # SetAttributes & Attribute Members
        set_attribute_methods_str = ""
        attribute_members_str = ""
        for aname, atype, default_val, _ in backward_attrs_list:
            saved_attr_name = GetSavedName(aname)
            set_attribute_methods_str += SET_ATTR_METHOD_TEMPLATE.format(
                aname, GetConstReference(atype), aname, saved_attr_name, aname)

            if default_val:
                attribute_members_str += ATTRIBUTE_MEMBER_WITH_DEFAULT_TEMPLATE.format(
                    RemoveConstAndReference(atype), saved_attr_name,
                    default_val)
            else:
                attribute_members_str += ATTRIBUTE_MEMBER_TEMPLATE.format(
                    RemoveConstAndReference(atype), saved_attr_name)

        grad_node_name = GetGradNodeName(forward_op_name)
        self.node_declaration_str = NODE_DECLARATION_TEMPLATE.format(
            grad_node_name, grad_node_name, grad_node_name, grad_node_name,
            grad_node_name, clear_tensor_wrapper_str,
            set_tensor_wrapper_methods_str, set_attribute_methods_str,
            tensor_wrapper_members_str, attribute_members_str)

        print("Generated Node Declaration: ", self.node_declaration_str)

    def GenerateNodeDefinition(self):
        namespace = self.namespace
        forward_api_name = self.forward_api_name
        backward_api_name = self.backward_api_name
        backward_forward_inputs_map = self.backward_forward_inputs_map
        backward_grad_inputs_map = self.backward_grad_inputs_map
        backward_grad_outputs_map = self.backward_grad_outputs_map
        backward_attrs_list = self.backward_attrs_list

        # Construct grad_api function args
        # Order: TensorWrappers, GradTensors, Attributes
        grad_api_args_len = len(backward_forward_inputs_map.keys()) + len(
            backward_grad_inputs_map.keys()) + len(backward_attrs_list)
        grad_api_args = ["" for i in range(grad_api_args_len)]
        for name, (_, is_fwd_input,
                   grad_api_position), in backward_forward_inputs_map.items():
            tensor_wrapper_name = GetSavedName(name)
            grad_api_args[
                grad_api_position] = f"egr::EagerUtils::RecoverTensorWrapper(&this->{tensor_wrapper_name}, nullptr)"

        for _, (ttype, fwd_position,
                grad_api_position) in backward_grad_inputs_map.items():
            if IsPlainTensorType(ttype):
                grad_api_args[
                    grad_api_position] = f"hooked_grads[{fwd_position}][0]"
            else:
                assert IsVectorTensorType(ttype)
                grad_api_args[
                    grad_api_position] = f"hooked_grads[{fwd_position}]"

        for name, _, _, grad_api_position in backward_attrs_list:
            saved_attribute_name = GetSavedName(name)
            grad_api_args[grad_api_position] = f"this->{saved_attribute_name}"
        grad_api_args_str = ", ".join(grad_api_args)

        # Construct grad_api returns
        num_bwd_outputs = len(backward_grad_outputs_map.keys())
        returns_str = f"std::vector<std::vector<paddle::experimental::Tensor>> returns({num_bwd_outputs});\n"
        for _, (ttype, fwd_position,
                grad_api_position) in backward_grad_outputs_map.items():
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
        returns_str += f"if(NeedComplexToRealConversion()) HandleComplexGradToRealGrad(&returns);\n"
        returns_str += f"return returns;\n"

        grad_node_name = GetGradNodeName(forward_api_name)

        fill_zero_str = ""
        if forward_api_name in ops_to_fill_zero_for_empty_grads:
            fill_zero_str = "egr::EagerUtils::FillZeroForEmptyGradInputs(&grads, this->InputMeta());\n"

        grad_api_namespace = f"paddle::experimental::{namespace}"

        self.node_definition_str = FUNCTION_TEMPLATE.format(
            grad_node_name, fill_zero_str, grad_node_name, grad_api_namespace,
            backward_api_name, grad_api_args_str, returns_str)

        print("Generated Node Definition: ", self.node_definition_str)

    def GenerateForwardDefinition(self, is_inplaced):
        namespace = self.namespace
        forward_api_name = GetInplacedFunctionName(
            self.forward_api_name) if is_inplaced else self.forward_api_name
        backward_api_name = self.backward_api_name
        forward_inputs_position_map = self.forward_inputs_position_map
        forward_outputs_position_map = self.forward_outputs_position_map
        forward_attrs_list = self.forward_attrs_list
        backward_forward_inputs_map = self.backward_forward_inputs_map
        backward_grad_inputs_map = self.backward_grad_inputs_map
        backward_grad_outputs_map = self.backward_grad_outputs_map
        backward_attrs_list = self.backward_attrs_list
        optional_inputs = self.optional_inputs
        intermediate_outputs = self.intermediate_outputs
        inplace_map = self.inplace_map

        # Get Function Args
        num_inputs = len(forward_attrs_list) + len(
            forward_inputs_position_map.keys())
        inputs_args_definition_list = ["" for i in range(num_inputs)]
        inputs_args_declaration_list = ["" for i in range(num_inputs)]
        inputs_call_list = ["" for i in range(num_inputs)]
        for name, (ttype, pos) in forward_inputs_position_map.items():
            inputs_call_list[pos] = f"{name}"
            is_optional = (name in optional_inputs)
            if IsPlainTensorType(ttype):
                if is_optional:
                    arg_str = f"const paddle::optional<paddle::experimental::Tensor>& {name}"
                else:
                    if inplace_map and name in inplace_map.keys():
                        arg_str = f"paddle::experimental::Tensor& {name}"
                    else:
                        arg_str = f"const paddle::experimental::Tensor& {name}"
            else:
                assert IsVectorTensorType(ttype)
                arg_str = f"const std::vector<paddle::experimental::Tensor>& {name}"

            inputs_args_definition_list[pos] = arg_str
            inputs_args_declaration_list[pos] = arg_str

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
        function_name = forward_api_name
        if len(intermediate_outputs) > 0:
            function_name = GetIntermediateAPIFunctionName(function_name)

        forward_call_str = f"auto api_result = paddle::experimental::{namespace}{function_name}({inputs_call_args_str});"

        # Get return type list & outputs
        num_outputs = len(forward_outputs_position_map.keys()) - len(
            intermediate_outputs)
        returns_type_list = ["" for i in range(num_outputs)]
        returns_list = ["" for i in range(num_outputs)]
        for name, (rtype, pos) in forward_outputs_position_map.items():
            if name in intermediate_outputs:
                continue
            if num_outputs == 1:
                returns_list[0] = f"api_result"
            else:
                # Tuple api_result
                returns_list[pos] = f"std::get<{pos}>(api_result)"

            if IsPlainTensorType(rtype):
                returns_type_list[pos] = "paddle::experimental::Tensor"
            else:
                assert IsVectorTensorType(rtype)
                returns_type_list[
                    pos] = "std::vector<paddle::experimental::Tensor>"

        if num_outputs == 1:
            returns_str = returns_list[0]
            returns_type_str = returns_type_list[0]
        else:
            returns_type_str = ", ".join(returns_type_list)
            returns_type_str = f"std::tuple<{returns_type_str}>"
            returns_str = ", ".join(returns_list)
            returns_str = f"std::make_tuple({returns_str})"

        self.GenerateNodeCreationCodes(forward_call_str)

        node_creation_str = self.node_creation_str
        dygraph_event_str = f"paddle::platform::RecordEvent dygraph_entrance_record_event(\"{forward_api_name} dygraph\", paddle::platform::TracerEventType::Operator, 1);"
        forward_function_name = GetDygraphForwardFunctionName(forward_api_name)

        self.forward_definition_str += FORWARD_FUNCTION_TEMPLATE.format(
            returns_type_str, forward_function_name, inputs_args_definition_str,
            dygraph_event_str, node_creation_str, returns_str)
        self.forward_declaration_str += f"{returns_type_str} {forward_function_name}({inputs_args_declaration_str});\n"

        print("Generated Forward Definition: ", self.forward_definition_str)
        print("Generated Forward Declaration: ", self.forward_declaration_str)

    def GenerateNodeCreationCodes(self, forward_call_str):
        forward_api_name = self.forward_api_name
        forward_inputs_position_map = self.forward_inputs_position_map
        forward_outputs_position_map = self.forward_outputs_position_map
        forward_attrs_list = self.forward_attrs_list
        backward_forward_inputs_map = self.backward_forward_inputs_map
        backward_grad_inputs_map = self.backward_grad_inputs_map
        backward_grad_outputs_map = self.backward_grad_outputs_map
        backward_attrs_list = self.backward_attrs_list
        optional_inputs = self.optional_inputs
        inplace_map = self.inplace_map

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
                    output_autograd_meta = f"    egr::AutogradMeta* {output_autograd_meta_name} = egr::EagerUtils::autograd_meta(&std::get<{pos}>(api_result));"
                else:
                    assert IsVectorTensorType(rtype)
                    output_autograd_meta = f"    std::vector<egr::AutogradMeta*> {output_autograd_meta_vec_name} = egr::EagerUtils::autograd_meta(&std::get<{pos}>(api_result));\n"
                    output_autograd_meta += f"    std::vector<egr::AutogradMeta*>* {output_autograd_meta_name} = &{output_autograd_meta_vec_name};"

            outputs_autograd_meta_list.append(output_autograd_meta)
            pass_stop_gradient_args_list.append(output_autograd_meta_name)

        # ComputeRequireGrad & PassStopGradient
        outputs_autograd_meta_str = "\n".join(outputs_autograd_meta_list)
        pass_stop_gradient_args_str = ",".join(pass_stop_gradient_args_list)

        # Check Inplace
        check_inplace_str = ""
        bump_inplace_version_str = ""
        for inplace_name in inplace_map.keys():
            inplace_autograd_meta_name = GetAutoGradMetaName(inplace_name)
            check_inplace_str += CHECK_INPLACE_TEMPLATE.format(
                inplace_name, inplace_autograd_meta_name)
            bump_inplace_version_str += BUMP_INPLACE_VERSION_TEMPLATE.format(
                inplace_name, inplace_name)

        # Node Construction
        num_backward_inputs = len(backward_grad_inputs_map.keys())
        num_backward_outputs = len(backward_grad_outputs_map.keys())
        grad_node_name = GetGradNodeName(forward_api_name)

        node_construction_str = f"            auto grad_node = std::make_shared<{grad_node_name}>({num_backward_inputs}, {num_backward_outputs});"

        # SetAttributes
        set_attributes_list = []
        forward_attrs_name_set = set()
        for name, _, _, _ in forward_attrs_list:
            forward_attrs_name_set.add(name)

        for name, _, default_val_attr, _ in backward_attrs_list:
            if name in forward_attrs_name_set:
                set_attributes = f"        grad_node->SetAttribute{name}({name});"
            else:
                set_attributes = f"        grad_node->SetAttribute{name}({default_val_attr});"
            set_attributes_list.append(set_attributes)
        set_attributes_str = "\n".join(set_attributes_list)

        # SetTensorWrappers
        set_tensor_wrappers_list = []
        for name, (atype, is_fwd_input,
                   pos) in backward_forward_inputs_map.items():
            is_optional = (name in optional_inputs)

            if is_fwd_input:
                if is_optional:
                    set_tensor_wrappers = f"        if({name}.is_initialized()) grad_node->SetTensorWrapper{name}({name}, true);"
                else:
                    set_tensor_wrappers = f"        grad_node->SetTensorWrapper{name}({name}, true);"
            else:
                if num_fwd_outputs > 1:
                    # Aligned with forward output position
                    assert name in forward_outputs_position_map.keys()
                    fwd_output_pos = forward_outputs_position_map[name][1]
                    tw_name = f"std::get<{fwd_output_pos}>(api_result)"
                else:
                    tw_name = f"api_result"

                if is_optional:
                    set_tensor_wrappers = f"        if({tw_name}.is_initialized()) grad_node->SetTensorWrapper{name}({tw_name}, false);"
                else:
                    set_tensor_wrappers = f"        grad_node->SetTensorWrapper{name}({tw_name}, false);"
            set_tensor_wrappers_list.append(set_tensor_wrappers)
        set_tensor_wrappers_str = "\n".join(set_tensor_wrappers_list)

        # SetGradOutMeta & SetEdges
        set_grad_out_meta_list = []
        set_edges_list = []
        for name, (_, pos) in forward_inputs_position_map.items():
            input_autograd_meta_name = GetAutoGradMetaName(name)
            set_grad_out_meta = f"        grad_node->SetGradOutMeta({name}, {pos});"
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

            if num_outputs == 1:
                set_retain_grad = f"        egr::EagerUtils::CheckAndRetainGrad(api_result);"
                set_grad_in_meta = f"       grad_node->SetGradInMeta(api_result, {pos});"
            else:
                set_retain_grad = f"            egr::EagerUtils::CheckAndRetainGrad(std::get<{pos}>(api_result));"
                set_grad_in_meta = f"            grad_node->SetGradInMeta(std::get<{pos}>(api_result), {pos});"
            set_out_rank_list.append(set_out_rank)
            set_history_list.append(set_history)
            set_grad_in_meta_list.append(set_grad_in_meta)
            set_retain_grad_list.append(set_retain_grad)

        set_out_rank_str = "\n".join(set_out_rank_list)
        set_history_str = "\n".join(set_history_list)
        set_grad_in_meta_str = "\n".join(set_grad_in_meta_list)
        set_retain_grad_str = "\n".join(set_retain_grad_list)

        node_event_name = forward_api_name + " node_creation"
        node_creation_event_str = f"paddle::platform::RecordEvent node_creation_record_event(\"{node_event_name}\", paddle::platform::TracerEventType::Operator, 1);\n"

        self.node_creation_str = NODE_CREATION_TEMPLATE.format(
            inputs_autograd_meta_str, compute_require_grad_args_str,
            check_inplace_str, forward_call_str, bump_inplace_version_str,
            node_creation_event_str, outputs_autograd_meta_str,
            pass_stop_gradient_args_str, node_construction_str,
            set_attributes_str, set_tensor_wrappers_str, set_grad_out_meta_str,
            set_edges_str, set_out_rank_str, set_history_str,
            set_grad_in_meta_str, set_retain_grad_str)

    def GenerateInplacedForwardDygraphFunctions(self):
        # Inplaced Version Dygraph Function Generation
        forward_api_name = self.forward_api_name
        forward_api_contents = self.forward_api_contents

        if forward_api_name != "sum" and "inplace" in forward_api_contents.keys(
        ):
            # Node Definition Generation
            self.GenerateForwardDefinition(is_inplaced=True)
            self.UpdateCoreOpsInformation(is_inplaced=True)

    def UpdateCoreOpsInformation(self, is_inplaced):
        forward_api_name = GetInplacedFunctionName(
            self.forward_api_name) if is_inplaced else self.forward_api_name
        forward_inputs_position_map = self.forward_inputs_position_map
        forward_outputs_position_map = self.forward_outputs_position_map
        forward_attrs_list = self.forward_attrs_list

        num_args = len(forward_inputs_position_map.keys()) + len(
            forward_attrs_list)
        num_returns = len(forward_outputs_position_map.keys())

        final_state_fwd_api_name = "final_state_" + forward_api_name
        core_ops_returns_info[
            final_state_fwd_api_name] = ["" for i in range(num_returns)]
        core_ops_args_info[
            final_state_fwd_api_name] = ["" for i in range(num_args)]
        core_ops_args_type_info[
            final_state_fwd_api_name] = ["" for i in range(num_args)]
        for name, (ttype, pos) in forward_inputs_position_map.items():
            core_ops_args_info[final_state_fwd_api_name][pos] = name
            if IsPlainTensorType(ttype):
                core_ops_args_type_info[final_state_fwd_api_name][
                    pos] = "tensor"
            else:
                assert IsVectorTensorType(ttype)
                core_ops_args_type_info[final_state_fwd_api_name][pos] = "list"

        for name, _, _, pos in forward_attrs_list:
            core_ops_args_info[final_state_fwd_api_name][pos] = name

        for name, (ttype, pos) in forward_outputs_position_map.items():
            core_ops_returns_info[final_state_fwd_api_name][pos] = name

    def run(self):
        # Basic Validation Check
        self.DygraphYamlValidationCheck()

        ##########################
        ## Parsing Raw Contents ##
        ##########################
        # Parse inplace_map
        self.ParseInplaceInfo()

        # Parse no_need_buffer
        self.ParseNoNeedBuffer()

        # Parse optional_inputs
        self.ParseDispensable()

        # Parse intermediate_outputs
        self.ParseIntermediate()
        self.IntermediateValidationCheck()

        # Initialize backward_forward_str, backward_inputs_list, backward_attrs_list, backward_returns_list
        self.CollectBackwardInfo()

        # Initialize forward_inputs_list, forward_attrs_list, forward_returns_list
        self.CollectForwardInfoFromBackwardContents()

        # Initialize orig_forward_inputs_list, orig_forward_attrs_list, orig_forward_returns_list
        self.CollectOriginalForwardInfo()

        # Forwards Validation Check
        self.ForwardsValidationCheck()

        #############################
        ## Process Parsed Contents ##
        #############################
        # Initialize forward_inputs_position_map, forward_outputs_position_map
        self.DetermineForwardPositionMap(self.forward_inputs_list,
                                         self.forward_returns_list)

        # Initialize forward_inputs_position_map, forward_outputs_position_map
        self.SlotNameMatching()

        # Backward Validation Check
        self.BackwardValidationCheck()

        #####################
        ## Code Generation ##
        #####################
        self.GenerateNodeDeclaration()
        self.GenerateNodeDefinition()
        self.GenerateForwardDefinition(is_inplaced=False)

        self.UpdateCoreOpsInformation(is_inplaced=False)

        self.GenerateInplacedForwardDygraphFunctions()


class DygraphYamlGenerator(YamlGeneratorBase):
    def __init__(self, api_yaml_path, backward_yaml_path):
        # Parent members: 
        # self.namespace
        # self.api_yaml_path
        # self.forward_api_list
        YamlGeneratorBase.__init__(self, api_yaml_path)

        self.backward_yaml_path = backward_yaml_path
        self.grad_api_dict = {}

        self.forward_definition_str = ""
        self.forward_declaration_str = ""
        self.node_declaration_str = ""
        self.node_definition_str = ""

    def ParseYamlContents(self):
        self.ParseForwardYamlContents()

        backward_yaml_path = self.backward_yaml_path
        self.grad_api_dict = ReadBwdFile(backward_yaml_path)

    def GetBackwardAPIContents(self, forward_api_contents):
        grad_api_dict = self.grad_api_dict

        if 'backward' not in forward_api_contents.keys(): return None

        backward_api_name = forward_api_contents['backward']
        assert backward_api_name in grad_api_dict.keys()
        backward_api_contents = grad_api_dict[backward_api_name]

        return backward_api_contents

    def GenerateCode(self):
        forward_api_list = self.forward_api_list
        grad_api_dict = self.grad_api_dict
        namespace = self.namespace

        for forward_api_contents in forward_api_list:
            backward_api_contents = self.GetBackwardAPIContents(
                forward_api_contents)
            if backward_api_contents is None: continue

            d_generator = DygraphSingleFunctionGenerator(
                forward_api_contents, backward_api_contents, namespace)
            d_generator.run()

            self.forward_definition_str += d_generator.forward_definition_str + "\n"
            self.forward_declaration_str += d_generator.forward_declaration_str + "\n"
            self.node_declaration_str += d_generator.node_declaration_str + "\n"
            self.node_definition_str += d_generator.node_definition_str + "\n"

        if len(namespace) > 0:
            if namespace.endswith("::"):
                namespace = namespace[:-2]
            self.forward_definition_str = NAMESPACE_WRAPPER_TEMPLATE.format(
                namespace, self.forward_definition_str)
            self.forward_declaration_str = NAMESPACE_WRAPPER_TEMPLATE.format(
                namespace, self.forward_declaration_str)
            self.node_declaration_str = NAMESPACE_WRAPPER_TEMPLATE.format(
                namespace, self.node_declaration_str)
            self.node_definition_str = NAMESPACE_WRAPPER_TEMPLATE.format(
                namespace, self.node_definition_str)

    def run(self):
        self.ParseYamlContents()

        self.InferNameSpace()

        self.GenerateCode()


##################
## File Writers ##
##################
def GenerateNodeCCFile(filepath, node_definition_str):
    if os.path.exists(filepath):
        os.remove(filepath)

    file_contents = NODE_CC_FILE_TEMPLATE.format(node_definition_str)
    with open(filepath, 'a') as f:
        f.write(file_contents)


def GenerateNodeHFile(filepath, node_declaration_str):
    if os.path.exists(filepath):
        os.remove(filepath)

    file_contents = NODE_H_FILE_TEMPLATE.format(node_declaration_str)
    with open(filepath, 'a') as f:
        f.write(file_contents)


def GenerateForwardCCFile(filepath, forward_definition_str):
    if os.path.exists(filepath):
        os.remove(filepath)

    core_ops_info_str = GenerateCoreOpInfoDefinition()
    file_contents = FORWARD_CC_FILE_TEMPLATE.format(core_ops_info_str,
                                                    forward_definition_str)
    with open(filepath, 'a') as f:
        f.write(file_contents)


def GenerateForwardHFile(filepath, forward_function_declaration_str):
    if os.path.exists(filepath):
        os.remove(filepath)

    core_ops_info_str = GenerateCoreOpInfoDeclaration()
    file_contents = FORWARD_H_FILE_TEMPLATE.format(
        core_ops_info_str, forward_function_declaration_str)
    with open(filepath, 'a') as f:
        f.write(file_contents)


if __name__ == "__main__":
    args = ParseArguments()

    api_yaml_paths = args.api_yaml_path.split(",")
    backward_yaml_paths = args.backward_yaml_path.split(",")

    # Generate per Dygraph API
    node_declaration_str = ""
    node_definition_str = ""
    forward_definition_str = ""
    forward_declaration_str = ""

    for i in range(len(api_yaml_paths)):
        api_yaml_path = api_yaml_paths[i]
        backward_yaml_path = backward_yaml_paths[i]

        generator = DygraphYamlGenerator(api_yaml_path, backward_yaml_path)
        generator.run()

        node_declaration_str += generator.node_declaration_str + "\n"
        node_definition_str += generator.node_definition_str + "\n"
        forward_definition_str += generator.forward_definition_str + "\n"
        forward_declaration_str += generator.forward_declaration_str + "\n"

    # Generate Files
    nodes_h_path = args.nodes_h_path
    nodes_cc_path = args.nodes_cc_path
    forwards_h_path = args.forwards_h_path
    forwards_cc_path = args.forwards_cc_path

    GenerateNodeCCFile(nodes_cc_path, node_definition_str)
    GenerateNodeHFile(nodes_h_path, node_declaration_str)
    GenerateForwardCCFile(forwards_cc_path, forward_definition_str)
    GenerateForwardHFile(forwards_h_path, forward_declaration_str)
