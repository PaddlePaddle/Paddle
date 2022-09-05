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
import logging
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
from codegen_utils import ParseYamlInplaceInfo
from codegen_utils import FunctionGeneratorBase, GeneratorBase
from codegen_utils import ops_to_fill_zero_for_empty_grads
from codegen_utils import AssertMessage, GetIndent

# Note: assign is a inplace api when parameter(output) isn't none,
# so we should check parameter(output) with rule of inplace.
# But because there is no check in old dygraph mode, in order to
# keeping the code compatible, here we also skip inplace check in new dygraph temporarily,
# and this will be fixed in the futrue.
inplace_check_blacklist = set(["assign_out_"])

# Black Ops list that's NO NEED to apply code generation
black_ops_list = [
    "conv2d", "conv2d_grad", "conv2d_grad_grad", "add_n", "add_n_grad"
]


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
"""  void SetTensorWrapper{}(const paddle::experimental::Tensor& {}) {{
    {} = egr::TensorWrapper({}, {});
  }}
"""

SET_VECTOR_TENSOR_WRAPPER_TEMPLATE = \
"""  void SetTensorWrapper{}(const std::vector<paddle::experimental::Tensor>& {}) {{
    for(const auto& eager_tensor : {}) {{
      {}.emplace_back(egr::TensorWrapper(eager_tensor, {}));
    }};
  }}
"""

PLAIN_TENSOR_MEMBER_TEMPLATE = \
"""  egr::TensorWrapper {};
"""

VECTOR_TENSOR_MEMBER_TEMPLATE = \
"""  std::vector<egr::TensorWrapper> {};
"""

CLEAR_TENSOR_WRAPPER_TEMPLATE = \
"""    {}.clear();
"""

CLEAR_VECTOR_TENSOR_WRAPPERS_TEMPLATE = \
"""    for (auto& tw : {}) {{
      tw.clear();
    }}
"""

SET_ATTR_METHOD_TEMPLATE = \
"""  void SetAttribute{}({} {}) {{
    {} = {};
  }}
"""

ATTRIBUTE_MEMBER_WITH_DEFAULT_TEMPLATE = \
"""  {} {} = {};
"""

ATTRIBUTE_MEMBER_TEMPLATE = \
"""  {} {};
"""

NODE_DECLARATION_TEMPLATE = \
"""
class {} : public egr::GradNodeBase {{
 public:
  {}() : egr::GradNodeBase() {{}}
  {}(size_t bwd_in_slot_num, size_t bwd_out_slot_num) :
      egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {{}}
  ~{}() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>, egr::kSlotSmallVectorSize> operator()(
      paddle::small_vector<std::vector<paddle::experimental::Tensor>, egr::kSlotSmallVectorSize>& grads, bool create_graph = false, bool is_new_grad = false) override;
  std::string name() override {{ return \"{}\"; }}

  void ClearTensorWrappers() override {{
{}
    SetIsTensorWrappersCleared(true);
  }}

  std::shared_ptr<GradNodeBase> Copy() const override {{
    auto copied_node = std::shared_ptr<{}>(new {}(*this));
    return copied_node;
  }}

  // SetTensorWrapperX, SetTensorWrapperY, ...
{}
  // SetAttributes
{}
 private:
  // TensorWrappers
{}
  // Attributes
{}}};
"""

GRAD_FUNCTION_TEMPLATE = \
"""
paddle::small_vector<std::vector<paddle::experimental::Tensor>, egr::kSlotSmallVectorSize> {}::operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>, egr::kSlotSmallVectorSize>& grads, bool create_graph, bool is_new_grad) {{
  // Fill Zero For GradIn Tensors
{}
  // Apply Gradient Hooks
  auto hooked_grads = ApplyGradientHooks(grads);

  // Collect GradIn Tensors, Attrs and Recovered TensorWrappers
{}
  // Prepare Grad function call
{}
  // Runtime check if we need next grad
{}
  // Inplace Check
{}
  // Inplace Strategy
{}
  // Call grad_api function
  VLOG(3) << \"Final State Running: {}\";
{}
  // Check NaN and Inf id needed
{}
  // Get GradOut autograd_meta
{}
  // Create Grad Node
{}
  // Return
{}
}}
"""

FORWARD_FUNCTION_TEMPLATE = \
"""
{} {}({}) {{
  // Dygraph Record Event
{}
  // AMP Logic
{}
  // Layout autotune
{}
  // Get Input AutoGradMeta
{}
  // Forward API Call
  VLOG(3) << \"Final State Running: \" << \"{}\";
{}
  // Check NaN and Inf if needed
{}
  // Get Outputs
{}
  // Get Output AutoGradMeta
{}
  bool trace_backward = egr::Controller::Instance().HasGrad();
  bool require_any_grad = egr::EagerUtils::ComputeRequireGrad({});

  // Check Inplace if needed
{}{}
  // Node Creation
{}
  // Returns
  return {};
}}
"""


FORWARD_ONLY_FUNCTION_TEMPLATE = \
"""
{} {}({}) {{
  // Dygraph Record Event
{}
  // AMP Logic
{}
  // Layout autotune
{}
  // Forward API Call
  VLOG(3) << \"Final State Running: \" << \"{}\";
{}
  // Get Outputs
{}

  // Returns
  return {};
}}
"""

FORWARD_BODY_TEMPLATE = \
"""  if(require_any_grad) {{
{}
    egr::EagerUtils::PassStopGradient({});

    // Node Construction
{}
    // SetAttributes if needed
{}
    // Set TensorWrappers for Forward Inputs if needed
{}
    // SetGradOutMeta & SetEdges
{}
    // SetOutRank & SetHistory & SetGradInMeta & RetainGrad
{}
{}
{}
{}
    // Set TensorWrappers for Forward Outputs if needed
{}
  }}
"""

HIHGER_ORDER_DERIVATIVE_VALUE_TEMPLATE = \
"""  if(trace_backward) {{
{}
    // Node Construction
{}
    // SetAttributes if needed
{}
    // Set TensorWrappers for Forward Inputs if needed
{}
    // SetGradOutMeta & SetEdges
{}
    // SetOutRank & SetHistory & SetGradInMeta & RetainGrad
{}
{}
{}
{}
    // Set TensorWrappers for Forward Outputs if needed
{}
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
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/api/generated/eager_generated/backwards/nodes.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/to_static/run_program_op_node.h"
#include "paddle/fluid/eager/nan_inf_utils.h"
#include "paddle/phi/api/include/sparse_api.h"
#include "paddle/fluid/eager/api/manual/eager_manual/nodes/nodes.h"
DECLARE_bool(check_nan_inf);
{}
"""

NODE_H_FILE_TEMPLATE = \
"""
#pragma once
#include "paddle/fluid/eager/tensor_wrapper.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/api/manual/eager_manual/nodes/nodes.h"

{}
"""

FORWARD_CC_FILE_TEMPLATE = \
"""
#include "paddle/phi/api/lib/dygraph_api.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/api/generated/eager_generated/backwards/nodes.h"
#include "paddle/fluid/eager/eager_layout_auto_tune.h"
#include "paddle/phi/api/include/strings_api.h"
#include "paddle/phi/api/include/sparse_api.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/fluid/eager/amp_utils.h"
#include "paddle/fluid/eager/eager_amp_auto_cast.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/fluid/eager/nan_inf_utils.h"
#include "paddle/fluid/eager/api/manual/eager_manual/dygraph_forward_api.h"
DECLARE_bool(check_nan_inf);
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
#include "paddle/fluid/eager/api/manual/eager_manual/dygraph_forward_api.h"

using CPUPlace = phi::CPUPlace;
{}
{}
"""

CORE_OPS_INFO_TEMPLATE = \
"""
std::unordered_map<std::string, std::vector<std::string>> core_ops_args_info = {{
    {}
}};
std::unordered_map<std::string, std::vector<std::string>> core_ops_args_type_info = {{
    {}
}};
std::unordered_map<std::string, std::vector<std::string>> core_ops_returns_info = {{
    {}
}};

"""

CORE_OPS_DECLARATION_TEMPLATE = \
"""
extern std::unordered_map<std::string, std::vector<std::string>> core_ops_args_info;
extern std::unordered_map<std::string, std::vector<std::string>> core_ops_args_type_info;
extern std::unordered_map<std::string, std::vector<std::string>> core_ops_returns_info;

"""

CHECK_INPLACE_TEMPLATE = \
"""
  egr::EagerUtils::CheckInplace({}, {}, require_any_grad);
"""

BUMP_INPLACE_VERSION_TEMPLATE = \
"""
  // Bump Inplace Version
  {}.bump_inplace_version();
  VLOG(3) << \"Tensor(\" << {}.name() << \") uses Inplace Strategy.\";
"""

AMP_LOGIC_TEMPLATE = \
"""  if (egr::Controller::Instance().GetAMPLevel() != paddle::imperative::AmpLevel::O0) {{
    VLOG(5) << "Check and Prepare For AMP";
    {}
    paddle::small_vector<std::vector<paddle::experimental::Tensor>, egr::kSlotSmallVectorSize> amp_tensors_vector = {};
    {}
    {}
    {}
    {{
      paddle::imperative::AutoCastGuard guard(egr::Controller::Instance().GetCurrentTracer(), paddle::imperative::AmpLevel::O0);
      {}
    }}
  }}
"""
LAYOUT_LOGIC_TEMPLATE=\
"""
  if (paddle::imperative::LayoutAutoTune::Instance().UseLayoutAutoTune()) {{
    VLOG(5) << "Check and Prepare For LAYOUT";
    paddle::small_vector<std::vector<paddle::experimental::Tensor>, egr::kSlotSmallVectorSize> tensors_vector = {};
    {} 
    {}
    paddle::imperative::LayoutAutoTune::Instance().DisableLayoutAutoTune(); 
    {}
    {}
    paddle::imperative::LayoutAutoTune::Instance().EnableLayoutAutoTune();
    // Returns
    return {};
  }}
"""
CREATE_PLAIN_OPTIONAL_TENSOR_TEMPLATE = \
"""
  paddle::optional<paddle::experimental::Tensor> {}_optional;
  if({}.initialized()) {}_optional = paddle::make_optional<paddle::experimental::Tensor>({});
"""

CREATE_RECOVER_OPTIONAL_TENSOR_TEMPLATE = \
"""
  paddle::optional<paddle::experimental::Tensor> {}_optional;
  if( {}.impl() ) {}_optional = paddle::make_optional<paddle::experimental::Tensor>({});
"""

CREATE_RECOVER_OPTIONAL_VECTOR_TENSOR_TEMPLATE = \
"""
  paddle::optional<std::vector<paddle::experimental::Tensor>> {}_optional;
  if( !{}.empty() ) {}_optional = paddle::make_optional<std::vector<paddle::experimental::Tensor>>({});
"""

CHECK_BACKWARD_INPLACE_TEMPLATE = \
"""
  bool can_be_inplaced = false;
  if ({}.initialized()) {{
    VLOG(10) << {}.name() << "({}) use_count: " << {}.impl().use_count();
    if ({}.impl().use_count() == 1 || ({}.impl().use_count() == 2 && {}.impl().get() == {}.impl().get())) {{
      can_be_inplaced = true;
    }}
  }}"""

CHECK_NAN_AND_INF_TEMPLATE = \
"""  if (FLAGS_check_nan_inf) {{ egr::CheckTensorHasNanOrInf("{}", {}); }}
"""

inplace_optional_out_type_map = {
    "Tensor":
    "paddle::optional<paddle::experimental::Tensor>&",
    "std::vector<Tensor>":
    "paddle::optional<std::vector<paddle::experimental::Tensor>>&"
}


def ExtractForwardApiNameFormInvoke(invoke_config):
    api_name = invoke_config.split('(')[0]
    if api_name[-1] == '_':
        api_name = api_name[:-1]
    return re.search(
        r"(?P<api_name>[a-zA-Z0-9_]+)(?P<intermediate>_intermediate)?",
        api_name).group('api_name')


def IsInvokeForwardApi(api_contents, forward_api_name_list):
    return 'invoke' in api_contents and ExtractForwardApiNameFormInvoke(
        api_contents['invoke']) in forward_api_name_list


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
class DygraphFunctionGeneratorBase(FunctionGeneratorBase):

    def __init__(self, forward_api_contents, grad_api_contents,
                 forward_apis_dict, namespace):
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
        #self.forward_inplace_map
        FunctionGeneratorBase.__init__(self, forward_api_contents, namespace)

        self.forward_apis_dict = forward_apis_dict
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

        self.backward_attrs_list = [
        ]  #[ [attr_name, attr_type, default_value, orig_position], ...]
        self.backward_inputs_list = [
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

        self.backward_inplace_map = {}  #{name : name, ...}

    def ParseBackwardInplaceInfo(self):
        grad_api_contents = self.grad_api_contents
        if 'inplace' not in grad_api_contents.keys(): return

        inplace_map_str = grad_api_contents['inplace']
        self.backward_inplace_map = ParseYamlInplaceInfo(inplace_map_str)

    def DygraphYamlValidationCheck(self):
        forward_api_contents = self.forward_api_contents
        grad_api_contents = self.grad_api_contents

        assert 'api' in forward_api_contents.keys(
        ), "Unable to find \"api\" in api.yaml"
        assert 'args' in forward_api_contents.keys(
        ), "Unable to find \"args\" in api.yaml"
        assert 'output' in forward_api_contents.keys(
        ), "Unable to find \"output\" in api.yaml"

        if grad_api_contents is not None:
            assert 'backward' in forward_api_contents.keys(
            ), "Unable to find \"backward\" in api.yaml"
            assert 'args' in grad_api_contents.keys(
            ), "Unable to find \"args\" in backward.yaml"
            assert 'output' in grad_api_contents.keys(
            ), "Unable to find \"output\" in backward.yaml"
            assert 'forward' in grad_api_contents.keys(
            ), "Unable to find \"forward\" in backward.yaml"

    def ForwardsValidationCheck(self):
        forward_inputs_list = self.forward_inputs_list
        forward_attrs_list = self.forward_attrs_list
        forward_returns_list = self.forward_returns_list

        orig_forward_inputs_list = self.orig_forward_inputs_list
        orig_forward_attrs_list = self.orig_forward_attrs_list
        orig_forward_returns_list = self.orig_forward_returns_list

        for i in range(len(forward_inputs_list)):
            forward_input_type = forward_inputs_list[i][1]
            forward_input_pos = forward_inputs_list[i][2]
            orig_input_type = orig_forward_inputs_list[i][1]
            orig_input_pos = orig_forward_inputs_list[i][2]

            assert forward_input_type == orig_input_type, AssertMessage(
                forward_input_type, orig_input_type)
            assert forward_input_pos == orig_input_pos, AssertMessage(
                forward_input_pos, orig_input_pos)

        for i in range(len(forward_attrs_list)):
            orig_attr_type = orig_forward_attrs_list[i][1]
            orig_attr_pos = orig_forward_attrs_list[i][3]
            forward_attr_type = forward_attrs_list[i][1]
            forward_attr_pos = forward_attrs_list[i][3]
            assert orig_attr_type == forward_attr_type, AssertMessage(
                orig_attr_type, forward_attr_type)
            assert orig_attr_pos == forward_attr_pos, AssertMessage(
                orig_attr_pos, forward_attr_pos)

        for i in range(len(forward_returns_list)):
            orig_return_type = orig_forward_returns_list[i][1]
            orig_return_pos = orig_forward_returns_list[i][2]
            forward_return_type = forward_returns_list[i][1]
            forward_return_pos = forward_returns_list[i][2]

            assert orig_return_type == forward_return_type, AssertMessage(
                orig_return_type, forward_return_type)
            assert orig_return_pos == forward_return_pos, AssertMessage(
                orig_return_pos, forward_return_pos)

        # Check Order: Inputs, Attributes
        max_input_position = -1
        for _, _, pos in forward_inputs_list:
            max_input_position = max(max_input_position, pos)

        for _, _, _, pos in forward_attrs_list:
            assert pos > max_input_position, AssertMessage(
                pos, max_input_position)

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
            assert pos > max_fwd_input_position, AssertMessage(
                pos, max_grad_tensor_position)
            max_grad_tensor_position = max(max_grad_tensor_position, pos)

        max_attr_position = -1
        for _, _, _, pos in backward_attrs_list:
            assert pos > max_grad_tensor_position, AssertMessage(
                pos, max_grad_tensor_position)
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
                assert pos in intermediate_positions, AssertMessage(
                    pos, intermediate_positions)

    def CollectBackwardInfo(self):
        forward_api_contents = self.forward_api_contents
        grad_api_contents = self.grad_api_contents

        self.backward_api_name = forward_api_contents['backward']
        self.backward_forward_str = grad_api_contents['forward']
        backward_args_str = grad_api_contents['args']
        backward_returns_str = grad_api_contents['output']

        self.backward_inputs_list, self.backward_attrs_list, self.backward_returns_list = ParseYamlBackward(
            backward_args_str, backward_returns_str)

    def CollectForwardInfoFromBackwardContents(self):

        backward_forward_str = self.backward_forward_str

        self.forward_inputs_list, self.forward_attrs_list, self.forward_returns_list = ParseYamlForwardFromBackward(
            backward_forward_str)

    def CollectForwardInfoFromYamlForward(self):
        self.forward_inputs_list, self.forward_attrs_list, self.forward_returns_list = ParseYamlForwardFromBackward(
            self.forward_api_contents['args'] + " -> " +
            self.forward_api_contents['output'])

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
                assert backward_fwd_name in forward_outputs_position_map.keys(
                ), AssertMessage(backward_fwd_name,
                                 forward_outputs_position_map.keys())
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
                    assert False, f"Cannot find {backward_input_name} in forward position map"

        for backward_output in backward_returns_list:
            backward_output_name = backward_output[0]
            backward_output_type = backward_output[1]
            backward_output_pos = backward_output[2]

            backward_fwd_name = FindForwardName(backward_output_name)
            assert backward_fwd_name is not None, f"Detected {backward_fwd_name} = None"
            assert backward_fwd_name in forward_inputs_position_map.keys(
            ), AssertMessage(backward_fwd_name,
                             forward_inputs_position_map.keys())

            matched_forward_input_type = forward_inputs_position_map[
                backward_fwd_name][0]
            matched_forward_input_pos = forward_inputs_position_map[
                backward_fwd_name][1]

            self.backward_grad_outputs_map[backward_output_name] = [
                backward_output_type, matched_forward_input_pos,
                backward_output_pos
            ]

    def GetPassStopGradientArgsList(self, forward_outputs_position_map):
        pass_stop_gradient_args_list = ["false"]
        for name, (_, _) in forward_outputs_position_map.items():
            output_autograd_meta_name = GetAutoGradMetaName(name)
            pass_stop_gradient_args_list.append(output_autograd_meta_name)
        pass_stop_gradient_args_str = ",".join(pass_stop_gradient_args_list)
        return pass_stop_gradient_args_str

    def GenerateNodeCreationCodes(self, for_backward=False):
        forward_api_name = self.forward_api_name
        forward_inputs_position_map = self.forward_inputs_position_map
        forward_outputs_position_map = self.forward_outputs_position_map
        forward_attrs_list = self.forward_attrs_list
        backward_forward_inputs_map = self.backward_forward_inputs_map
        backward_grad_inputs_map = self.backward_grad_inputs_map
        backward_grad_outputs_map = self.backward_grad_outputs_map
        backward_attrs_list = self.backward_attrs_list
        optional_inputs = self.optional_inputs

        # Pass Stop Gradient Args
        pass_stop_gradient_args_str = self.GetPassStopGradientArgsList(
            forward_outputs_position_map)

        # Node Construction
        num_backward_inputs = len(forward_outputs_position_map.keys())
        num_backward_outputs = len(forward_inputs_position_map.keys())
        grad_node_name = GetGradNodeName(self.backward_api_name)

        # Helper
        indent = GetIndent(2)
        # NOTE(Aurelius74): DO NOT use make_shared here. Because some Node contains experimental::Scalar
        # which contains "complex128" as data. "complex128" is memory-aligned manually. But make_shared
        # request MEMALIGN for allocation (Maybe).
        # See https://stackoverflow.com/questions/31228656/how-can-shared-ptr-disrupt-alignment
        # and https://github.com/MRtrix3/mrtrix3/issues/957
        node_construction_str = f"{indent}auto grad_node = std::shared_ptr<{grad_node_name}>(new {grad_node_name}({num_backward_inputs}, {num_backward_outputs}));"

        # SetAttributes
        set_attributes_list = []
        forward_attrs_name_set = set()
        for name, _, _, _ in forward_attrs_list:
            forward_attrs_name_set.add(name)

        for name, _, default_val_attr, _ in backward_attrs_list:
            if name in forward_attrs_name_set:
                set_attributes = f"{indent}grad_node->SetAttribute{name}({name});"
            else:
                set_attributes = f"{indent}grad_node->SetAttribute{name}({default_val_attr});"
            set_attributes_list.append(set_attributes)
        set_attributes_str = "\n".join(set_attributes_list)

        # SetTensorWrappers
        set_input_tensor_wrappers_list = []
        set_output_tensor_wrappers_list = []
        num_fwd_outputs = len(forward_outputs_position_map.keys())
        for name, (atype, is_fwd_input,
                   pos) in backward_forward_inputs_map.items():
            is_optional = (name in optional_inputs)

            if is_fwd_input:
                if is_optional:
                    set_tensor_wrappers = f"{indent}if({name}) grad_node->SetTensorWrapper{name}(*{name});"
                else:
                    set_tensor_wrappers = f"{indent}grad_node->SetTensorWrapper{name}({name});"
                set_input_tensor_wrappers_list.append(set_tensor_wrappers)
            else:  # Forwad's output as backward's input
                if num_fwd_outputs > 1:
                    # Aligned with forward output position
                    assert name in forward_outputs_position_map.keys(
                    ), AssertMessage(name, forward_outputs_position_map.keys())

                if is_optional:
                    set_tensor_wrappers = f"{indent}if({name}) grad_node->SetTensorWrapper{name}(*{name});"
                else:
                    set_tensor_wrappers = f"{indent}grad_node->SetTensorWrapper{name}({name});"
                set_output_tensor_wrappers_list.append(set_tensor_wrappers)
        set_input_tensor_wrappers_str = "\n".join(
            set_input_tensor_wrappers_list)
        set_output_tensor_wrappers_str = "\n".join(
            set_output_tensor_wrappers_list)

        # SetGradOutMeta & SetEdges
        grad_node_out_list = []
        set_grad_out_meta_list = []
        set_edges_list = []
        for name, (_, pos) in forward_inputs_position_map.items():
            # Has corresponding grad output
            has_corresponding_grad_output = False
            for _, (_, corresponding_pos,
                    _) in backward_grad_outputs_map.items():
                if pos == corresponding_pos:
                    has_corresponding_grad_output = True
            if not has_corresponding_grad_output:
                continue

            grad_node_out_list.append(name)
            is_optional = (name in self.optional_inputs)
            if is_optional:
                set_grad_out_meta = f"{indent}if({name}.get_ptr() != nullptr) grad_node->SetGradOutMeta(*({name}.get_ptr()), {pos});"
            else:
                set_grad_out_meta = f"{indent}grad_node->SetGradOutMeta({name}, {pos});"

            set_grad_out_meta_list.append(set_grad_out_meta)
        set_grad_out_meta_str = "\n".join(set_grad_out_meta_list)

        # SetOutRank & SetHistory & SetGradInMeta & CheckAndRetainGrad
        set_out_rank_list = []
        set_history_list = []
        set_grad_in_meta_list = []
        set_retain_grad_list = []
        num_outputs = len(forward_outputs_position_map.keys())
        for name, (_, pos) in forward_outputs_position_map.items():
            output_autograd_meta_name = GetAutoGradMetaName(name)
            set_out_rank = f"""{indent}if ({output_autograd_meta_name}) {{
{indent}  egr::EagerUtils::SetOutRankWithSlot({output_autograd_meta_name}, {pos});
{indent}}}"""

            set_history = f"""{indent}if ({output_autograd_meta_name}) {{
{indent}  egr::EagerUtils::SetHistory({output_autograd_meta_name}, grad_node);
{indent}}}"""

            set_grad_in_meta = f"{indent}grad_node->SetGradInMeta({name}, {pos});"
            set_retain_grad = f"{indent}egr::EagerUtils::CheckAndRetainGrad({name});"

            set_out_rank_list.append(set_out_rank)
            set_history_list.append(set_history)
            set_grad_in_meta_list.append(set_grad_in_meta)
            set_retain_grad_list.append(set_retain_grad)

        set_out_rank_str = "\n".join(set_out_rank_list)
        set_history_str = "\n".join(set_history_list)
        set_grad_in_meta_str = "\n".join(set_grad_in_meta_list)
        set_retain_grad_str = "\n".join(set_retain_grad_list)

        node_event_name = forward_api_name + " node_creation"
        node_creation_event_str = f"{indent}paddle::platform::RecordEvent node_creation_record_event(\"{node_event_name}\", paddle::platform::TracerEventType::OperatorInner, 1);\n"
        if not for_backward:
            self.node_creation_str = FORWARD_BODY_TEMPLATE.format(
                node_creation_event_str, pass_stop_gradient_args_str,
                node_construction_str, set_attributes_str,
                set_input_tensor_wrappers_str, set_grad_out_meta_str,
                set_out_rank_str, set_history_str, set_grad_in_meta_str,
                set_retain_grad_str, set_output_tensor_wrappers_str)
        else:
            self.node_creation_str = HIHGER_ORDER_DERIVATIVE_VALUE_TEMPLATE.format(
                node_creation_event_str, node_construction_str,
                set_attributes_str, set_input_tensor_wrappers_str,
                set_grad_out_meta_str, set_out_rank_str, set_history_str,
                set_grad_in_meta_str, set_retain_grad_str,
                set_output_tensor_wrappers_str)

        self.grad_node_out_list = grad_node_out_list

    def run(self):
        # Basic Validation Check
        self.DygraphYamlValidationCheck()

        ##########################
        ## Parsing Raw Contents ##
        ##########################
        # Parse forward and backward inplace_map
        self.ParseForwardInplaceInfo()
        if self.grad_api_contents is not None:
            self.ParseBackwardInplaceInfo()
            # Parse no_need_buffer
            self.ParseNoNeedBuffer()

        # Parse optional_inputs
        self.ParseDispensable()

        # Parse intermediate_outputs
        self.ParseIntermediate()
        self.IntermediateValidationCheck()

        if self.grad_api_contents is not None:
            # Initialize backward_forward_str, backward_inputs_list, backward_attrs_list, backward_returns_list
            self.CollectBackwardInfo()

            # Initialize forward_inputs_list, forward_attrs_list, forward_returns_list
            self.CollectForwardInfoFromBackwardContents()

        if self.is_forward_only:
            self.CollectForwardInfoFromYamlForward()

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

        if self.grad_api_contents is not None:
            # Initialize backward_forward_inputs_map, backward_grad_inputs_map, backward_grad_outputs_map
            self.SlotNameMatching()
            # Backward Validation Check
            self.BackwardValidationCheck()


class DygraphForwardFunctionGenerator(DygraphFunctionGeneratorBase):

    def __init__(self, forward_api_contents, grad_api_contents,
                 forward_apis_dict, namespace):
        DygraphFunctionGeneratorBase.__init__(self, forward_api_contents,
                                              grad_api_contents,
                                              forward_apis_dict, namespace)

        # Generated Results
        self.forward_definition_str = ""
        self.forward_declaration_str = ""

    def GenerateForwardDefinitionAndDeclaration(self, is_inplaced):
        namespace = self.namespace
        if self.forward_api_name[-1] == '_' and not is_inplaced:
            return
        forward_api_name = GetInplacedFunctionName(
            self.forward_api_name) if is_inplaced else self.forward_api_name

        forward_inputs_position_map = self.forward_inputs_position_map
        forward_outputs_position_map = self.forward_outputs_position_map
        forward_attrs_list = self.forward_attrs_list
        if not self.is_forward_only:
            backward_grad_outputs_map = self.backward_grad_outputs_map

        optional_inputs = self.optional_inputs
        intermediate_outputs = self.intermediate_outputs
        forward_inplace_map = self.forward_inplace_map if is_inplaced else {}
        indent = GetIndent(1)

        # Get Function Args
        num_inputs = len(forward_attrs_list) + len(
            forward_inputs_position_map.keys())
        inputs_args_definition_list = ["" for i in range(num_inputs)]
        inputs_args_declaration_list = ["" for i in range(num_inputs)]
        inputs_call_list = ["" for i in range(num_inputs)]

        amp_inputs_call_list = ["" for i in range(num_inputs)]
        amp_tensors_vector_list = []
        amp_tensors_vector_optional_list = []
        amp_autocast_list = []
        amp_autocast_optional_list = []
        layout_autotune_list = []
        layout_autotune_optional_list = []
        layout_tensors_vector_optional_list = []
        for name, (ttype, pos) in forward_inputs_position_map.items():
            inputs_call_list[pos] = f"{name}"
            amp_inputs_call_list[pos] = f"NEW_{name}"
            is_optional = (name in optional_inputs)
            if IsPlainTensorType(ttype):
                if is_optional:
                    if self.is_forward_only and is_inplaced and forward_inplace_map and name in forward_inplace_map.keys(
                    ):
                        arg_str = f"paddle::optional<paddle::experimental::Tensor>& {name}"
                    else:
                        arg_str = f"const paddle::optional<paddle::experimental::Tensor>& {name}"
                    amp_tensors_vector_optional_list.append(
                        f"if ({name}) amp_tensors_vector.push_back({{ *{name} }});\n"
                    )
                    amp_autocast_optional_list.append(
                        f"auto NEW_{name} = egr::EagerAmpAutoCast(\"{name}\", {name}, amp_dst_dtype, op_name);\n"
                    )
                    layout_tensors_vector_optional_list.append(
                        f"if ({name}) tensors_vector.push_back({{ *{name} }});\n"
                    )
                    layout_autotune_optional_list.append(
                        f"auto NEW_{name} = transformer->TransInTensor(\"{name}\", {name});\n"
                    )
                else:
                    if is_inplaced and forward_inplace_map and name in forward_inplace_map.keys(
                    ):
                        arg_str = f"paddle::experimental::Tensor& {name}"
                        amp_tensors_vector_list.append(f"{{{name}}}")
                        amp_autocast_list.append(
                            f"auto NEW_{name} = egr::EagerAmpAutoCast(\"{name}\", {name}, amp_dst_dtype, op_name);\n"
                        )
                    else:
                        arg_str = f"const paddle::experimental::Tensor& {name}"
                        amp_tensors_vector_list.append(f"{{{name}}}")
                        amp_autocast_list.append(
                            f"auto NEW_{name} = egr::EagerAmpAutoCast(\"{name}\", {name}, amp_dst_dtype, op_name);\n"
                        )
                    layout_autotune_list.append(
                        f"auto NEW_{name} = transformer->TransInTensor(\"{name}\", {name});\n"
                    )
            else:
                assert IsVectorTensorType(ttype)
                if is_optional:
                    if self.is_forward_only and is_inplaced and forward_inplace_map and name in forward_inplace_map.keys(
                    ):
                        arg_str = f"paddle::optional<std::vector<paddle::experimental::Tensor>>& {name}"
                    else:
                        arg_str = f"const paddle::optional<std::vector<paddle::experimental::Tensor>>& {name}"
                    amp_tensors_vector_optional_list.append(
                        f"if ({name}) amp_tensors_vector.push_back( *{name} );\n"
                    )
                    amp_autocast_optional_list.append(
                        f"auto NEW_{name} = egr::EagerAmpAutoCasts(\"{name}\", {name}, amp_dst_dtype, op_name);\n"
                    )
                    layout_autotune_optional_list.append(
                        f"auto NEW_{name} = transformer->TransInTensor(\"{name}\", {name});\n"
                    )
                else:
                    if is_inplaced and forward_inplace_map and name in forward_inplace_map.keys(
                    ):
                        arg_str = f"std::vector<paddle::experimental::Tensor>& {name}"
                    else:
                        arg_str = f"const std::vector<paddle::experimental::Tensor>& {name}"
                    amp_tensors_vector_list.append(f"{name}")
                    amp_autocast_list.append(
                        f"auto NEW_{name} = egr::EagerAmpAutoCasts(\"{name}\", {name}, amp_dst_dtype, op_name);\n"
                    )
                    layout_autotune_list.append(
                        f"auto NEW_{name} = transformer->TransInTensor(\"{name}\", {name});\n"
                    )

            inputs_args_definition_list[pos] = arg_str
            inputs_args_declaration_list[pos] = arg_str

        # for layout autotune attr
        lightly_sensitive_attr = [
            'axis', 'axes', 'dim', 'dims', 'start', 'end', 'stop'
        ]
        heavily_sensitive_attr = ['data_format', 'data_layout']
        layout_autotune_attr = []
        layout_autotune_attr_code_list = []
        layout_autotune_attr_type_list = []
        layout_autotune_attr_code_list.append(
            f"auto op_name = phi::TransToFluidOpName(\"{forward_api_name}\");\n"
        )

        lightly_flag = False
        heavily_flag = False
        for name, atype, default_val, pos in forward_attrs_list:
            for attr_name in lightly_sensitive_attr:
                if name.find(
                        attr_name) != -1 and name not in layout_autotune_attr:
                    lightly_flag = True
                    layout_autotune_attr.append(name)
                    layout_autotune_attr_type_list.append(atype)
            if lightly_flag is False:
                for attr_name in heavily_sensitive_attr:
                    if name.find(attr_name
                                 ) != -1 and name not in layout_autotune_attr:
                        layout_autotune_attr.append(name)
                        layout_autotune_attr_type_list.append(atype)
                        heavily_flag = True
        if len(layout_autotune_attr) == 0:
            layout_autotune_attr_code_list.append(
                f"auto transformer = egr::EagerLayoutAutotune(op_name, tensors_vector);\n"
            )
        elif len(layout_autotune_attr) == 1:
            layout_autotune_attr_code_list.append(
                f"auto transformer = egr::EagerLayoutAutotune<{layout_autotune_attr_type_list[0]}>(op_name, tensors_vector, &{layout_autotune_attr[0]});\n"
            )
        elif len(layout_autotune_attr) == 2:
            layout_autotune_attr_code_list.append(
                f"auto transformer = egr::EagerLayoutAutotune<{layout_autotune_attr_type_list[0]}, {layout_autotune_attr_type_list[1]}>(op_name, tensors_vector, &{layout_autotune_attr[0]}, &{layout_autotune_attr[1]});\n"
            )
        else:
            layout_autotune_attr_code_list.append(
                f"auto transformer = egr::EagerLayoutAutotune(op_name, tensors_vector, {len(layout_autotune_attr)});\n"
            )

        # forward attrs
        for name, atype, default_val, pos in forward_attrs_list:
            inputs_call_list[pos] = name
            amp_inputs_call_list[pos] = name
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
            if is_inplaced:
                function_name = GetIntermediateAPIFunctionName(
                    forward_api_name[:-1]) + '_'
            else:
                function_name = GetIntermediateAPIFunctionName(function_name)

        api_out_type = "auto"
        if is_inplaced and len(forward_outputs_position_map) == 1:
            api_out_type = "auto&"
        forward_call_str = f"{indent}{api_out_type} api_result = paddle::experimental::{namespace}{function_name}({inputs_call_args_str});"
        num_outputs = len(
            forward_outputs_position_map.keys()) - len(intermediate_outputs)

        # Check Nan and Inf
        check_nan_inf_str = CHECK_NAN_AND_INF_TEMPLATE.format(
            function_name, "api_result")

        # Get Outputs
        get_outputs_str = ""
        for name, (rtype, pos) in forward_outputs_position_map.items():
            if num_outputs == 1 and len(intermediate_outputs) == 0:
                get_outputs_str += f"{indent}auto& {name} = api_result;\n"
            else:
                get_outputs_str += f"{indent}auto& {name} = std::get<{pos}>(api_result);\n"

        # Get return type list & outputs
        returns_type_list = ["" for i in range(num_outputs)]
        returns_list = ["" for i in range(num_outputs)]
        for name, (rtype, pos) in forward_outputs_position_map.items():
            if name in intermediate_outputs:
                continue
            returns_list[pos] = f"{name}"

            if IsPlainTensorType(rtype):
                if is_inplaced and forward_inplace_map and name in forward_inplace_map.values(
                ):
                    ind = list(forward_inplace_map.values()).index(name)
                    if list(forward_inplace_map.keys()
                            )[ind] in self.optional_inputs:
                        returns_type_list[pos] = inplace_optional_out_type_map[
                            rtype]
                    else:
                        returns_type_list[pos] = "paddle::experimental::Tensor&"
                else:
                    returns_type_list[pos] = "paddle::experimental::Tensor"
            else:
                assert IsVectorTensorType(rtype)
                if is_inplaced and forward_inplace_map and name in forward_inplace_map.values(
                ):
                    ind = list(forward_inplace_map.values()).index(name)
                    if list(forward_inplace_map.keys()
                            )[ind] in self.optional_inputs:
                        returns_type_list[pos] = inplace_optional_out_type_map[
                            rtype]
                    else:
                        returns_type_list[
                            pos] = "std::vector<paddle::experimental::Tensor>&"
                else:
                    returns_type_list[
                        pos] = "std::vector<paddle::experimental::Tensor>"

        if num_outputs == 1:
            returns_str = returns_list[0]
            returns_type_str = returns_type_list[0]
        else:
            returns_type_str = ", ".join(returns_type_list)
            returns_type_str = f"std::tuple<{returns_type_str}>"
            returns_str = ", ".join(returns_list)
            returns_str = f"{returns_type_str}{{{returns_str}}}"

        # Node Creation Pre-Processing
        if not self.is_forward_only:
            # 1. Get Input AutoGradMeta
            inputs_autograd_meta_list = []
            compute_require_grad_args_list = ["trace_backward"]
            for name, (ttype, pos) in forward_inputs_position_map.items():
                # Has corresponding grad output
                has_corresponding_grad_output = False
                for _, (_, corresponding_pos,
                        _) in backward_grad_outputs_map.items():
                    if pos == corresponding_pos:
                        has_corresponding_grad_output = True
                if has_corresponding_grad_output or (
                        name in forward_inplace_map and forward_api_name
                        not in inplace_check_blacklist) or self.is_forward_only:
                    input_autograd_meta_name = GetAutoGradMetaName(name)
                    if IsPlainTensorType(ttype):
                        input_autograd_meta = f"{indent}egr::AutogradMeta* {input_autograd_meta_name} = egr::EagerUtils::nullable_autograd_meta({name});"
                    else:
                        assert IsVectorTensorType(ttype)
                        input_autograd_meta_vec_name = GetAutoGradMetaVectorName(
                            name)
                        input_autograd_meta = f"{indent}std::vector<egr::AutogradMeta*> {input_autograd_meta_vec_name} = egr::EagerUtils::nullable_autograd_meta({name});\n"
                        input_autograd_meta += f"{indent}std::vector<egr::AutogradMeta*>* {input_autograd_meta_name} = &{input_autograd_meta_vec_name};"
                    inputs_autograd_meta_list.append(input_autograd_meta)
                    compute_require_grad_args_list.append(
                        input_autograd_meta_name)

            inputs_autograd_meta_str = "\n".join(inputs_autograd_meta_list)
            compute_require_grad_args_str = ",".join(
                compute_require_grad_args_list)

            # 2. Get Output AutoGradMeta
            outputs_autograd_meta_list = []
            num_fwd_outputs = len(forward_outputs_position_map.keys())

            for name, (rtype, pos) in forward_outputs_position_map.items():
                output_autograd_meta_name = GetAutoGradMetaName(name)
                output_autograd_meta_vec_name = GetAutoGradMetaVectorName(name)
                if num_fwd_outputs == 1:
                    if IsPlainTensorType(rtype):
                        output_autograd_meta = f"{indent}egr::AutogradMeta* {output_autograd_meta_name} = egr::EagerUtils::autograd_meta(&{name});"
                    else:
                        assert IsVectorTensorType(rtype)
                        output_autograd_meta = f"{indent}std::vector<egr::AutogradMeta*> {output_autograd_meta_vec_name} = egr::EagerUtils::autograd_meta(&{name});\n"
                        output_autograd_meta += f"{indent}std::vector<egr::AutogradMeta*>* {output_autograd_meta_name} = &{output_autograd_meta_vec_name};"
                else:
                    # Tuple api_result
                    if IsPlainTensorType(rtype):
                        output_autograd_meta = f"{indent}egr::AutogradMeta* {output_autograd_meta_name} = egr::EagerUtils::autograd_meta(&{name});"
                    else:
                        assert IsVectorTensorType(rtype)
                        output_autograd_meta = f"{indent}std::vector<egr::AutogradMeta*> {output_autograd_meta_vec_name} = egr::EagerUtils::autograd_meta(&{name});\n"
                        output_autograd_meta += f"{indent}std::vector<egr::AutogradMeta*>* {output_autograd_meta_name} = &{output_autograd_meta_vec_name};"

                outputs_autograd_meta_list.append(output_autograd_meta)
            outputs_autograd_meta_str = "\n".join(outputs_autograd_meta_list)

            # 3. Check Inplace
            check_inplace_str = ""
            bump_inplace_version_str = ""
            if is_inplaced:
                for inplace_name in forward_inplace_map.keys():
                    if forward_api_name not in inplace_check_blacklist:
                        inplace_autograd_meta_name = GetAutoGradMetaName(
                            inplace_name)
                        check_inplace_str += CHECK_INPLACE_TEMPLATE.format(
                            inplace_name, inplace_autograd_meta_name)
                    bump_inplace_version_str += BUMP_INPLACE_VERSION_TEMPLATE.format(
                        inplace_name, inplace_name)

            # Node Creation
            self.GenerateNodeCreationCodes()
            node_creation_str = self.node_creation_str

        dygraph_event_str = f"{indent}paddle::platform::RecordEvent dygraph_entrance_record_event(\"{forward_api_name} dygraph\", paddle::platform::TracerEventType::Operator, 1);\n"
        forward_function_name = GetDygraphForwardFunctionName(forward_api_name)

        # Forward amp logic
        kernel_trans2_op_name_str = f"auto op_name = phi::TransToFluidOpName(\"{forward_api_name}\");"
        amp_tensors_vector_list_str = "{ " + ",".join(
            amp_tensors_vector_list) + " }"
        amp_tensors_vector_optional_list_str = "    ".join(
            amp_tensors_vector_optional_list)
        amp_get_dst_dtype_str = f"auto amp_dst_dtype = egr::GetAmpDestDtype(op_name, amp_tensors_vector);\n"
        amp_autocast_list_str = "    ".join(
            amp_autocast_list) + "    " + "    ".join(
                amp_autocast_optional_list)
        amp_inputs_call_args_str = ", ".join(amp_inputs_call_list)
        amp_call_str = f"return {forward_function_name}({amp_inputs_call_args_str});"
        if is_inplaced or (forward_api_name == "cast"):
            amp_logic_str = ""
        else:
            amp_logic_str = AMP_LOGIC_TEMPLATE.format(
                kernel_trans2_op_name_str, amp_tensors_vector_list_str,
                amp_tensors_vector_optional_list_str, amp_get_dst_dtype_str,
                amp_autocast_list_str, amp_call_str)

        # Forward layout autotune
        layout_inputs_call_args_str = amp_inputs_call_args_str
        layout_tmp_result_list = []
        layout_autotune_outs_list = ""
        if num_outputs == 1:
            layout_autotune_outs_list += f"{indent}auto {returns_str} = api_result;\n"
            layout_autotune_outs_list += f"{indent}transformer -> SetOutTensorLayout(&{returns_str});\n"
        else:
            for name, (rtype, pos) in forward_outputs_position_map.items():
                if name in intermediate_outputs:
                    continue
                layout_autotune_outs_list += f"{indent}auto& {name} = std::get<{len(layout_tmp_result_list)}>(api_result);\n"
                layout_autotune_outs_list += f"{indent}transformer -> SetOutTensorLayout(&{name});\n"
                layout_tmp_result_list.append(f"{name}")

        if returns_type_str == "paddle::experimental::Tensor&" or forward_api_name == "slice" or forward_api_name == "strided_slice" or len(
                layout_autotune_attr) == 0:
            layout_logic_str = ""
        else:
            # after_call_str = f"return {forward_function_name}({layout_inputs_call_args_str});\n"
            after_call_str = f"auto api_result = {forward_function_name}({layout_inputs_call_args_str});\n"
            layout_logic_str = LAYOUT_LOGIC_TEMPLATE.format(
                amp_tensors_vector_list_str,
                "    ".join(layout_tensors_vector_optional_list),
                "    ".join(layout_autotune_attr_code_list) + "    " +
                "    ".join(layout_autotune_list) +
                "   ".join(layout_autotune_optional_list), after_call_str,
                layout_autotune_outs_list, returns_str)

        # Generate forward_definition_str and forward_declaration_str
        if self.is_forward_only:
            if len(amp_tensors_vector_list) == 0:
                amp_logic_str = ""
            self.forward_definition_str += FORWARD_ONLY_FUNCTION_TEMPLATE.format(
                returns_type_str, forward_function_name,
                inputs_args_definition_str, dygraph_event_str, amp_logic_str,
                layout_logic_str, forward_function_name, forward_call_str,
                get_outputs_str, returns_str)
        else:
            self.forward_definition_str += FORWARD_FUNCTION_TEMPLATE.format(
                returns_type_str, forward_function_name,
                inputs_args_definition_str, dygraph_event_str, amp_logic_str,
                layout_logic_str, inputs_autograd_meta_str,
                forward_function_name, forward_call_str, check_nan_inf_str,
                get_outputs_str, outputs_autograd_meta_str,
                compute_require_grad_args_str, check_inplace_str,
                bump_inplace_version_str, node_creation_str, returns_str)

        self.forward_declaration_str += f"{returns_type_str} {forward_function_name}({inputs_args_declaration_str});\n"

    def GenerateInplacedForwardDygraphFunctions(self):
        # Inplaced Version Dygraph Function Generation
        forward_api_name = self.forward_api_name
        forward_api_contents = self.forward_api_contents

        if forward_api_name != "sum" and "inplace" in forward_api_contents.keys(
        ):
            # Function Definition and Declaration Generation
            self.GenerateForwardDefinitionAndDeclaration(is_inplaced=True)
            self.UpdateCoreOpsInformation(is_inplaced=True)

    def UpdateCoreOpsInformation(self, is_inplaced):
        forward_api_name = GetInplacedFunctionName(
            self.forward_api_name) if is_inplaced else self.forward_api_name
        forward_inputs_position_map = self.forward_inputs_position_map
        forward_outputs_position_map = self.forward_outputs_position_map
        forward_attrs_list = self.forward_attrs_list

        num_args = len(
            forward_inputs_position_map.keys()) + len(forward_attrs_list)
        num_returns = len(forward_outputs_position_map.keys())

        fwd_api_name = "" + forward_api_name
        core_ops_returns_info[fwd_api_name] = ["" for i in range(num_returns)]
        core_ops_args_info[fwd_api_name] = ["" for i in range(num_args)]
        core_ops_args_type_info[fwd_api_name] = ["" for i in range(num_args)]

        for name, (ttype, pos) in forward_inputs_position_map.items():
            core_ops_args_info[fwd_api_name][pos] = name
            if IsPlainTensorType(ttype):
                core_ops_args_type_info[fwd_api_name][pos] = "tensor"
            else:
                assert IsVectorTensorType(ttype)
                core_ops_args_type_info[fwd_api_name][pos] = "list"

        for name, _, _, pos in forward_attrs_list:
            core_ops_args_info[fwd_api_name][pos] = name

        for name, (ttype, pos) in forward_outputs_position_map.items():
            core_ops_returns_info[fwd_api_name][pos] = name

    def run(self):
        super().run()

        #####################
        ## Code Generation ##
        #####################

        # Definition And Declaration
        self.GenerateForwardDefinitionAndDeclaration(is_inplaced=False)

        self.UpdateCoreOpsInformation(is_inplaced=False)

        self.GenerateInplacedForwardDygraphFunctions()


class DygraphNodeGenerator(DygraphFunctionGeneratorBase):

    def __init__(self,
                 forward_api_contents,
                 grad_api_contents,
                 forward_apis_dict,
                 namespace,
                 next_grad_api_contents=None):
        DygraphFunctionGeneratorBase.__init__(self, forward_api_contents,
                                              grad_api_contents,
                                              forward_apis_dict, namespace)

        # Record name mapping from forward_var_name to grad_var_names
        self.to_next_grad_name_mapping = {}  # {name : name}

        # Generated Results
        self.node_declaration_str = ""
        self.node_definition_str = ""
        self.next_grad_api_contents = next_grad_api_contents

    def TransformToNextGradName(self, string):
        name_mapping = self.to_next_grad_name_mapping
        if string in name_mapping.keys():
            return name_mapping[string]
        return string

    def ResetOptionalInputs(self):
        namespace = self.namespace
        grad_api_contents = self.grad_api_contents

        base_generator = FunctionGeneratorBase(grad_api_contents, namespace)
        base_generator.ParseDispensable()

        self.optional_inputs = base_generator.optional_inputs

    def RecordGrad2NextGradNameMapping(self, next_node_generator):
        next_orig_inputs_list = next_node_generator.orig_forward_inputs_list
        next_orig_returns_list = next_node_generator.orig_forward_returns_list

        next_forward_inputs_list = next_node_generator.forward_inputs_list
        next_forward_returns_list = next_node_generator.forward_returns_list
        for i in range(len(next_orig_inputs_list)):
            grad_name = next_orig_inputs_list[i][0]
            next_forward_name = next_forward_inputs_list[i][0]
            self.to_next_grad_name_mapping[grad_name] = next_forward_name

        for i in range(len(next_orig_returns_list)):
            grad_ret_name = next_orig_returns_list[i][0]
            next_ret_name = next_forward_returns_list[i][0]
            self.to_next_grad_name_mapping[grad_ret_name] = next_ret_name

    def GenerateHigherOrderNodeCreationCode(self):
        namespace = self.namespace
        grad_api_contents = self.grad_api_contents
        forward_apis_dict = self.forward_apis_dict
        next_grad_api_contents = self.next_grad_api_contents

        next_grad_node_creation_str = ""
        next_grad_node_out_list = []
        next_node_generator = None
        if next_grad_api_contents:
            # Fake forward_api_contents and backward_api_contents
            forward_api_contents = grad_api_contents
            forward_api_contents['api'] = forward_api_contents['backward_api']
            backward_api_contents = next_grad_api_contents

            next_node_generator = DygraphFunctionGeneratorBase(
                forward_api_contents, backward_api_contents, forward_apis_dict,
                namespace)
            next_node_generator.run()
            next_node_generator.GenerateNodeCreationCodes(True)

            next_grad_node_creation_str = next_node_generator.node_creation_str
            next_grad_node_out_list = next_node_generator.grad_node_out_list

            self.RecordGrad2NextGradNameMapping(next_node_generator)
        if next_node_generator is not None:
            return next_grad_node_creation_str, next_grad_node_out_list, next_node_generator.backward_forward_inputs_map
        else:
            return next_grad_node_creation_str, next_grad_node_out_list, None

    def GenerateNodeDeclaration(self):
        forward_op_name = self.forward_api_name
        backward_forward_inputs_map = self.backward_forward_inputs_map
        backward_attrs_list = self.backward_attrs_list
        no_need_buffers = self.no_need_buffers

        # SetTensorWrapper Methods & TensorWrapper Members & ClearTensorWrappers
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

        grad_node_name = GetGradNodeName(self.backward_api_name)
        self.node_declaration_str = NODE_DECLARATION_TEMPLATE.format(
            grad_node_name, grad_node_name, grad_node_name, grad_node_name,
            grad_node_name, clear_tensor_wrapper_str, grad_node_name,
            grad_node_name, set_tensor_wrapper_methods_str,
            set_attribute_methods_str, tensor_wrapper_members_str,
            attribute_members_str)

    def GenerateNodeDefinition(self, next_grad_node_creation_str,
                               next_grad_node_out_list,
                               backward_forward_inputs_map_next):
        namespace = self.namespace
        forward_api_name = self.forward_api_name
        backward_api_name = self.backward_api_name
        backward_forward_inputs_map = self.backward_forward_inputs_map
        backward_grad_inputs_map = self.backward_grad_inputs_map
        backward_grad_outputs_map = self.backward_grad_outputs_map
        backward_attrs_list = self.backward_attrs_list
        backward_inplace_map = self.backward_inplace_map
        indent = GetIndent(1)

        is_invoke_forward_api = IsInvokeForwardApi(self.grad_api_contents,
                                                   self.forward_apis_dict)
        # Construct grad_api function args
        # Order: TensorWrappers, GradTensors, Attributes
        grad_api_args_len = len(backward_forward_inputs_map.keys()) + len(
            backward_grad_inputs_map.keys()) + len(backward_attrs_list)
        grad_api_args = ["" for i in range(grad_api_args_len)]
        get_grad_in_args_list = []

        # Fill Grad Ins with Zero
        fill_zero_str = ""
        if backward_api_name in ops_to_fill_zero_for_empty_grads:
            fill_zero_str = f"{indent}const auto& input_metas = this->InputMeta();\n"
            for name, (ttype, fwd_position,
                       grad_api_position) in backward_grad_inputs_map.items():
                if name in self.optional_inputs:
                    if IsPlainTensorType(ttype):
                        fill_zero_str += f"{indent}egr::EagerUtils::FillZeroForEmptyOptionalGradInput(&grads[{fwd_position}][0], input_metas[{fwd_position}][0]);\n"
                else:
                    if IsPlainTensorType(ttype):
                        fill_zero_str += f"{indent}egr::EagerUtils::FillZeroForEmptyGradInput(&grads[{fwd_position}][0], input_metas[{fwd_position}][0]);\n"
                    else:
                        fill_zero_str += f"{indent}egr::EagerUtils::FillZeroForEmptyGradInput(&grads[{fwd_position}], input_metas[{fwd_position}]);\n"

        inplace_grad_input_str = ""
        inplaced_tensor_wrapper = False
        inplace_check_str = ""
        optional_inplace_var_name = []
        # Grad Ins from TensorWrappers
        for name, (backward_input_type, is_fwd_input,
                   grad_api_position), in backward_forward_inputs_map.items():
            tensor_wrapper_name = GetSavedName(name)
            transformed_tensor_name = self.TransformToNextGradName(name)

            is_optional = (name in self.optional_inputs)
            tensor_wrapper_recover_str = f"{indent}auto {transformed_tensor_name} = egr::EagerUtils::RecoverTensorWrapper(&this->{tensor_wrapper_name});"
            if backward_inplace_map and name in backward_inplace_map.keys():
                if len(next_grad_node_creation_str) > 0:
                    if (transformed_tensor_name
                            in backward_forward_inputs_map_next) and (
                                backward_forward_inputs_map_next[
                                    transformed_tensor_name][1]):
                        optional_inplace_var_name.append(
                            transformed_tensor_name)
                tensor_wrapper_intermidiate_tensor_str = f"(&this->{tensor_wrapper_name})->get_intermidiate_tensor()"
                inplace_check_str += CHECK_BACKWARD_INPLACE_TEMPLATE.format(
                    transformed_tensor_name, transformed_tensor_name, name,
                    transformed_tensor_name, transformed_tensor_name,
                    transformed_tensor_name, transformed_tensor_name,
                    tensor_wrapper_intermidiate_tensor_str)
                inplace_grad_input_str = transformed_tensor_name
            if is_optional:
                if backward_input_type == "std::vector<Tensor>":
                    tensor_wrapper_recover_str += "\n" + CREATE_RECOVER_OPTIONAL_VECTOR_TENSOR_TEMPLATE.format(
                        transformed_tensor_name, transformed_tensor_name,
                        transformed_tensor_name, transformed_tensor_name)
                else:
                    tensor_wrapper_recover_str += "\n" + CREATE_RECOVER_OPTIONAL_TENSOR_TEMPLATE.format(
                        transformed_tensor_name, transformed_tensor_name,
                        transformed_tensor_name, transformed_tensor_name)

                grad_api_args[
                    grad_api_position] = transformed_tensor_name + "_optional"

            else:
                grad_api_args[grad_api_position] = transformed_tensor_name

            get_grad_in_args_list.append(tensor_wrapper_recover_str)

        # Grad Ins from grads
        for name, (ttype, fwd_position,
                   grad_api_position) in backward_grad_inputs_map.items():
            transformed_tensor_name = self.TransformToNextGradName(name)

            is_optional = (name in self.optional_inputs)
            if IsPlainTensorType(ttype):
                get_tensor_str = f"{indent}auto& {transformed_tensor_name} = hooked_grads[{fwd_position}][0];"

                # Inplace in backward op
                if backward_inplace_map and name in backward_inplace_map.keys():
                    if len(next_grad_node_creation_str) > 0:
                        if (transformed_tensor_name
                                in backward_forward_inputs_map_next) and (
                                    backward_forward_inputs_map_next[
                                        transformed_tensor_name][1]):
                            optional_inplace_var_name.append(
                                transformed_tensor_name)
                    grads_tensor_str = f"grads[{fwd_position}][0]"
                    inplace_check_str += CHECK_BACKWARD_INPLACE_TEMPLATE.format(
                        transformed_tensor_name, transformed_tensor_name, name,
                        transformed_tensor_name, transformed_tensor_name,
                        transformed_tensor_name, transformed_tensor_name,
                        grads_tensor_str)
                    inplace_grad_input_str = transformed_tensor_name

                if is_optional:
                    get_tensor_str += "\n" + CREATE_PLAIN_OPTIONAL_TENSOR_TEMPLATE.format(
                        transformed_tensor_name, transformed_tensor_name,
                        transformed_tensor_name, transformed_tensor_name)
                    grad_api_args[
                        grad_api_position] = f"{transformed_tensor_name}_optional"
                else:
                    grad_api_args[grad_api_position] = transformed_tensor_name
            else:
                assert IsVectorTensorType(ttype)
                get_tensor_str = f"{indent}auto& {transformed_tensor_name} = hooked_grads[{fwd_position}];"
                grad_api_args[grad_api_position] = transformed_tensor_name

            get_grad_in_args_list.append(get_tensor_str)

        # Grad Attrs
        for name, _, _, grad_api_position in backward_attrs_list:
            saved_attribute_name = GetSavedName(name)
            get_attr_str = f"{indent}auto& {name} = this->{saved_attribute_name};"

            grad_api_args[grad_api_position] = name
            get_grad_in_args_list.append(get_attr_str)

        get_grad_in_args_str = "\n".join(get_grad_in_args_list)

        # Grad Function Call String
        slot_num_bwd_outputs = len(self.forward_inputs_position_map.keys())
        grad_api_namespace = f"paddle::experimental::{namespace}"
        grad_function_prepare_str = f"""
  const auto& out_metas = OutputMeta();
  paddle::small_vector<std::vector<paddle::experimental::Tensor>, egr::kSlotSmallVectorSize> returns({slot_num_bwd_outputs});
  for (int i = 0; i < {slot_num_bwd_outputs}; ++i) {{
    out_metas[i].size() == 0 ? returns[i].resize(1) : returns[i].resize(out_metas[i].size());
  }}
"""
        inplace_for_grad_outs_str = ""
        optional_inplace_str = ""
        # Grad Outputs
        out_index = -1
        out_assign_str = ""
        for name, (ttype, fwd_position,
                   grad_api_position) in backward_grad_outputs_map.items():
            transformed_tensor_name = self.TransformToNextGradName(name)
            out_index = out_index + 1
            if is_invoke_forward_api:
                if len(backward_grad_outputs_map) == 1:
                    out_assign_str += f"{indent}*api_output_{out_index} = api_output;\n"
                else:
                    out_assign_str += f"{indent}*api_output_{out_index} = std::get<{out_index}>(api_output);\n"
            else:
                grad_api_args.append(f"api_output_{out_index}")
            if inplace_grad_input_str in optional_inplace_var_name:
                optional_inplace_str = "VLOG(6) << \"No Inplace should happend for wrappered input: {inplace_grad_input_str}\";"
            else:
                optional_inplace_str = f"""if (api_output_{out_index} != nullptr && can_be_inplaced) {{
      egr::EagerUtils::HandleViewBetweenInputAndOutput({inplace_grad_input_str}, api_output_{out_index});
    }}"""
            if IsPlainTensorType(ttype):

                if backward_inplace_map and name in backward_inplace_map.values(
                ):
                    inplace_str = f""" if (api_output_{out_index} != nullptr && can_be_inplaced) {{
      egr::EagerUtils::HandleViewBetweenInputAndOutput({inplace_grad_input_str}, api_output_{out_index});
    }}"""
                    if len(next_grad_node_creation_str) > 0:
                        inplace_for_grad_outs_str += f"""
  if (trace_backward) {{
    {optional_inplace_str}
  }} else {{
    {inplace_str}
  }}"""
                    else:
                        inplace_for_grad_outs_str += inplace_str

                grad_function_prepare_str += f"""
  auto* api_output_{out_index} = (out_metas[{fwd_position}].empty() || out_metas[{fwd_position}][0].IsStopGradient()) ? nullptr : &returns[{fwd_position}][0];"""

            else:
                assert IsVectorTensorType(ttype)
                grad_function_prepare_str += f"""
  std::vector<paddle::experimental::Tensor*> api_output_{out_index};
  api_output_{out_index}.reserve(returns[{fwd_position}].size());
  for (size_t i = 0; i < returns[{fwd_position}].size(); ++i) {{
    if (out_metas[{fwd_position}].empty() || out_metas[{fwd_position}][i].IsStopGradient()) {{
      api_output_{out_index}.push_back(nullptr);
    }} else {{
      api_output_{out_index}.push_back(&returns[{fwd_position}][i]);
    }}
  }}"""

        grad_api_args_str = ", ".join(grad_api_args)

        if is_invoke_forward_api:
            autograd_api_out = "auto"
            if len(self.backward_inplace_map) > 0 and len(
                    backward_grad_outputs_map) == 1:
                autograd_api_out = "auto&"
            forward_api_name = self.grad_api_contents['invoke'].split(
                '(')[0].strip()
            autograd_api = self.grad_api_contents['invoke'].replace(
                forward_api_name, forward_api_name + '_dygraph_function', 1)
            grad_function_call_str = f"""
  if (trace_backward) {{            
  {indent}{autograd_api_out} api_output = {autograd_api};
  {out_assign_str}}} else {{
  {indent}{autograd_api_out} api_output = paddle::experimental::{self.namespace}{self.grad_api_contents['invoke']};
  {out_assign_str}{indent}}}
  """
        else:
            grad_function_call_str = f"""
{indent}{grad_api_namespace}{backward_api_name}({grad_api_args_str});"""

        # Check Nan and Inf
        check_nan_inf_str = CHECK_NAN_AND_INF_TEMPLATE.format(
            backward_api_name, "returns")

        # Prepare for Node Creation if Necessary
        outputs_autograd_meta_str = ""
        compute_require_next_grad_str = ""
        if len(next_grad_node_creation_str) > 0 or is_invoke_forward_api:
            compute_require_next_grad_str = f"{indent}bool trace_backward = egr::Controller::Instance().HasGrad() && create_graph;\n"

        # 3. Get Output AutoGradMeta
        outputs_autograd_meta_list = []
        # TODO(jiabin): Optimize this with SetStopGradient instead of Pass Stop gradient

        num_fwd_outputs = len(backward_grad_outputs_map.keys())
        for name, (rtype, pos,
                   grad_api_position) in backward_grad_outputs_map.items():
            transformed_tensor_name = self.TransformToNextGradName(name)

            output_autograd_meta_name = GetAutoGradMetaName(
                transformed_tensor_name)
            output_autograd_meta_vec_name = GetAutoGradMetaVectorName(
                transformed_tensor_name)
            if IsPlainTensorType(rtype):
                output_autograd_meta = f"""
  auto& {transformed_tensor_name} = returns[{pos}][0];
  egr::AutogradMeta* {output_autograd_meta_name} = returns[{pos}][0].initialized() ? egr::EagerUtils::autograd_meta(&{transformed_tensor_name}) : nullptr;
  if ({output_autograd_meta_name}) {output_autograd_meta_name}->SetStopGradient(false);
  """

            else:
                assert IsVectorTensorType(rtype)
                if len(next_grad_node_creation_str) > 0:
                    output_autograd_meta = f"""
    auto& {transformed_tensor_name} = returns[{pos}];
    std::vector<egr::AutogradMeta*> {output_autograd_meta_vec_name} = egr::EagerUtils::autograd_meta(&{transformed_tensor_name});
    std::vector<egr::AutogradMeta*>* {output_autograd_meta_name} = &{output_autograd_meta_vec_name};
    for(auto* meta : {output_autograd_meta_vec_name}){{
        meta->SetStopGradient(false);
    }}
"""
                else:
                    output_autograd_meta = f"""
    auto& {transformed_tensor_name} = returns[{pos}];
    std::vector<egr::AutogradMeta*> {output_autograd_meta_vec_name} = egr::EagerUtils::autograd_meta(&{transformed_tensor_name});
    for(auto* meta : {output_autograd_meta_vec_name}){{
        meta->SetStopGradient(false);
    }}
"""
            outputs_autograd_meta_list.append(output_autograd_meta)

        outputs_autograd_meta_str = "\n".join(outputs_autograd_meta_list)

        returns_str = f"{indent}if(NeedComplexToRealConversion()) HandleComplexGradToRealGrad(&returns);\n"
        returns_str += f"{indent}return returns;\n"

        grad_node_name = GetGradNodeName(self.backward_api_name)

        self.node_definition_str = GRAD_FUNCTION_TEMPLATE.format(
            grad_node_name, fill_zero_str, get_grad_in_args_str,
            grad_function_prepare_str, compute_require_next_grad_str,
            inplace_check_str, inplace_for_grad_outs_str, grad_node_name,
            grad_function_call_str, check_nan_inf_str,
            outputs_autograd_meta_str, next_grad_node_creation_str, returns_str)

    def run(self):
        super().run()

        self.ResetOptionalInputs()

        #####################
        ## Code Generation ##
        #####################
        # Higher-order GradNode generation
        next_grad_node_creation_str, next_grad_node_out_list, backward_forward_inputs_map = self.GenerateHigherOrderNodeCreationCode(
        )

        self.GenerateNodeDeclaration()

        self.GenerateNodeDefinition(next_grad_node_creation_str,
                                    next_grad_node_out_list,
                                    backward_forward_inputs_map)


class DygraphForwardAndNodesGenerator(GeneratorBase):

    def __init__(self, api_yaml_path, backward_yaml_path):
        # Parent members:
        # self.namespace
        # self.api_yaml_path
        # self.forward_api_list
        GeneratorBase.__init__(self, api_yaml_path)

        self.backward_yaml_path = backward_yaml_path
        self.grad_api_dict = {}

        self.forward_declaration_str = ""
        self.forward_definition_str = ""

        self.node_declaration_str = ""
        self.node_definition_str = ""

    def CollectIsForwardOnly(self, forward_api_contents):
        self.is_forward_only = False if 'backward' in forward_api_contents.keys(
        ) else True

    def ParseYamlContents(self):
        self.ParseForwardYamlContents()

        backward_yaml_path = self.backward_yaml_path

        # string api is forward_only, no backward_yaml respectively
        if backward_yaml_path is not None:
            self.grad_api_dict = ReadBwdFile(backward_yaml_path)

    def GetBackwardAPIContents(self, forward_api_contents):
        grad_api_dict = self.grad_api_dict

        if 'backward' not in forward_api_contents.keys(): return None

        backward_api_name = forward_api_contents['backward']
        assert backward_api_name in grad_api_dict.keys(), AssertMessage(
            backward_api_name, grad_api_dict.keys())
        backward_api_contents = grad_api_dict[backward_api_name]

        return backward_api_contents

    def GenerateCode(self):
        forward_api_list = self.forward_api_list
        grad_api_dict = self.grad_api_dict
        forward_apis_dict = {}
        for api_item in forward_api_list:
            forward_apis_dict[api_item['api']] = api_item
        namespace = self.namespace

        for forward_api_contents in forward_api_list:
            if forward_api_contents['api'] in black_ops_list: continue

            self.CollectIsForwardOnly(forward_api_contents)

            if self.is_forward_only:
                backward_api_contents = None
            else:
                backward_api_contents = self.GetBackwardAPIContents(
                    forward_api_contents)

            # Generate Dygraph Forward Function
            function_generator = DygraphForwardFunctionGenerator(
                forward_api_contents, backward_api_contents, forward_apis_dict,
                namespace)
            function_generator.run()

            self.forward_definition_str += function_generator.forward_definition_str + "\n"
            self.forward_declaration_str += function_generator.forward_declaration_str + "\n"

            # Generate Dygraph GradNode Function
            while True:
                if backward_api_contents is None:
                    break
                next_grad_api_contents = self.GetBackwardAPIContents(
                    backward_api_contents)

                node_generator = DygraphNodeGenerator(forward_api_contents,
                                                      backward_api_contents,
                                                      forward_apis_dict,
                                                      namespace,
                                                      next_grad_api_contents)
                node_generator.run()
                self.node_declaration_str += node_generator.node_declaration_str + "\n"
                self.node_definition_str += node_generator.node_definition_str + "\n"

                if next_grad_api_contents is None: break

                # Detect if there exists higher-order GradNode
                forward_api_contents = backward_api_contents

                # Fake forward_api_content
                forward_api_contents['api'] = forward_api_contents[
                    'backward_api']
                backward_api_contents = next_grad_api_contents

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

    forward_declaration_str = ""
    forward_definition_str = ""

    for i in range(len(api_yaml_paths)):
        api_yaml_path = api_yaml_paths[i]

        # string api is forwrad only
        if not api_yaml_path.endswith('strings_api.yaml'):
            backward_yaml_path = backward_yaml_paths[i]
        else:
            backward_yaml_path = None

        generator = DygraphForwardAndNodesGenerator(api_yaml_path,
                                                    backward_yaml_path)
        generator.run()

        node_declaration_str += generator.node_declaration_str + "\n"
        node_definition_str += generator.node_definition_str + "\n"

        forward_declaration_str += generator.forward_declaration_str + "\n"
        forward_definition_str += generator.forward_definition_str + "\n"

    # Generate Files
    nodes_h_path = args.nodes_h_path
    nodes_cc_path = args.nodes_cc_path
    forwards_h_path = args.forwards_h_path
    forwards_cc_path = args.forwards_cc_path

    GenerateNodeCCFile(nodes_cc_path, node_definition_str)
    GenerateNodeHFile(nodes_h_path, node_declaration_str)
    GenerateForwardCCFile(forwards_cc_path, forward_definition_str)
    GenerateForwardHFile(forwards_h_path, forward_declaration_str)
