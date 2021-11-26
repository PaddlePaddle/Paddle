// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/pybind/pybind.h"
#include "paddle/fluid/string/string_helper.h"

static std::unordered_set<std::string> operators_to_skip = {
    "fused_elemwise_add_activation",  // No Default Attr
    "fused_elemwise_activation",      // No Default Attr
    "reverse",                        // Attr Error
    "flip",                           // Attr Error
    "cast",                           // Attr Error
    "sum",
    "minus",  // Multiple ops_
    "pull_sparse",
    "pull_box_extended_sparse",
    "pull_sparse_v2",
    "pull_box_sparse",
    "fused_attention",
    "diag_v2",
};
/*
static std::unordered_set<std::string> operators_to_codegen = {
    "sigmoid",      "matmul_v2",   "reduce_sum", "elementwise_add",
    "share_buffer", "var_conv_2d", "split"};
*/

static std::unordered_set<std::string> skipped_operators = {};

namespace paddle {
namespace framework {

static std::string AttrTypeToString(const proto::AttrType& type) {
  std::string ret;
  switch (type) {
    case (proto::AttrType::INT): {
      ret = "int";
      break;
    }
    case (proto::AttrType::FLOAT): {
      ret = "float";
      break;
    }
    case (proto::AttrType::STRING): {
      ret = "std::string&";
      break;
    }
    case (proto::AttrType::INTS): {
      ret = "std::vector<int>&";
      break;
    }
    case (proto::AttrType::FLOATS): {
      ret = "std::vector<float>&";
      break;
    }
    case (proto::AttrType::STRINGS): {
      ret = "std::vector<std::string>&";
      break;
    }
    case (proto::AttrType::BOOLEAN): {
      ret = "bool";
      break;
    }
    case (proto::AttrType::BOOLEANS): {
      ret = "std::vector<bool>&";
      break;
    }
    case (proto::AttrType::LONG): {
      ret = "int64_t";
      break;
    }
    case (proto::AttrType::LONGS): {
      ret = "std::vector<int64_t>&";
      break;
    }
    case (proto::AttrType::BLOCK): {
      ret = "paddle::framework::BlockDesc*";
      break;
    }
    case (proto::AttrType::BLOCKS): {
      ret = "std::vector<paddle::framework::BlockDesc*>&";
      break;
    }
    case (proto::AttrType::FLOAT64S): {
      ret = "std::vector<double>&";
      break;
    }
    default: {
      PADDLE_THROW(
          platform::errors::Fatal("Unable to recognize AttrType: %d", type));
    }
  }
  return ret;
}

template <typename T>
static std::string GetAttrValue(const framework::Attribute& attr,
                                bool is_vector) {
  std::string val = "";
  if (is_vector) {
    val += "{";
    for (auto x : BOOST_GET_CONST(std::vector<T>, attr)) {
      val += std::to_string(x) + ",";
    }
    if (val.size() > 1) val.pop_back();
    val += "}";
  } else {
    val = std::to_string(BOOST_GET_CONST(T, attr));
  }
  return val;
}

static std::pair<std::string, std::string> GetAttrType(
    const framework::Attribute& attr, bool is_arg) {
  std::string ret = "";
  std::string val = "";
  size_t variant_pos = attr.which();
  switch (variant_pos) {
    case (1): {
      ret = "int";
      val = GetAttrValue<int>(attr, false);
      break;
    }
    case (2): {
      ret = "float";
      val = GetAttrValue<float>(attr, false);
      break;
    }
    case (3): {
      ret = "std::string";
      if (is_arg) ret += "&";
      val = "\"" + BOOST_GET_CONST(std::string, attr) + "\"";
      break;
    }
    case (4): {
      ret = "std::vector<int>";
      if (is_arg) ret += "&";
      val = GetAttrValue<int>(attr, true);
      break;
    }
    case (5): {
      ret = "std::vector<float>";
      if (is_arg) ret += "&";
      val = GetAttrValue<float>(attr, true);
      break;
    }
    case (6): {
      ret = "std::vector<std::string>";
      if (is_arg) ret += "&";
      val += "{";
      for (auto x : BOOST_GET_CONST(std::vector<std::string>, attr)) {
        val += "\"" + x + "\"" + ",";
      }
      if (val.size() > 1) val.pop_back();
      val += "};";
      break;
    }
    case (7): {
      ret = "bool";
      val = GetAttrValue<bool>(attr, false);
      break;
    }
    case (8): {
      ret = "std::vector<bool>";
      if (is_arg) ret += "&";
      val = GetAttrValue<bool>(attr, true);
      break;
    }
    case (9): {
      ret = "BlockDesc*";
      break;
    }
    case (10): {
      ret = "int64_t";
      val = GetAttrValue<int64_t>(attr, false);
      break;
    }
    case (11): {
      ret = "std::vector<BlockDesc*>";
      if (is_arg) ret += "&";
      break;
    }
    case (12): {
      ret = "std::vector<int64_t>";
      if (is_arg) ret += "&";
      val = GetAttrValue<int64_t>(attr, true);
      break;
    }
    case (13): {
      ret = "std::vector<double>";
      if (is_arg) ret += "&";
      val = GetAttrValue<double>(attr, true);
      break;
    }
    default: {
      PADDLE_THROW(platform::errors::Fatal("Unable to recognize AttrType: %d",
                                           variant_pos));
    }
  }
  return {ret, val};
}

static void SlotNameMatching(
    const std::map<
        std::string,
        std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>&
        grad_map,
    const std::map<
        std::string,
        std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>&
        fwd_ins,
    const std::map<
        std::string,
        std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>&
        fwd_outs,
    std::map<std::string, std::string>* grad_fwd_slotname_map_ptr,
    std::map<std::string, std::string>* grad_grad_slotname_map_ptr) {
  std::map<std::string, std::string>& grad_fwd_slotname_map =
      *grad_fwd_slotname_map_ptr;
  std::map<std::string, std::string>& grad_grad_slotname_map =
      *grad_grad_slotname_map_ptr;
  for (const auto& iter : grad_map) {
    const std::string& grad_slot_name = iter.first;
    const std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>&
        grad_vars = iter.second;

    // Find matching fwd_slot_name
    bool found_matching = false;
    for (const std::shared_ptr<paddle::imperative::VariableWrapper>& grad_var :
         grad_vars) {
      for (const auto& fwd_iter : fwd_ins) {
        const std::string& fwd_slot_name = fwd_iter.first;
        const std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>&
            fwd_vars = fwd_iter.second;
        for (const std::shared_ptr<paddle::imperative::VariableWrapper>&
                 fwd_var : fwd_vars) {
          if (grad_var == fwd_var) {
            if (grad_fwd_slotname_map.count(grad_slot_name) &&
                grad_fwd_slotname_map[grad_slot_name] != fwd_slot_name) {
              PADDLE_THROW(platform::errors::Fatal(
                  "grad_slot_name %s matches both %s and %s fwd_slot_name",
                  grad_slot_name, grad_fwd_slotname_map[grad_slot_name],
                  fwd_slot_name));
            }
            grad_fwd_slotname_map[grad_slot_name] = fwd_slot_name;
            found_matching = true;
          }

          if (fwd_var->GetGradVar() && grad_var == fwd_var->GetGradVar()) {
            if (grad_grad_slotname_map.count(grad_slot_name) &&
                grad_grad_slotname_map[grad_slot_name] != fwd_slot_name) {
              PADDLE_THROW(platform::errors::Fatal(
                  "grad_slot_name %s matches both %s and %s fwd_slot_name",
                  grad_slot_name, grad_grad_slotname_map[grad_slot_name],
                  fwd_slot_name));
            }
            grad_grad_slotname_map[grad_slot_name] = fwd_slot_name;
            found_matching = true;
          }
        }
      }
      for (const auto& fwd_iter : fwd_outs) {
        const std::string& fwd_slot_name = fwd_iter.first;
        const std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>&
            fwd_vars = fwd_iter.second;
        for (const std::shared_ptr<paddle::imperative::VariableWrapper>&
                 fwd_var : fwd_vars) {
          if (grad_var == fwd_var) {
            if (grad_fwd_slotname_map.count(grad_slot_name) &&
                grad_fwd_slotname_map[grad_slot_name] != fwd_slot_name) {
              PADDLE_THROW(platform::errors::Fatal(
                  "grad_slot_name %s matches both %s and %s fwd_slot_name",
                  grad_slot_name, grad_fwd_slotname_map[grad_slot_name],
                  fwd_slot_name));
            }
            grad_fwd_slotname_map[grad_slot_name] = fwd_slot_name;
            found_matching = true;
          }

          if (fwd_var->GetGradVar() && grad_var == fwd_var->GetGradVar()) {
            if (grad_grad_slotname_map.count(grad_slot_name) &&
                grad_grad_slotname_map[grad_slot_name] != fwd_slot_name) {
              PADDLE_THROW(platform::errors::Fatal(
                  "grad_slot_name %s matches both %s and %s fwd_slot_name",
                  grad_slot_name, grad_grad_slotname_map[grad_slot_name],
                  fwd_slot_name));
            }
            grad_grad_slotname_map[grad_slot_name] = fwd_slot_name;
            found_matching = true;
          }
        }
      }
    }

    if (!found_matching) {
      PADDLE_THROW(platform::errors::Fatal(
          "Found no matching fwd_slot_name for grad_slot_name: %s",
          grad_slot_name));

    } else {
      std::string fwd_slot_name = grad_grad_slotname_map.count(grad_slot_name)
                                      ? grad_grad_slotname_map[grad_slot_name]
                                      : grad_fwd_slotname_map[grad_slot_name];
      VLOG(6) << "Found matching fwd_slot_name: " << fwd_slot_name
              << " for grad_slot_name: " << grad_slot_name;
    }
  }
}

static bool CheckOpProto(proto::OpProto* op_proto) {
  if (op_proto == nullptr) {
    return false;
  }
  const std::string& op_type = op_proto->type();

  // Skip ooerator which is not inherit form OperatorWithKernel, like while,
  // since only OperatorWithKernel can run in dygraph mode.
  auto& all_kernels = paddle::framework::OperatorWithKernel::AllOpKernels();
  if (!all_kernels.count(op_type)) {
    return false;
  }

  // Only handle matmul_v2 for now
  VLOG(1) << "------ Analyzing Op ------: " << op_type;

  // if (!operators_to_codegen.count(op_type)) return false;
  if (operators_to_skip.count(op_type)) return false;

  return true;
}

/* -------------------------------- */
/* --------- Collect Info --------- */
/* -------------------------------- */
static bool CollectInformationFromOpInfo(
    const paddle::framework::OpInfo& op_info,
    std::vector<paddle::framework::AttributeMap>* grad_node_default_attr_maps,
    std::vector<std::string>* grad_op_types,
    std::unordered_map<std::string, size_t>* fwd_inputs_name_pos_map,
    std::unordered_map<std::string, size_t>* fwd_outputs_name_pos_map,
    std::map<std::string, std::string>* grad_outs_slotname_map,
    std::map<std::string, std::string>* grad_ins_fwd_slotname_map,
    std::map<std::string, std::string>* grad_ins_grad_slotname_map,
    std::map<std::string,
             std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>*
        grad_ins,
    std::map<std::string,
             std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>*
        grad_outs) {
  const proto::OpProto& op_proto = *op_info.proto_;
  const std::string& op_type = op_proto.type();
  std::vector<int64_t> dims = {1, 1, 1, 1};

  /* ------ Prepare "ins" ------ */
  std::map<std::string,
           std::vector<std::shared_ptr<paddle::imperative::VarBase>>>
      ins;
  for (const proto::OpProto::Var& input : op_proto.inputs()) {
    const std::string& in_name = input.name();

    // Handle dispensable input:
    // 1. At python level, dispensable input will be detected at Python-C
    // interface and filled with an empty vector
    // 2. At C++ level, customers should always pass an empty vector for any
    // dispensable input
    // 3. During further lowering, there will always be a placeholder VarBase
    // in ins/outs no matter whether it's dispensable or not
    // As a result, we always create input VarBase regardless of its
    // dispensability.

    // Handle duplicable input: list(VarBase) or VarBase
    // We dont know the exact number of inputs expected,
    // but we only need to identify the slot name order,
    // therefore fill in 1 single input VarBase is enough in this scenario
    ins[in_name] = {std::shared_ptr<paddle::imperative::VarBase>(
        new paddle::imperative::VarBase("auto_" + in_name))};
    ins[in_name][0]->SetOverridedStopGradient(false);
    ins[in_name][0]->MutableVar()->GetMutable<framework::LoDTensor>();
  }
  VLOG(6) << "Prepared Forward Ins Map, size = " << ins.size();

  /* ------ Prepare "outs" ------ */
  std::map<std::string,
           std::vector<std::shared_ptr<paddle::imperative::VarBase>>>
      outs;
  for (const proto::OpProto::Var& output : op_proto.outputs()) {
    const std::string& out_name = output.name();

    // We always create output VarBase regardless of its dispensability.
    // We dont know the exact number of outputs during code generation,
    // however, simply identifying the slot name order would be enough
    outs[out_name] = {std::shared_ptr<paddle::imperative::VarBase>(
        new paddle::imperative::VarBase("auto_" + out_name))};
    outs[out_name][0]->SetOverridedStopGradient(false);
    outs[out_name][0]->MutableVar()->GetMutable<framework::LoDTensor>();
  }
  VLOG(6) << "Prepared Forward Outs Map, size = " << outs.size();

  framework::AttributeMap attrs;
  paddle::framework::AttributeMap default_attrs;
  auto* attr_checker = op_info.Checker();
  if (attr_checker) {
    attr_checker->Check(&attrs, true, /*only_check_exist_value=*/true);
    default_attrs = attr_checker->GetDefaultAttrMap();
  } else {
    VLOG(6) << "Detected Null Attribute Checker, use empty default_attrs";
  }

  VLOG(6) << "Prepared Default Attributes Map, size = " << default_attrs.size();

  /* ---------------------------- */
  /* --------- Backward --------- */
  /* ---------------------------- */
  /* ------ Fwd paddle::imperative::VariableWrapper Map ------ */
  std::map<std::string,
           std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>
      fwd_ins;
  std::map<std::string,
           std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>
      fwd_outs;
  for (const auto& iter : ins) {
    fwd_ins[iter.first] = {};
    for (const std::shared_ptr<paddle::imperative::VarBase>& var_base :
         iter.second) {
      fwd_ins[iter.first].push_back(var_base->SharedVar());
    }
  }
  for (const auto& iter : outs) {
    fwd_outs[iter.first] = {};
    for (const std::shared_ptr<paddle::imperative::VarBase>& var_base :
         iter.second) {
      fwd_outs[iter.first].push_back(var_base->SharedVar());
    }
  }
  VLOG(6) << "Constructed Forward paddle::imperative::VariableWrapper Map";

  /* ------ Run GradOpMaker ------ */
  if (!op_info.dygraph_grad_op_maker_) {
    VLOG(6) << op_type << " has no GradOpMaker, skip it";
    skipped_operators.insert(op_type);
    return false;
  }

  std::shared_ptr<paddle::imperative::GradOpNode> grad_node =
      op_info.dygraph_grad_op_maker_(op_type, ins, outs, attrs, default_attrs,
                                     {});

  if (!grad_node) {
    VLOG(6) << "Got nullptr GradOpNode for " << op_type
            << " likely registered EmptyGradOpMaker, skip it";
    skipped_operators.insert(op_type);
    return false;
  }

  if (grad_node->size() > 1) {
    // Backward attributes can be super complicated
    VLOG(6) << "Skip GradOpNode with multiple OpBases for now: " << op_type;
    skipped_operators.insert(op_type);
    return false;
  }

  VLOG(6) << "Prepared GradOpNode";

  /* ---- Collect Default Attr Map ---- */
  for (auto iter = grad_node->begin(); iter < grad_node->end(); iter++) {
    // Each OpBase
    paddle::imperative::OpBase& op_base = *iter;
    grad_node_default_attr_maps->push_back(op_base.DefaultAttrsMap());
    grad_op_types->push_back(op_base.Type());
  }

  /* ------ Get Grad ins/outs ---- */
  // In case of multiple OpBase, stitch all the respective ins/outs into one
  VLOG(6) << "In function size: " << grad_node->size();
  for (auto iter = grad_node->begin(); iter < grad_node->end(); iter++) {
    const paddle::imperative::OpBase& op_base = *iter;
    const std::map<std::string, paddle::imperative::SavedVariableWrapperList>&
        g_ins = op_base.GetInsMap();
    const std::map<std::string, paddle::imperative::SavedVariableWrapperList>&
        g_outs = op_base.GetOutsMap();

    for (const auto& it : g_ins) {
      if (!grad_ins->count(it.first)) (*grad_ins)[it.first] = {};
      for (auto vw_iter = it.second.begin(); vw_iter != it.second.end();
           vw_iter++) {
        std::shared_ptr<paddle::imperative::VariableWrapper> vw = *vw_iter;
        (*grad_ins)[it.first].push_back(vw);
      }
    }

    for (const auto& it : g_outs) {
      if (!grad_outs->count(it.first)) (*grad_outs)[it.first] = {};
      for (auto vw_iter = it.second.begin(); vw_iter != it.second.end();
           vw_iter++) {
        std::shared_ptr<paddle::imperative::VariableWrapper> vw = *vw_iter;
        (*grad_outs)[it.first].push_back(vw);
      }
    }
  }

  /* ------ Slot Name Matching ---- */
  // grad_ins -> fwd_ins, fwd_outs
  SlotNameMatching(*grad_ins, fwd_ins, fwd_outs, grad_ins_fwd_slotname_map,
                   grad_ins_grad_slotname_map);
  VLOG(6) << "Finished Slotname Matching for Grad_Ins";

  // grad_outs -> fwd_ins, fwd_outs
  SlotNameMatching(*grad_outs, fwd_ins, fwd_outs, grad_outs_slotname_map,
                   grad_outs_slotname_map);
  VLOG(6) << "Finished Slotname Matching for Grad_Outs";

  /* ------ Maping forward slot name to fwd position ------ */
  size_t in_pos = 0;
  for (const auto& iter : ins) {
    VLOG(6) << "Mapping input tensor: " << iter.first
            << " To position: " << in_pos;
    (*fwd_inputs_name_pos_map)[iter.first] = in_pos;
    in_pos++;
  }
  size_t out_pos = 0;
  for (const auto& iter : outs) {
    VLOG(6) << "Mapping output tensor: " << iter.first
            << " To position: " << out_pos;
    (*fwd_outputs_name_pos_map)[iter.first] = out_pos;
    out_pos++;
  }

  return true;
}

/* --------------------------------------------------- */
/* --------- CodeGen: Forward GradNode Creation ------ */
/* --------------------------------------------------- */
static std::string GenerateGradNodeCreationContent(
    const std::vector<paddle::framework::AttributeMap>&
        grad_node_default_attr_maps,
    const std::unordered_map<std::string, size_t>& fwd_inputs_name_pos_map,
    const std::unordered_map<std::string, size_t>& fwd_outputs_name_pos_map,
    const std::map<std::string, std::string>& grad_ins_fwd_slotname_map,
    const proto::OpProto& op_proto) {
  VLOG(6) << "Generating GradNode Creation codes";

  const std::string& op_type = op_proto.type();

  // [Generation] Construct GradOpNode
  // Run ComputeRequiredGrad

  // If single output slotname and not duplicable,
  // then generate: "egr::AutogradMeta* p_autograd_out =
  // egr::EagerUtils::autograd_meta("op_proto->outputs()[0].name()")"

  // TODO(zhanlve): in case of multiple slotname but none of which are
  // duplicable,
  // avoid constructing vector<AutogradMeta*>, generate seperate
  // AutogradMeta* objects respectively.
  std::string get_autograd_meta_str = "  // Prepare Autograd Meta \n";
  for (const proto::OpProto::Var& input : op_proto.inputs()) {
    const std::string& input_name = input.name();
    const std::string& input_autograd_name = "p_autograd_" + input_name;

    if (input.duplicable()) {
      const char* GET_MULTI_AUTOGRAD_META_TEMPLATE =
          "  std::vector<egr::AutogradMeta*> %s = "
          "egr::EagerUtils::unsafe_autograd_meta(%s);\n";
      get_autograd_meta_str += paddle::string::Sprintf(
          GET_MULTI_AUTOGRAD_META_TEMPLATE, input_autograd_name, input_name);

    } else {
      const char* GET_SINGLE_AUTOGRAD_META_TEMPLATE =
          "  egr::AutogradMeta& %s = "
          "*egr::EagerUtils::unsafe_autograd_meta(%s);\n";
      get_autograd_meta_str += paddle::string::Sprintf(
          GET_SINGLE_AUTOGRAD_META_TEMPLATE, input_autograd_name, input_name);
    }
  }
  VLOG(6) << "Generated inputs autograd_meta";

  // If single output slotname and not duplicable,
  // then generate: "egr::AutogradMeta* p_autograd_out =
  // egr::EagerUtils::autograd_meta("op_proto.outputs()[0].name()")"

  // TODO(zhanlve): in case of multiple slotname but none of which are
  // duplicable,
  // avoid constructing vector<AutogradMeta*>, generate seperate
  // AutogradMeta* objects respectively.
  for (const proto::OpProto::Var& output : op_proto.outputs()) {
    const std::string& output_name = output.name();
    const std::string& output_autograd_name = "p_autograd_" + output_name;

    if (output.duplicable()) {
      const char* GET_MULTI_AUTOGRAD_META_TEMPLATE =
          "  std::vector<egr::AutogradMeta*> %s = "
          "egr::EagerUtils::multi_autograd_meta(&%s);\n";
      get_autograd_meta_str += paddle::string::Sprintf(
          GET_MULTI_AUTOGRAD_META_TEMPLATE, output_autograd_name, output_name);

    } else {
      const char* GET_SINGLE_AUTOGRAD_META_TEMPLATE =
          "  egr::AutogradMeta& %s = "
          "*egr::EagerUtils::autograd_meta(&%s);\n";
      get_autograd_meta_str += paddle::string::Sprintf(
          GET_SINGLE_AUTOGRAD_META_TEMPLATE, output_autograd_name, output_name);
    }
  }
  VLOG(6) << "Generated outputs autograd_meta";

  std::string prepare_autograd_meta_str = "";
  prepare_autograd_meta_str += get_autograd_meta_str;
  prepare_autograd_meta_str += "\n";

  // [GradOpNode] GetTraceBackward
  std::string trace_backward_str =
      "  bool trace_backward = egr::Controller::Instance().HasGrad();\n";
  prepare_autograd_meta_str += trace_backward_str;
  prepare_autograd_meta_str += "\n";

  // [GradOpNode] Generation
  std::string grad_node_creation_str = "";

  size_t bwd_in_slot_num = op_proto.outputs().size();
  size_t bwd_out_slot_num = op_proto.inputs().size();
  const char* GRAD_OP_NODE_TEMPLATE =
      "    auto grad_node = std::make_shared<GradNode%s>(%d, %d);\n";
  grad_node_creation_str += "    // Create GradOpNode\n";
  grad_node_creation_str += paddle::string::Sprintf(
      GRAD_OP_NODE_TEMPLATE, op_type, bwd_in_slot_num, bwd_out_slot_num);
  grad_node_creation_str += "\n";

  VLOG(6) << "Generated GradOpNode construction";

  // [GradOpNode] Set Attrs
  grad_node_creation_str += "    // Set Attributes\n";
  grad_node_creation_str += "    grad_node->SetAttrMap(std::move(attrs));\n";
  grad_node_creation_str +=
      "    grad_node->SetDefaultAttrMap(std::move(default_attrs));\n";
  grad_node_creation_str += "\n";

  // [GradOpNode] Set TensorWrappers
  grad_node_creation_str += "    // Set Tensor Wrappers\n";
  for (auto& kv : grad_ins_fwd_slotname_map) {
    const std::string& tensor_wrapper_name = kv.second;
    const char* SET_TENSOR_WRAPPER_TEMPLATE =
        "    grad_node->SetTensorWrapper%s(%s);\n";
    grad_node_creation_str += paddle::string::Sprintf(
        SET_TENSOR_WRAPPER_TEMPLATE, tensor_wrapper_name, tensor_wrapper_name);
  }
  grad_node_creation_str += "\n";
  VLOG(6) << "Generated SetTensorWrapper";

  // [GradOpNode] SetGradOutMeta
  // [GradOpNode] Add Edges
  std::string compute_require_grad_args = "trace_backward";
  for (const proto::OpProto::Var& input : op_proto.inputs()) {
    const std::string& input_name = input.name();
    const std::string& input_autograd_name = "p_autograd_" + input_name;
    compute_require_grad_args += ", &" + input_autograd_name;
    size_t input_position = fwd_inputs_name_pos_map.at(input_name);

    const char* SET_GRAD_OUT_META_TEMPLATE =
        "    grad_node->SetGradOutMeta(%s, %d);\n";
    grad_node_creation_str += paddle::string::Sprintf(
        SET_GRAD_OUT_META_TEMPLATE, input_autograd_name, input_position);

    const char* ADD_EDGES_TEMPLATE = "    grad_node->AddEdges(%s, %d);\n";
    grad_node_creation_str += paddle::string::Sprintf(
        ADD_EDGES_TEMPLATE, input_autograd_name, input_position);
  }

  // [GradOpNode] SetGradInMeta
  // [AutogradMeta] SetOutRank
  // [AutogradMeta] SetHistory
  std::string pass_stop_gradient_args = "false";
  for (const proto::OpProto::Var& output : op_proto.outputs()) {
    const std::string& output_name = output.name();
    const std::string& output_autograd_name = "p_autograd_" + output_name;
    pass_stop_gradient_args += ", &" + output_autograd_name;
    size_t output_position = fwd_outputs_name_pos_map.at(output_name);

    const char* SET_GRAD_IN_META_TEMPLATE =
        "    grad_node->SetGradInMeta(%s, %d);\n";
    grad_node_creation_str += paddle::string::Sprintf(
        SET_GRAD_IN_META_TEMPLATE, output_autograd_name, output_position);

    const char* SET_OUT_RANK_TEMPLATE =
        "    egr::EagerUtils::SetOutRankWithSlot(&%s, %d);\n";
    grad_node_creation_str += paddle::string::Sprintf(
        SET_OUT_RANK_TEMPLATE, output_autograd_name, output_position);

    const char* SET_HISTORY_TEMPLATE =
        "    egr::EagerUtils::SetHistory(&%s, grad_node);\n";
    grad_node_creation_str +=
        paddle::string::Sprintf(SET_HISTORY_TEMPLATE, output_autograd_name);
  }
  VLOG(6) << "Generated SetGradIn/OutMeta";

  // [Generation] GradNode Creation
  const char* GRAD_NODE_CREATION_TEMPLATE =
      "  %s"
      "  bool require_any_grad = egr::ComputeRequireGrad(%s);\n"
      "  if(require_any_grad) {\n"
      "    egr::PassStopGradient(%s);\n"
      "%s\n  }";
  std::string grad_node_creation_body_str = paddle::string::Sprintf(
      GRAD_NODE_CREATION_TEMPLATE, prepare_autograd_meta_str,
      compute_require_grad_args, pass_stop_gradient_args,
      grad_node_creation_str);

  return grad_node_creation_body_str;
}

static std::string AppendUseOp(const std::string& op_type) {
  // [Generation] Append USE_OP
  const char* USE_OP_TEMPLATE = "USE_OP(%s);\n";
  std::string return_str = paddle::string::Sprintf(USE_OP_TEMPLATE, op_type);

  // Special Ops
  if (op_type == "reduce_sum")
    return_str += paddle::string::Sprintf(USE_OP_TEMPLATE, "reduce_sum_grad");

  return return_str;
}

}  // namespace framework
}  // namespace paddle
