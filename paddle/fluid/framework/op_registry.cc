/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/ops_extra_info.h"

#include "glog/logging.h"

namespace paddle {
namespace framework {

std::unique_ptr<OperatorBase> OpRegistry::CreateOp(
    const std::string& type,
    const VariableNameMap& inputs,
    const VariableNameMap& outputs,
    const AttributeMap& attrs,
    bool attr_check) {
  AttributeMap standard_attrs;
  AttributeMap runtime_attrs =
      paddle::operators::ExtraInfoUtils::Instance().GetExtraAttrsMap(type);
  for (auto& attr : attrs) {
    auto it = runtime_attrs.find(attr.first);
    if (it != runtime_attrs.end()) {
      it->second = attr.second;
    } else {
      standard_attrs[attr.first] = attr.second;
    }
  }
  auto& info = OpInfoMap::Instance().Get(type);
  if (attr_check) {
    if (info.Checker() != nullptr) {
      info.Checker()->Check(&standard_attrs);
    }
    const auto& extra_attr_checkers =
        operators::ExtraInfoUtils::Instance().GetExtraAttrsChecker(type);
    if (!extra_attr_checkers.empty()) {
      for (const auto& checker : extra_attr_checkers) {
        checker(&runtime_attrs, false);
      }
    }
  }
  auto op_base = std::unique_ptr<OperatorBase>(
      info.Creator()(type, inputs, outputs, standard_attrs));
  op_base->SetRuntimeAttributeMap(runtime_attrs);
  return op_base;
}

std::unique_ptr<OperatorBase> OpRegistry::CreateOp(
    const std::string& type,
    const VariableNameMap& inputs,
    const VariableNameMap& outputs,
    const AttributeMap& attrs,
    const AttributeMap& runtime_attrs,
    bool attr_check) {
  std::unique_ptr<OperatorBase> op_base;
  auto& info = OpInfoMap::Instance().Get(type);
  if (attr_check && info.Checker() != nullptr) {
    auto tmp_attrs = attrs;
    info.Checker()->Check(&tmp_attrs);
    op_base = std::unique_ptr<OperatorBase>(
        info.Creator()(type, inputs, outputs, tmp_attrs));
  } else {
    op_base = std::unique_ptr<OperatorBase>(
        info.Creator()(type, inputs, outputs, attrs));
  }
  const auto& extra_attr_checkers =
      operators::ExtraInfoUtils::Instance().GetExtraAttrsChecker(type);
  if (!extra_attr_checkers.empty()) {
    auto op_runtime_attr_map = runtime_attrs;
    for (const auto& checker : extra_attr_checkers) {
      checker(&op_runtime_attr_map, false);
    }
    op_base->SetRuntimeAttributeMap(op_runtime_attr_map);
  }
  return op_base;
}

static VariableNameMap ConvertOpDescVarsToVarNameMap(
    const google::protobuf::RepeatedPtrField<proto::OpDesc::Var>&
        op_desc_vars) {
  VariableNameMap ret_val;
  for (auto& var : op_desc_vars) {
    auto& var_names = ret_val[var.parameter()];
    auto& var_names_in_proto = var.arguments();
    var_names.reserve(static_cast<size_t>(var_names_in_proto.size()));
    std::copy(var_names_in_proto.begin(),
              var_names_in_proto.end(),
              std::back_inserter(var_names));
  }
  return ret_val;
}

std::unique_ptr<OperatorBase> OpRegistry::CreateOp(
    const proto::OpDesc& op_desc) {
  VLOG(1) << "CreateOp directly from OpDesc is deprecated. It should only be"
             "used in unit tests. Use CreateOp(const OpDesc& op_desc) "
             "instead.";
  VariableNameMap inputs = ConvertOpDescVarsToVarNameMap(op_desc.inputs());
  VariableNameMap outputs = ConvertOpDescVarsToVarNameMap(op_desc.outputs());
  AttributeMap attrs;
  AttributeMap extra_attrs =
      paddle::operators::ExtraInfoUtils::Instance().GetExtraAttrsMap(
          op_desc.type());
  for (auto& attr : op_desc.attrs()) {
    auto it = extra_attrs.find(attr.name());
    if (it != extra_attrs.end()) {
      it->second = GetAttrValue(attr);
    } else {
      attrs[attr.name()] = GetAttrValue(attr);
    }
  }

  return CreateOp(op_desc.type(), inputs, outputs, attrs, extra_attrs);
}

std::unique_ptr<OperatorBase> OpRegistry::CreateOp(const OpDesc& op_desc) {
  return CreateOp(op_desc.Type(),
                  op_desc.Inputs(),
                  op_desc.Outputs(),
                  op_desc.GetAttrMap(),
                  op_desc.GetRuntimeAttrMap());
}

}  // namespace framework
}  // namespace paddle
