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
  auto& info = OpInfoMap::Instance().Get(type);
  if (attr_check && info.Checker() != nullptr) {
    auto tmp_attrs = attrs;
    info.Checker()->Check(&tmp_attrs);
    return std::unique_ptr<OperatorBase>(
        info.Creator()(type, inputs, outputs, tmp_attrs));
  }
  return std::unique_ptr<OperatorBase>(
      info.Creator()(type, inputs, outputs, attrs));
}

std::unique_ptr<OperatorBase> OpRegistry::CreateOp(
    const std::string& type,
    const VariableNameMap& inputs,
    const VariableNameMap& outputs,
    const AttributeMap& attrs,
    const AttributeMap& runtime_attrs,
    bool attr_check) {
  auto op_base = CreateOp(type, inputs, outputs, attrs, attr_check);
  op_base->SetRuntimeAttributeMap(runtime_attrs);
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
