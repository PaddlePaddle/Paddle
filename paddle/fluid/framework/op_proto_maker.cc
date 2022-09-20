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

#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/operators/ops_extra_info.h"

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

void OpProtoAndCheckerMaker::Validate() {
  validated_ = true;
  CheckNoDuplicatedInOutAttrs();
}

OpProtoAndCheckerMaker::VariableBuilder OpProtoAndCheckerMaker::AddInput(
    const std::string& name, const std::string& comment) {
  auto* input = proto_->add_inputs();
  input->set_name(name);
  input->set_comment(comment);
  return OpProtoAndCheckerMaker::VariableBuilder{input};
}

OpProtoAndCheckerMaker::VariableBuilder OpProtoAndCheckerMaker::AddOutput(
    const std::string& name, const std::string& comment) {
  auto* output = proto_->add_outputs();
  output->set_name(name);
  output->set_comment(comment);
  return OpProtoAndCheckerMaker::VariableBuilder{output};
}

void OpProtoAndCheckerMaker::CheckNoDuplicatedInOutAttrs() {
  std::unordered_set<std::string> names;
  auto checker = [&](const std::string& name) {
    PADDLE_ENFORCE_EQ(
        names.count(name),
        0,
        platform::errors::AlreadyExists("Attribute [%s] is duplicated.", name));
    names.insert(name);
  };
  for (auto& attr : proto_->attrs()) {
    checker(attr.name());
  }
  for (auto& input : proto_->inputs()) {
    checker(input.name());
  }
  for (auto& output : proto_->outputs()) {
    checker(output.name());
  }
}

void OpProtoAndCheckerMaker::operator()(proto::OpProto* proto,
                                        OpAttrChecker* attr_checker) {
  proto_ = proto;
  op_checker_ = attr_checker;
  Make();
  op_checker_->RecordExplicitCheckerNum();

  const AttributeMap* extra_attrs_ptr = nullptr;
  const std::string& op_type = proto->type();

  const auto& extra_attr_map =
      operators::ExtraInfoUtils::Instance().GetExtraAttrsMap(op_type);
  if (!extra_attr_map.empty()) {
    extra_attrs_ptr = &extra_attr_map;
  }
  op_checker_->InitDefaultAttributeMap(extra_attrs_ptr);

  AddAttr<int>(OpRoleAttrName(), "The role of this operator")
      .InEnum(
          {static_cast<int>(OpRole::kForward),
           static_cast<int>(OpRole::kBackward),
           static_cast<int>(OpRole::kOptimize),
           static_cast<int>(OpRole::kRPC),
           static_cast<int>(OpRole::kDist),
           static_cast<int>(OpRole::kLRSched),
           static_cast<int>(OpRole::kLoss) | static_cast<int>(OpRole::kForward),
           static_cast<int>(OpRole::kLoss) |
               static_cast<int>(OpRole::kBackward),
           static_cast<int>(OpRole::kOptimize) |
               static_cast<int>(OpRole::kLRSched),
           static_cast<int>(OpRole::kNotSpecified)})
      .SetDefault(static_cast<int>(OpRole::kNotSpecified))
      .AsExtra();
  AddAttr<std::vector<std::string>>(OpRoleVarAttrName(),
                                    "Optimized for variable")
      .SetDefault({})
      .AsExtra();

  AddAttr<std::string>(OpNamescopeAttrName(), "Operator name with namesope.")
      .SetDefault("")
      .AsExtra();

  AddAttr<std::vector<std::string>>(OpCreationCallstackAttrName(),
                                    "Callstack for Op Creatation.")
      .SetDefault({})
      .AsExtra();
  AddAttr<std::string>(OpDeviceAttrName(), "Device type of this operator.")
      .SetDefault("")
      .AsExtra();

  AddAttr<bool>(OpWithQuantAttrName(),
                "Whether the operator has attributes used by quantization. ")
      .SetDefault(false)
      .AsExtra();

  Validate();
}

}  // namespace framework
}  // namespace paddle
