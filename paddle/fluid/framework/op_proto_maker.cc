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
#include <string>
#include <vector>

namespace paddle {
namespace framework {

void OpProtoAndCheckerMaker::Validate() {
  validated_ = true;
  CheckNoDuplicatedInOutAttrs();
  CheckReuseVars();
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

void OpProtoAndCheckerMaker::Reuse(const std::string& name,
                                   const std::string& reused_name) {
  bool found = false;
  proto::OpProto::Var* var;

  for (auto& var : proto_->inputs()) {
    if (var.name() == reused_name) {
      found = true;
      break;
    }
  }
  PADDLE_ENFORCE(found == true,
                 "Input/Output name: %s reused_name: %s, one of them is not "
                 "exists or not matched.",
                 name, reused_name);

  found = false;
  for (int i = 0; i < proto_->outputs().size(); ++i) {
    var = proto_->mutable_outputs()->Mutable(i);
    if (var->name() == name) {
      PADDLE_ENFORCE(!var->has_reuse(),
                     "Output(%s) has been set reused var of %s", name,
                     var->reuse());
      found = true;
      var->set_reuse(reused_name);
      break;
    }
  }
  PADDLE_ENFORCE(found == true,
                 "Input/Output name: %s reused_name: %s, one of them is not "
                 "exists or not matched.",
                 name, reused_name);
}

void OpProtoAndCheckerMaker::CheckNoDuplicatedInOutAttrs() {
  std::unordered_set<std::string> names;
  auto checker = [&](const std::string& name) {
    PADDLE_ENFORCE(!names.count(name), "[%s] is duplicated", name);
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

void OpProtoAndCheckerMaker::CheckReuseVars() {
  std::unordered_set<std::string> names;
  for (auto& input : proto_->inputs()) {
    names.insert(input.name());
  }
  auto checker = [&](const std::string& name, const std::string& reused) {
    PADDLE_ENFORCE(
        names.count(reused),
        "Output [%s] reuse Input [%s], but the input is not registered.", name,
        reused);
  };
  for (auto& output : proto_->outputs()) {
    if (output.has_reuse()) {
      checker(output.name(), output.reuse());
    }
  }
}

void OpProtoAndCheckerMaker::operator()(proto::OpProto* proto,
                                        OpAttrChecker* attr_checker) {
  proto_ = proto;
  op_checker_ = attr_checker;
  Make();

  AddAttr<int>(OpRoleAttrName(), "The role of this operator")
      .InEnum(
          {static_cast<int>(OpRole::kForward),
           static_cast<int>(OpRole::kBackward),
           static_cast<int>(OpRole::kOptimize), static_cast<int>(OpRole::kRPC),
           static_cast<int>(OpRole::kDist), static_cast<int>(OpRole::kLRSched),
           static_cast<int>(OpRole::kLoss) | static_cast<int>(OpRole::kForward),
           static_cast<int>(OpRole::kLoss) |
               static_cast<int>(OpRole::kBackward),
           static_cast<int>(OpRole::kNotSpecified)})
      .SetDefault(static_cast<int>(OpRole::kNotSpecified));
  AddAttr<std::vector<std::string>>(OpRoleVarAttrName(),
                                    "Optimized for variable")
      .SetDefault({});

  AddAttr<std::string>(OpNamescopeAttrName(), "Operator name with namesope.")
      .SetDefault("");

  Validate();
}

}  // namespace framework
}  // namespace paddle
