/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/framework/op_registry.h"

#include <vector>

#include "paddle/string/printf.h"

namespace paddle {
namespace framework {

OpProtoAndCheckerMaker::VariableBuilder&
OpProtoAndCheckerMaker::VariableBuilder::SetMultiple() {
  var_->set_multiple(true);
  on_multiple_();
  return *this;
}

OpProtoAndCheckerMaker::VariableBuilder&
OpProtoAndCheckerMaker::VariableBuilder::SetTemporary() {
  PADDLE_ENFORCE(bool(on_temporary_), "Cannot set temporary");
  var_->set_temporary(true);
  on_temporary_();
  return *this;
}

OpProtoAndCheckerMaker::VariableBuilder&
OpProtoAndCheckerMaker::VariableBuilder::IgnoreGradient() {
  var_->set_ignore_gradient(true);
  return *this;
}

OpProtoAndCheckerMaker::VariableBuilder OpProtoAndCheckerMaker::AddInput(
    const std::string& name, const std::string& comment) {
  auto input = proto_->mutable_inputs()->Add();
  *input->mutable_name() = name;
  *input->mutable_comment() = comment;
  return VariableBuilder{input, [=] { this->SetHasMultipleInput(); }, nullptr};
}

OpProtoAndCheckerMaker::VariableBuilder OpProtoAndCheckerMaker::AddOutput(
    const std::string& name, const std::string& comment) {
  auto output = proto_->mutable_outputs()->Add();
  *output->mutable_name() = name;
  *output->mutable_comment() = comment;
  return VariableBuilder{output, [=] { this->SetHasMultipleOutput(); },
                         [=] { this->SetHasTemporaryOutput(); }};
}

const char* kAttributeDocString = R"DOC(
This attribute is used by Paddle core framework. Paddle's Op support each input
or output could be a list of variable. This attribute is used to show how that
list organized.

e.g.
  input = ["a", "b", "c", "d", "e", "f"]
  input_format = [0, 4, 5, 6]

means
  The number of all input variables this op is six, and they are segmented into
  three inputs.

  The first input is input[0:4], second is input[4:5], third is input[5:6].
)DOC";

const char* kTemporaryOutputDocString = R"DOC(The temporary index of output.

Not all output of Paddle Op is used by user. For faster computation, each op
could output some its internal state to other op, other op could take that
output to make compute faster.

Add a mark to which output is temporary is helpful for future optimization.
)DOC";

void OpProtoAndCheckerMaker::SetHasMultiple(const std::string& in_out,
                                            bool* flag) {
  if (!*flag) {
    AddAttr<std::vector<int>>(
        in_out + "_format",
        string::Sprintf("The multiple index of %s \n%s", kAttributeDocString),
        true /*generated*/);
    *flag = true;
  }
}

void OpProtoAndCheckerMaker::SetHasMultipleInput() {
  SetHasMultiple("input", &has_multiple_input_);
}

void OpProtoAndCheckerMaker::SetHasMultipleOutput() {
  SetHasMultiple("output", &has_multiple_output_);
}

void OpProtoAndCheckerMaker::SetHasTemporaryOutput() {
  if (!has_temporary_output_) {
    AddAttr<std::vector<int>>("temporary_index", kTemporaryOutputDocString,
                              true)
        .SetDefault(std::vector<int>());
    has_temporary_output_ = true;
  }
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

static std::shared_ptr<OperatorBase> OpRegistry::CreateOp(
    const std::string& type, const VarNameList& inputs,
    const VarNameList& outputs, const AttributeMap& attrs) {
  auto op_create_it = op_creators().find(type);
  PADDLE_ENFORCE(op_create_it != op_creators().end(),
                 "Operator %s cannot be found.", type);

  auto op = op_create_it->second();
  op->type_ = type;
  op->inputs_ = inputs;
  op->outputs_ = outputs;

  op->attrs_ = attrs;
  op_checkers().at(type).Check(op->attrs_);

  GenerateTempVariableName(op);

  {
    auto var_index_it = VarIndexMaps().find(type);
    if (var_index_it != VarIndexMaps().end()) {
      op->in_out_idxs_ = var_index_it->second;
    }
  }

  op->Init();
  return std::shared_ptr<OperatorBase>(op);
}

static std::shared_ptr<OperatorBase> OpRegistry::CreateOp(
    const OpDesc& op_desc) {
  std::vector<std::string> inputs;
  inputs.reserve((size_t)op_desc.inputs_size());
  std::copy(op_desc.inputs().begin(), op_desc.inputs().end(),
            std::back_inserter(inputs));

  std::vector<std::string> outputs;
  outputs.reserve((size_t)op_desc.outputs_size());
  std::copy(op_desc.outputs().begin(), op_desc.outputs().end(),
            std::back_inserter(outputs));

  AttributeMap attrs;
  for (auto& attr : op_desc.attrs()) {
    attrs[attr.name()] = GetAttrValue(attr);
  }

  return CreateOp(op_desc.type(), inputs, outputs, attrs);
}

}  // namespace framework
}  // namespace paddle
