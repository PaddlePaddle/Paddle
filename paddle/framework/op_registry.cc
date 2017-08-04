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

VariableBuilder OpProtoAndCheckerMaker::AddInput(const std::string& name,
                                                 const std::string& comment) {
  auto input = proto_->mutable_inputs()->Add();
  *input->mutable_name() = name;
  *input->mutable_comment() = comment;
  return VariableBuilder{input, [=] { this->SetHasMultipleInput(); }, nullptr};
}

VariableBuilder OpProtoAndCheckerMaker::AddOutput(const std::string& name,
                                                  const std::string& comment) {
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

}  // namespace framework
}  // namespace paddle
