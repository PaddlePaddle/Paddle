/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/codegen_helper.h"

namespace paddle {
namespace framework {
namespace ir {

class CodeGenerator {
 public:
  explicit CodeGenerator(CodeTemplate code_template);
  std::string GenerateCode(TemplateVariable template_var);
  // TODO(wangchao66) std::string GenerateCode(const Graph& graph)

 private:
  CodeTemplate code_template_;
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle
