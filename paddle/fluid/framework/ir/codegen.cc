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
#include "paddle/fluid/framework/ir/codegen.h"
#include <set>
#include <sstream>
#include "paddle/fluid/framework/ir/codegen_helper.h"
namespace paddle {
namespace framework {
namespace ir {

CodeGenerator::CodeGenerator(CodeTemplate code_template) {
  code_template_ = code_template;
}

// in order to get the right result of expression, we need to calculate, we
// store the expression as
// suffix Expressions using vector
std::string CodeGenerator::GenerateCode(TemplateVariable template_var) {
  auto cuda_kernel = kernel_function + code_template_.Format(template_var);
  return cuda_kernel;
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle
