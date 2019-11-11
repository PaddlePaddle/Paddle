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
#include "paddle/fluid/framework/ir/fusion_group/code_generator_helper.h"

namespace paddle {
namespace framework {
namespace ir {
namespace fusion_group {

class CodeGenerator {
 public:
  CodeGenerator();

  std::string GenerateCode(std::string func_name,
                           std::vector<OperationExpression> expressions);

  // TODO(wangchao): add a more general interface
  // std::string Generate(const std::string name, const SubGraph& subgraph);

 private:
  // we get the parameter list code for the expression information
  std::string EmitParameters(std::vector<OperationExpression> expressions,
                             std::string dtype);

  std::string EmitComputeBody(std::vector<OperationExpression> expressions);

 private:
  std::vector<CodeTemplate> code_templates_;
};

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle
