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
#include "paddle/fluid/framework/ir/fusion_group/subgraph.h"

namespace paddle {
namespace framework {
namespace ir {
namespace fusion_group {

class CodeGenerator {
 public:
  CodeGenerator();

  std::string Generate(std::string func_name,
                       std::vector<OperationExpression> expressions);

  std::string Generate(SubGraph* subgraph);

 private:
  // we get the parameter list code for the expression information
  std::string EmitParameters(std::vector<OperationExpression> expressions,
                             std::string dtype);

  std::string EmitComputeBody(std::vector<OperationExpression> expressions);

  // Encode all var nodes in the subgraph with an unique number.
  std::unordered_map<std::string, int> EncodeVarNodes(SubGraph* subgraph);

  // Insert a new expression into the vertor. Note that expressions should be
  // maintain in an order that the var is read after written.
  void InsertOperationExpression(std::vector<OperationExpression>* expressions,
                                 OperationExpression expr);

 private:
  std::vector<CodeTemplate> code_templates_;
};

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle
