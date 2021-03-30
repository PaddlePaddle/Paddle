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

#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/ir/fusion_group/code_generator_helper.h"
#include "paddle/fluid/framework/ir/fusion_group/subgraph.h"

namespace paddle {
namespace framework {
namespace ir {
namespace fusion_group {

class SubGraph;

class CodeGenerator {
 public:
  CodeGenerator();

  std::string Generate(std::string func_name,
                       const std::vector<OperationExpression>& expressions);

  std::string Generate(SubGraph* subgraph);

  std::vector<OperationExpression> ConvertToExpressions(SubGraph* subgraph);

 private:
  std::set<int> DistilInputIds(
      const std::vector<OperationExpression>& expressions);
  std::set<int> DistilOutputIds(
      const std::vector<OperationExpression>& expressions);
  std::set<int> DistilIntermediateIds(
      const std::vector<OperationExpression>& expressions);
  std::unordered_map<int, std::string> DistilDtypes(
      const std::vector<OperationExpression>& expressions);

  // we get the parameter list code for the expression information
  std::string EmitParameters(
      const std::set<int>& input_ids, const std::set<int>& output_ids,
      const std::set<int>& intermediate_ids,
      const std::unordered_map<int, std::string>& dtypes) const;

  std::string EmitComputeBody(
      const std::vector<OperationExpression>& expressions,
      const std::set<int>& input_ids, const std::set<int>& output_ids,
      const std::set<int>& intermediate_ids,
      const std::unordered_map<int, std::string>& dtypes) const;

  // Encode all var nodes in the subgraph with an unique number.
  std::unordered_map<Node*, int> EncodeVarNodes(SubGraph* subgraph);

 private:
  std::vector<CodeTemplate> code_templates_;
};

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle
