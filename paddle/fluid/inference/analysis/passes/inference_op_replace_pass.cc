// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/inference/analysis/passes/inference_op_replace_pass.h"

#include "paddle/fluid/inference/analysis/argument.h"

namespace paddle::inference::analysis {

void InferenceOpReplacePass::RunImpl(Argument* argument) {
  if (argument->use_pir()) {
    return;
  }

  std::unordered_map<std::string, std::string> replaced_map{
      {"conditional_block", "conditional_block_infer"},
      {"merge_lod_tensor", "merge_lod_tensor_infer"},
  };

  auto& graph = argument->main_graph();
  auto nodes = graph.Nodes();

  for (auto& node : nodes) {
    if (!node->IsOp()) continue;
    auto* op_desc = node->Op();
    std::string op_type = op_desc->Type();
    if (!replaced_map.count(op_type)) continue;
    op_desc->SetType(replaced_map[op_type]);
    op_desc->Flush();
  }
}

std::string InferenceOpReplacePass::repr() const {
  return "inference_op_replace_pass";
}

}  // namespace paddle::inference::analysis
