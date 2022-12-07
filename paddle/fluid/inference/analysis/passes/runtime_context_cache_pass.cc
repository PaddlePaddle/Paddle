// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/analysis/passes/runtime_context_cache_pass.h"

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/inference/analysis/argument.h"

namespace paddle {
namespace inference {
namespace analysis {

using framework::ir::Node;

void RuntimeContextCachePass::RunImpl(Argument* argument) {
  static constexpr char kNotAllowInferShapeCahce[] =
      "@NOT_ALLOW_INFERSHAPE_CACHE@";
  VLOG(3) << "Applies Runtime Context Cache strategy.";
  auto& graph = argument->main_graph();
  for (auto* n : graph.Nodes()) {
    if (n->IsOp() && n->Op()) {
      n->Op()->SetAttr(framework::kEnableCacheRuntimeContext, true);
    }
  }

  // if op1 -> var0 and op2 -> var0, then op1 and op2 not support
  // InferShapeCache.
  std::unordered_map<std::string, std::vector<Node*>> var2ops;
  for (auto* op_node : framework::ir::TopologySortOperations(graph)) {
    for (auto* var_node : op_node->outputs) {
      var2ops[var_node->Name()].push_back(op_node);
    }
  }
  for (auto& it : var2ops) {
    if (it.second.size() > 1) {
      for (auto op_node : it.second) {
        op_node->Op()->SetAttr(kNotAllowInferShapeCahce, true);
      }
    }
  }
}

std::string RuntimeContextCachePass::repr() const {
  return "runtime_context_cache_pass";
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
