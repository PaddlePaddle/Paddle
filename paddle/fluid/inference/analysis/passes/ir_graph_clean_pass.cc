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

#include "paddle/fluid/inference/analysis/passes/ir_graph_clean_pass.h"
#include <algorithm>
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/node.h"

namespace paddle {
namespace inference {
namespace analysis {

void IrInferCleanGraphPass::RunImpl(Argument* argument) {
  auto& graph = argument->main_graph();
  auto is_valid_node = [](framework::ir::Node* x) {
    return x && IsControlDepVar(*x) && x->IsVar() && !x->Var();
  };

  std::unordered_set<const framework::ir::Node*> invalid_nodes;
  int valid_op = 0;
  for (auto* node : graph.Nodes()) {
    PADDLE_ENFORCE_NOT_NULL(node);
    if (is_valid_node(node)) {
      invalid_nodes.insert(node);
    } else if (node->IsOp()) {
      ++valid_op;
    }
  }

  GraphSafeRemoveNodes(&graph, invalid_nodes);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
