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

#include "paddle/fluid/framework/ir/lock_free_optimize_embedding_pass.h"
#include <string>
#include <vector>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

std::unique_ptr<ir::Graph> LockFreeOptimizeEmbeddingPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());

  for (auto* node : graph->Nodes()) {
    for (Node* input_node : node->inputs) {
      LOG(ERROR) << "Input link: " << input_node->Name() << "_"
                 << input_node->id() << " --> " << node->Name() << "_"
                 << node->id();
    }

    for (Node* output_node : node->outputs) {
      LOG(ERROR) << "Output link: " << node->Name() << "_" << node->id()
                 << " --> " << output_node->Name() << "_" << output_node->id();
    }
  }

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(lock_free_optimize_embedding_pass,
              paddle::framework::ir::LockFreeOptimizeEmbeddingPass);
