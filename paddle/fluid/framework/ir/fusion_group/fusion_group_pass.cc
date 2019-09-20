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

#include "paddle/fluid/framework/ir/fusion_group/fusion_group_pass.h"
#include <vector>
#include "paddle/fluid/framework/ir/fusion_group/elementwise_group_detector.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

void FusionGroupPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph);

  int num_elementwise_groups = DetectFusionGroup(graph, 0);
  LOG(INFO) << "Detect " << num_elementwise_groups
            << " elementwise fusion groups.";
}

int FusionGroupPass::DetectFusionGroup(Graph* graph, int type) const {
  std::vector<std::unordered_set<Node*>> subgraphs;
  std::unordered_set<Node*> all_nodes = graph->Nodes();
  for (Node* n : all_nodes) {
    bool is_found = false;
    for (auto& subgraph : subgraphs) {
      if (subgraph.find(n) != subgraph.end()) {
        is_found = true;
        break;
      }
    }
    if (is_found) {
      continue;
    }

    std::unordered_set<Node*> subgraph;
    if (type == 0) {
      ElementwiseGroupDetector detector;
      int num_operations = detector(n);
      if (num_operations >= 2) {
        subgraph = detector.GetSubgraph();
      }
    }

    subgraphs.push_back(subgraph);
  }
  return subgraphs.size();
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fusion_group_pass, paddle::framework::ir::FusionGroupPass);
