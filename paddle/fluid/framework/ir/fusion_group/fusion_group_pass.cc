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
#include "paddle/fluid/framework/ir/fusion_group/elementwise_pattern.h"

namespace paddle {
namespace framework {
namespace ir {

void FusionGroupPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph);
  FusePassBase::Init("fusion_group", graph);

  int found_fusion_group_count = 0;
  // for (int num_operations = 2; num_operations < 3; ++num_operations) {
  //   found_fusion_group_count += ApplyPattern(graph, num_operations);
  // }
  found_fusion_group_count += ApplyPattern(graph, 3);

  AddStatis(found_fusion_group_count);
}

int FusionGroupPass::ApplyPattern(Graph* graph, int num_operations) const {
  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()->NewNode(
      [=](Node* n) {
        if (patterns::IsInputOfElementwiseOp(n) &&
            patterns::NumAbjacentElementwiseOps(n, n->inputs) ==
                num_operations) {
          return true;
        }
        return false;
      },
      "fusison_group/in");
  patterns::ElementwiseGroupPattern pattern(gpd.mutable_pattern(),
                                            "fusion_group");
  pattern(x, num_operations);

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (subgraph.count(x) <= 0) {
      LOG(WARNING) << "The subgraph is empty.";
      return;
    }

    fusion_count++;
  };
  gpd(graph, handler);
  return fusion_count;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fusion_group_pass, paddle::framework::ir::FusionGroupPass);
