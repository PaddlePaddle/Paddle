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
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/ir/fusion_group/subgraph.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"

namespace paddle {
namespace framework {
namespace ir {
namespace fusion_group {

class ElementwiseGroupDetector {
 public:
  explicit ElementwiseGroupDetector(Graph* graph, bool backward) {
    Init(graph, backward);
  }

  SubGraph operator()(Node* n);

 private:
  void Init(Graph* graph, bool backward);

  bool IsElementwiseOp(Node* n);
  bool IsInputOfElementwiseOp(Node* n, std::string name = "");
  bool IsOutputOfElementwiseOp(Node* n);

  int Search(Node* n, std::vector<Node*> except_nodes, SubGraph* subgraph);

 private:
  Graph* graph_{nullptr};  // Not owned
  bool backward_{false};
  std::unordered_set<Node*> elementwise_ops_;
};

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle
