/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

const char kGraphvizMarkedNodeAttr[] = "__graphviz__marked_node__";

class GraphVizPass : public Pass {
 public:
  using marked_nodes_t = std::unordered_set<const Node*>;

 protected:
  void ApplyImpl(ir::Graph* graph) const override;

  // Tell whether there are any marked nodes in the graph. Consume the
  // corresponding attribute.
  marked_nodes_t ConsumeMarkedNodes(Graph* graph) const;
};

static GraphVizPass::marked_nodes_t& GetMarkedNodes(Graph* graph) {
  if (!graph->Has(kGraphvizMarkedNodeAttr)) {
    graph->Set(kGraphvizMarkedNodeAttr, new GraphVizPass::marked_nodes_t);
  }
  return graph->Get<GraphVizPass::marked_nodes_t>(kGraphvizMarkedNodeAttr);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
