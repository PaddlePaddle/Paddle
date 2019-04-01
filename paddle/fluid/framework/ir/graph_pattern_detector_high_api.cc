// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/graph_pattern_detector_high_api.h"

namespace paddle {
namespace framework {
namespace ir {
const PDNode2 &PDNode2::operator>>(const PDNode2 &other) const {
  pattern_->AddEdge(node_, other.node_);
  // automatically add out op link relation.
  if (other.pd_node().IsOp()) {
    CHECK(!other.op_type_.empty());
    node_->assert_is_op_input(other.op_type_);
  }

  return other;
}
const PDNode2 &PDNode2::operator>>(const std::vector<PDNode2> &nodes) const {
  for (auto &node : nodes) {
    *this >> node;
  }
  return *this;
}
const PDNode2 &operator>>(const std::vector<PDNode2> &others,
                          const PDNode2 &me) {
  for (const auto &o : others) {
    o >> me;
  }
  return me;
}
void FuseBase::PerformPatternDetector(Graph *graph) {
  LOG(INFO) << "\n" << detector_.pattern().DotString();
  // Get subgraphs and record the ir::Node pointers for each PDNode.
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     ir::Graph *g) {
    // get all the reigistered nodes.
    key2nodes_.emplace_back();
    for (auto &item : nodes_) {
      key2nodes_.back()[item.first] = subgraph.at(&item.second.pd_node());
    }
  };

  detector_(graph, handler);
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle
