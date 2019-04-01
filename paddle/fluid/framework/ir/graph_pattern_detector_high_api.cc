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

void FuseBase::DeleteInterNodes(ir::Graph *graph) {
  std::set<std::string> keys;
  for (auto &node2 : nodes_) {
    if (node2.second.pd_node().IsIntermediate()) {
      keys.insert(node2.first);
    }
  }

  LOG(INFO) << "keys.size " << keys.size();

  std::unordered_set<const ir::Node *> nodes2rm;
  for (auto &matched : key2nodes_) {
    LOG(INFO) << "get matched " << matched.size();
    for (const auto &key : keys) {
      nodes2rm.insert(matched.at(key));
    }
  }

  LOG(INFO) << "clean nodes " << nodes2rm.size();
  GraphSafeRemoveNodes(graph, nodes2rm);
}

PDNode2 &FuseBase::Node(const std::string &key) {
  auto it = nodes_.find(key);
  if (it != nodes_.end()) {
    return it->second;
  }
  nodes_.emplace(key, PDNode2{detector_.mutable_pattern(), key});
  it = nodes_.find(key);
  return it->second;
}

PDNode2 &FuseBase::OpNode(const std::string &key, const std::string &op_type) {
  Node(key).SetOpType(op_type);
  Node(key).pd_node().AsOp(op_type);
  return Node(key);
}

PDNode2 &FuseBase::VarNode(const std::string &key) {
  Node(key).pd_node().AsVar();
  return Node(key);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
