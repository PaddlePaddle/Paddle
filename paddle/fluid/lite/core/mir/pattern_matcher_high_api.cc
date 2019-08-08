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

#include "paddle/fluid/lite/core/mir/pattern_matcher_high_api.h"
#include <glog/logging.h>

namespace paddle {
namespace lite {
namespace mir {

void FuseBase::PerformPatternMatcher(SSAGraph *graph) {
  VLOG(4) << "\n" << matcher_.pattern().DotString();
  // Get subgraphs and record the mir::Node pointers for each PMNode.
  auto handler = [&](const PatternMatcher::subgraph_t &subgraph, SSAGraph *g) {
    // get all the reigistered nodes.
    key2nodes_.emplace_back();
    for (auto &item : nodes_) {
      key2nodes_.back()[item.first] = subgraph.at(item.second);
    }
  };

  matcher_(graph, handler);
}

void FuseBase::DeleteInterNodes(SSAGraph *graph) {
  std::set<std::string> keys;
  for (auto &node : nodes_) {
    if (node.second->IsIntermediate()) {
      keys.insert(node.first);
    }
  }

  VLOG(4) << "keys: " << key2nodes_.size();
  std::unordered_set<const Node *> nodes2rm;
  for (auto &matched : key2nodes_) {
    for (const auto &key : keys) {
      nodes2rm.insert(matched.at(key));
    }
  }

  VLOG(3) << "clean nodes " << nodes2rm.size();
  GraphSafeRemoveNodes(graph, nodes2rm);
}

PMNode *FuseBase::GetOrCreateNode(const std::string &key) {
  auto it = nodes_.find(key);
  if (it != nodes_.end()) {
    return it->second;
  }
  nodes_.emplace(key,
                 matcher_.mutable_pattern()->NewNode(patterns::UniqueKey(key)));
  it = nodes_.find(key);
  return it->second;
}

PMNode *FuseBase::OpNode(const std::string &key, const std::string &op_type) {
  GetOrCreateNode(key)->set_op_type(op_type);
  GetOrCreateNode(key)->AsOp(op_type);
  return GetOrCreateNode(key);
}

PMNode *FuseBase::VarNode(const std::string &key) {
  GetOrCreateNode(key)->AsVar();
  return GetOrCreateNode(key);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
