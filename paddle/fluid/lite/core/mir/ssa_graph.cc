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

#include "paddle/fluid/lite/core/mir/ssa_graph.h"

namespace paddle {
namespace lite {
namespace mir {

bool SSAGraph::CheckBidirectionalConnection() {
  LOG(INFO) << "node count " << node_storage_.size();
  for (auto &node : node_storage_) {
    for (auto *in : node.inlinks) {
      CHECK(in->outlinks.end() !=
            std::find(in->outlinks.begin(), in->outlinks.end(), &node));
    }
    for (auto *out : node.outlinks) {
      CHECK(out->inlinks.end() !=
            std::find(out->inlinks.begin(), out->inlinks.end(), &node));
    }
  }
  return true;
}

std::map<mir::Node *, std::set<mir::Node *>> SSAGraph::BuildOperationAdjList() {
  std::map<mir::Node *, std::set<mir::Node *>> adj_list;

  for (auto &n : mutable_nodes()) {
    if (!n.IsInstruct()) continue;
    if (adj_list.find(&n) == adj_list.end()) {
      adj_list[&n] = std::set<mir::Node *>();
    }
    std::vector<mir::Node *> nodes;
    for (auto &var : n.inlinks) {
      for (auto &adj_n : var->inlinks) {
        PADDLE_ENFORCE(adj_n->IsInstruct());
        nodes.push_back(adj_n);
      }
    }
    std::sort(nodes.begin(), nodes.end(),
              [](mir::Node *node1, mir::Node *node2) { return node1 > node2; });
    adj_list[&n].insert(std::make_move_iterator(nodes.begin()),
                        std::make_move_iterator(nodes.end()));
  }
  return adj_list;
}

void SSAGraph::SortHelper(
    const std::map<mir::Node *, std::set<mir::Node *>> &adj_list,
    mir::Node *node, std::set<mir::Node *> *visited,
    std::vector<mir::Node *> *ret) {
  visited->insert(node);

  for (auto adj : adj_list.at(node)) {
    if (visited->find(adj) == visited->end()) {
      SortHelper(adj_list, adj, visited, ret);
    }
  }

  ret->push_back(node);
}

std::vector<mir::Node *> SSAGraph::InstructTopologicalOrder() {
  CheckBidirectionalConnection();

  std::stack<mir::Node *> stack;
  std::set<mir::Node *> visited;
  std::vector<mir::Node *> res;

  auto adj_list = BuildOperationAdjList();

  for (auto adj : adj_list) {
    if (visited.find(adj.first) == visited.end()) {
      SortHelper(adj_list, adj.first, &visited, &res);
    }
  }

  return res;
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
