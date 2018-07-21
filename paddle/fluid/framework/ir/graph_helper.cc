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

#include <algorithm>
#include <unordered_set>

#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace ir {
namespace {
void SortHelper(
    const std::map<ir::Node *, std::unordered_set<ir::Node *>> &adj_list,
    ir::Node *node, std::unordered_set<ir::Node *> *visited,
    std::vector<ir::Node *> *ret) {
  visited->insert(node);

  for (auto adj : adj_list.at(node)) {
    if (visited->find(adj) == visited->end()) {
      SortHelper(adj_list, adj, visited, ret);
    }
  }

  VLOG(3) << "topology sort insert: " << node->Name()
          << reinterpret_cast<void *>(node) << " input " << node->inputs.size();
  ret->push_back(node);
}

bool HasCircleHelper(
    ir::Node *node,
    const std::map<ir::Node *, std::unordered_set<ir::Node *>> &adj_list,
    std::unordered_set<ir::Node *> *visited,
    std::unordered_set<ir::Node *> *in_trace) {
  if (visited->find(node) == visited->end()) {
    visited->insert(node);
    in_trace->insert(node);

    for (ir::Node *in : adj_list.at(node)) {
      if (visited->find(in) == visited->end() &&
          HasCircleHelper(in, adj_list, visited, in_trace)) {
        return true;
      } else if (in_trace->find(in) != in_trace->end()) {
        return true;
      }
    }
  }
  in_trace->erase(node);
  return false;
}
}  // namespace

bool HasCircle(const Graph &graph) {
  std::map<ir::Node *, std::unordered_set<ir::Node *>> adj_list =
      BuildOperationAdjList(graph);

  std::unordered_set<ir::Node *> visited;
  std::unordered_set<ir::Node *> in_trace;
  for (auto &adj : adj_list) {
    if (HasCircleHelper(adj.first, adj_list, &visited, &in_trace)) {
      return true;
    }
  }
  return false;
}

std::vector<ir::Node *> TopologySortOperations(const Graph &graph) {
  std::map<ir::Node *, std::unordered_set<ir::Node *>> adj_list =
      BuildOperationAdjList(graph);
  std::unordered_set<ir::Node *> visited;
  std::vector<ir::Node *> ret;
  for (auto adj : adj_list) {
    if (visited.find(adj.first) == visited.end()) {
      SortHelper(adj_list, adj.first, &visited, &ret);
    }
  }
  return ret;
}

std::map<ir::Node *, std::unordered_set<ir::Node *>> BuildOperationAdjList(
    const Graph &graph) {
  std::map<ir::Node *, std::unordered_set<ir::Node *>> adj_list;

  for (auto &n : graph.Nodes()) {
    if (n->NodeType() != ir::Node::Type::kOperation) continue;
    if (adj_list.find(n) == adj_list.end()) {
      adj_list[n] = std::unordered_set<ir::Node *>();
    }
    for (auto &var : n->inputs) {
      for (auto &adj_n : var->inputs) {
        PADDLE_ENFORCE(adj_n->NodeType() == ir::Node::Type::kOperation);
        adj_list[n].insert(adj_n);
        VLOG(3) << "adj " << adj_n->Name() << reinterpret_cast<void *>(adj_n)
                << " -> " << n->Name() << reinterpret_cast<void *>(n)
                << "  via " << var->Name() << reinterpret_cast<void *>(var);
      }
    }
  }
  return adj_list;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
