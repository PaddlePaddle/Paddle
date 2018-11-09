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

#include "paddle/fluid/framework/ir/graph_helper.h"
#include <algorithm>
#include <deque>
#include <unordered_set>

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

bool HasCircleInternal(
    const std::map<ir::Node *, std::unordered_set<ir::Node *>> &adj_list) {
  std::unordered_set<ir::Node *> visited;
  std::unordered_set<ir::Node *> in_trace;
  for (auto &adj : adj_list) {
    if (HasCircleHelper(adj.first, adj_list, &visited, &in_trace)) {
      return true;
    }
  }
  return false;
}
}  // namespace

bool HasCircle(const Graph &graph) {
  return HasCircleInternal(BuildOperationAdjList(graph));
}

std::vector<ir::Node *> TopologySortOperations(const Graph &graph) {
  std::map<ir::Node *, std::unordered_set<ir::Node *>> adj_list =
      BuildOperationAdjList(graph);
  PADDLE_ENFORCE(!HasCircleInternal(adj_list));
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
        VLOG(4) << "adj " << adj_n->Name() << reinterpret_cast<void *>(adj_n)
                << " -> " << n->Name() << reinterpret_cast<void *>(n)
                << "  via " << var->Name() << reinterpret_cast<void *>(var);
        adj_list[n].insert(adj_n);
      }
    }
  }
  return adj_list;
}

size_t GraphNum(const Graph &graph) {
  std::unordered_set<ir::Node *> nodes = graph.Nodes();
  std::unordered_set<ir::Node *> visited_nodes;
  visited_nodes.reserve(nodes.size());
  std::deque<ir::Node *> q_nodes;
  std::vector<std::unordered_set<ir::Node *>> graph_nodes;
  std::unordered_set<ir::Node *> g_nodes;
  // q_set used to record records in the queue.
  std::unordered_set<ir::Node *> q_set;
  size_t graph_count = 0;

  auto traverse_nodes = [&visited_nodes, &q_nodes,
                         &q_set](const std::vector<ir::Node *> &nodes) {
    for (auto n : nodes) {
      if (visited_nodes.count(n) == 0 && q_set.count(n) == 0) {
        q_nodes.push_back(n);
        q_set.insert(n);
      }
    }
  };

  while (visited_nodes.size() != nodes.size()) {
    if (!q_nodes.empty()) {
      auto cur_node = q_nodes.front();
      q_nodes.pop_front();
      q_set.erase(cur_node);
      visited_nodes.insert(cur_node);
      g_nodes.insert(cur_node);
      traverse_nodes(cur_node->inputs);
      traverse_nodes(cur_node->outputs);
    } else {
      ++graph_count;
      if (g_nodes.size()) {
        graph_nodes.emplace_back(g_nodes);
      }
      g_nodes.clear();
      for (auto &n : nodes) {
        if (visited_nodes.count(n) == 0) {
          q_nodes.push_back(n);
          q_set.insert(n);
          break;
        }
      }
    }
  }

  if (g_nodes.size()) {
    graph_nodes.emplace_back(g_nodes);
  }

  if (VLOG_IS_ON(10)) {
    VLOG(10) << "graph_num: " << graph_nodes.size();
    for (auto &g_n : graph_nodes) {
      VLOG(10) << "graph_nodes: " << g_n.size();
      if (g_n.size() < 10) {
        std::stringstream out;
        for (auto &node : g_n) {
          out << "\nNode: " << node->Name() << " in [";
          for (auto &n : node->inputs) {
            out << n->Name() << ", ";
          }
          out << "], out[";
          for (auto &n : node->outputs) {
            out << n->Name() << ", ";
          }
          out << "]";
        }
        VLOG(10) << out.str();
      }
    }
  }

  return graph_count;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
