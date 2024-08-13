// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/graph_utils.h"

#include <glog/logging.h>

#include <deque>
#include <functional>
#include <set>
#include <stack>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/utils/dot_lang.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace common {

namespace {

void DFSSortUtil(const GraphNode *node, std::vector<GraphNode *> *order) {}

std::vector<GraphNode *> DFSSort(const std::vector<GraphNode *> &nodes) {
  PADDLE_THROW(::common::errors::Unimplemented("Not Implemented"));
  return {};
}

}  // namespace

std::set<GraphNode *> Graph::dependencies(
    const std::vector<GraphNode *> &targets) {
  // A naive implementation.
  std::set<GraphNode *> _targets(targets.begin(), targets.end());
  std::set<GraphNode *> res;
  int targets_count = 0;
  while (targets_count != _targets.size()) {
    targets_count = _targets.size();
    for (auto *node : nodes()) {
      if (_targets.count(node)) continue;
      for (auto &edge : node->outlinks()) {
        if (_targets.count(edge->sink())) {
          res.insert(edge->sink());
          _targets.insert(edge->sink());
        }
      }
    }
  }
  return res;
}

std::vector<const GraphNode *> Graph::nodes() const {
  std::vector<const GraphNode *> res;
  for (auto &s : nodes_) res.push_back(s.get());
  return res;
}
std::vector<GraphNode *> Graph::nodes() {
  std::vector<GraphNode *> res;
  for (auto &s : nodes_) res.push_back(s.get());
  return res;
}

std::tuple<std::vector<GraphNode *>, std::vector<GraphEdge *>>
Graph::topological_order() const {
  std::vector<GraphNode *> node_order;
  std::vector<GraphEdge *> edge_order;
  std::deque<GraphNode *> queue;

  // collect indegree.
  std::map<std::string, int> indegree;
  for (auto *n : nodes()) {
    indegree[n->id()] = n->inlinks().size();
  }

  // insert start points first.
  for (auto *n : start_points()) {
    queue.push_back(&Reference(n));
  }

  // start to visit
  int count = 0;
  while (!queue.empty()) {
    auto *top_node = queue.front();
    top_node->set_index(count);
    node_order.push_back(top_node);
    count++;

    queue.pop_front();

    for (auto &edge : top_node->outlinks()) {
      PADDLE_ENFORCE_EQ(edge->source(),
                        top_node,
                        ::common::errors::InvalidArgument(
                            "The edge's source is not equal to the top node."));
      edge_order.push_back(edge.get());
      auto *sink = edge->sink();
      if ((--indegree[sink->id()]) == 0) {
        queue.push_back(sink);
      }
    }
  }

  PADDLE_ENFORCE_EQ(node_order.size(),
                    nodes().size(),
                    ::common::errors::InvalidArgument(
                        "The node_order size is not equal to the nodes size."));

  return std::make_tuple(node_order, edge_order);
}

std::vector<GraphNode *> Graph::dfs_order() {
  return std::vector<GraphNode *>();
}

std::vector<const GraphNode *> Graph::start_points() const {
  std::vector<const GraphNode *> res;
  for (auto *node : nodes()) {
    if (node->inlinks().empty()) res.push_back(node);
  }
  return res;
}

std::vector<GraphNode *> Graph::start_points() {
  std::vector<GraphNode *> res;
  for (auto *node : nodes()) {
    if (node->inlinks().empty()) res.push_back(node);
  }
  return res;
}

GraphNode *Graph::RegisterNode(size_t key, GraphNode *node) {
  registry_.emplace(key, node);
  nodes_.emplace_back(node);
  return node;
}

GraphNode *Graph::RegisterNode(const std::string &key, GraphNode *node) {
  return RegisterNode(std::hash<std::string>{}(key), node);
}

GraphNode *Graph::RetrieveNode(size_t key) const {
  auto it = registry_.find(key);
  return it == registry_.end() ? nullptr : it->second;
}

GraphNode *Graph::RetrieveNode(const std::string &key) const {
  return RetrieveNode(std::hash<std::string>()(key));
}

std::string Graph::Visualize() const {
  utils::DotLang dot;

  // 1. create nodes
  for (auto &node : nodes_) {
    dot.AddNode(node->id(), {}, "", "", true);
  }

  // 2. link each other
  for (auto &source : nodes_) {
    for (auto &sink : source->outlinks()) {
      dot.AddEdge(source->id(), sink->sink()->id(), {});
    }
  }

  return dot();
}

void Graph::ClearUnlinkedNodes(
    absl::flat_hash_map<std::string, std::vector<int>> *shape_dict,
    absl::flat_hash_map<std::string, Type> *type_dict,
    absl::flat_hash_map<std::string, std::string> *layout_dict) {
  PADDLE_ENFORCE_NOT_NULL(
      shape_dict,
      ::common::errors::InvalidArgument(
          "The shpe_dict %s is null,please change", shape_dict));
  PADDLE_ENFORCE_NOT_NULL(
      type_dict,
      ::common::errors::InvalidArgument(
          "The type_dict %s is null,please change ", type_dict));
  PADDLE_ENFORCE_NOT_NULL(
      layout_dict,
      ::common::errors::InvalidArgument(
          "The layout_dict%s is null,please change", layout_dict));
  for (auto it = nodes_.begin(); it < nodes_.end(); ++it) {
    auto node = *it;
    if (node->inlinks().empty() && node->outlinks().empty()) {
      VLOG(2) << "delete unlinked node: " << node->id();
      nodes_.erase(it);
      if (shape_dict->count(node->id())) {
        shape_dict->erase(node->id());
      }
      if (type_dict->count(node->id())) {
        type_dict->erase(node->id());
      }
      if (layout_dict->count(node->id())) {
        layout_dict->erase(node->id());
      }
      --it;
    }
  }
}

const char *GraphNode::__type_info__ = "GraphNode";

bool GraphEdgeCompare::operator()(const Shared<GraphEdge> &a,
                                  const Shared<GraphEdge> &b) const {
  if (a->source()->id() == b->source()->id()) {
    if (a->sink()->id() == b->sink()->id()) {
      return a->index() < b->index();
    }
    return a->sink()->id() > b->sink()->id();
  }
  return a->source()->id() < b->source()->id();
}

std::set<GraphNode *> Graph::CollectNodes(
    std::function<bool(const cinn::common::GraphNode *)> &&teller) {
  std::set<GraphNode *> res;
  for (auto *node : nodes()) {
    if (teller(node)) res.insert(node);
  }
  return res;
}

}  // namespace common
}  // namespace cinn
