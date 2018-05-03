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

/*
 * Data flow graph is an pass that build the basic graph. It contains a graph
 * and the iterators that enable the iteration over the graph.
 */

#pragma once

#include <deque>
#include <stack>
#include <unordered_set>
#include "paddle/fluid/inference/analysis/graph_traits.h"
#include "paddle/fluid/inference/analysis/node.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {
namespace analysis {

struct DataFlowGraph {
  NodeMap nodes;
  std::vector<Node *> inputs;
  std::vector<Node *> outputs;
};

/*
 * An graph trait help to traverse the graph using BFS.
 * The BFS start from a graph's inputs, the graph should be fully-connected, so
 * that the iterator can reach the end.
 */
template <>
struct GraphTraits<DataFlowGraph> {
  // BFS iterator on nodes.
  struct NodesBFSIterator : public std::forward_iterator_tag {
    NodesBFSIterator() = default;
    explicit NodesBFSIterator(const std::vector<Node *> &source);
    explicit NodesBFSIterator(NodesBFSIterator &&other);
    // NOTE Heavy to use.
    explicit NodesBFSIterator(const NodesBFSIterator &other);

    Node &operator*();
    NodesBFSIterator &operator++();
    Node *operator->();
    // TODO(Superjomn) current implementation just compare the first
    // element, need to compare the graph and all the elements in the queue and
    // set.
    NodesBFSIterator &operator=(const NodesBFSIterator &other);
    bool operator==(const NodesBFSIterator &other);

   private:
    std::deque<Node *> queue_;
    std::unordered_set<Node *> visited_;
  };

  // DFS iterator on nodes.
  struct NodesDFSIterator : public std::iterator {
    NodesDFSIterator() = default;
    NodesDFSIterator(std::vector<Node *> source);
    NodesDFSIterator(NodesDFSIterator &&other);
    NodesDFSIterator(const NodesDFSIterator &other);

    Node &operator*();
    NodesDFSIterator &operator++();
    // TODO(Superjomn) current implementation just compare the first
    // element, need to compare the graph and all the elements in the queue and
    // set.
    NodesDFSIterator &operator=(const NodesDFSIterator &other);
    bool operator==(const NodesDFSIterator &other);
    Node *operator->();

   private:
    std::stack<Node *> stack_;
    std::unordered_set<Node *> visited_;
  };

  GraphTraits(const DataFlowGraph &graph) : graph_(graph) {}

  // default use BFS to visit the nodes.
  iterator_range nodes() {
    return iterator_range(nodes_bfs_begin(), nodes_bfs_end());
  }
  iterator_range nodes_in_BFS() {
    return iterator_range(nodes_bfs_begin(), nodes_bfs_end());
  }
  iterator_range nodes_in_DFS() {
    return iterator_range(nodes_dfs_begin(), nodes_dfs_end());
  }

 private:
  NodesBFSIterator nodes_bfs_begin() { return NodesBFSIterator(graph_.inputs); }
  NodesBFSIterator nodes_bfs_end() { return NodesBFSIterator(); }
  NodesDFSIterator nodes_dfs_begin() { return NodesDFSIterator(graph_.inputs); }
  NodesDFSIterator nodes_dfs_end() { return NodesDFSIterator(); }

 private:
  const DataFlowGraph &graph_;
};
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
