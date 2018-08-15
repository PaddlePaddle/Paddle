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
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/inference/analysis/graph_traits.h"
#include "paddle/fluid/inference/analysis/node.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * DataFlowGraph - A container of Value and Function Nodes.
 *
 * This is the base graph for any other type of graphs, such as SSA or CFG.
 */
struct DataFlowGraph {
  NodeMap nodes;
  std::vector<Node *> inputs;
  std::vector<Node *> outputs;

  // Extract inputs and outputs of the graph.
  void Build();

  // Output a DOT graph file for debug.
  std::string DotString() const;

  std::string HumanReadableInfo(bool show_values = true,
                                bool show_functions = true) const;

 private:
  // Remove duplicate edges and so on.
  void Clean();
};

/*
 * An graph trait help to traverse the graph using BFS.
 * The BFS start from a graph's inputs, the graph should be fully-connected, so
 * that the iterator can reach the end.
 */
template <>
struct GraphTraits<DataFlowGraph> {
  // BFS iterator on nodes.
  struct NodesBFSIterator
      : public std::iterator<std::forward_iterator_tag, Node *> {
    NodesBFSIterator() = default;
    explicit NodesBFSIterator(const std::vector<Node *> &source);
    // NodesBFSIterator(NodesBFSIterator &&other) noexcept;
    // NOTE Heavy to use.
    NodesBFSIterator(const NodesBFSIterator &other);

    Node &operator*();
    NodesBFSIterator &operator++();
    Node *operator->();
    // TODO(Superjomn) current implementation just compare the first
    // element, need to compare the graph and all the elements in the queue and
    // set.
    NodesBFSIterator &operator=(const NodesBFSIterator &other);
    bool operator==(const NodesBFSIterator &other);
    bool operator!=(const NodesBFSIterator &other) { return !(*this == other); }

   private:
    std::deque<Node *> queue_;
    std::unordered_set<Node *> visited_;
  };

  // DFS iterator on nodes.
  struct NodesDFSIterator
      : public std::iterator<std::forward_iterator_tag, Node *> {
    NodesDFSIterator() = default;
    explicit NodesDFSIterator(const std::vector<Node *> &source);
    // NodesDFSIterator(NodesDFSIterator &&other) noexcept;
    NodesDFSIterator(const NodesDFSIterator &other);

    Node &operator*();
    NodesDFSIterator &operator++();
    // TODO(Superjomn) current implementation just compare the first
    // element, need to compare the graph and all the elements in the queue and
    // set.
    NodesDFSIterator &operator=(const NodesDFSIterator &other);
    bool operator==(const NodesDFSIterator &other);
    bool operator!=(const NodesDFSIterator &other) { return !(*this == other); }
    Node *operator->();

   private:
    std::stack<Node *> stack_;
    std::unordered_set<Node *> visited_;
  };

  // Topological sorting iterator on nodes.
  struct NodesTSIterator
      : public std::iterator<std::forward_iterator_tag, Node *> {
    NodesTSIterator() = default;
    explicit NodesTSIterator(const std::vector<Node *> &source);
    NodesTSIterator(NodesTSIterator &&other)
        : sorted_(std::move(other.sorted_)), cursor_(other.cursor_) {
      other.cursor_ = 0;
    }
    NodesTSIterator(const NodesTSIterator &other);

    Node &operator*();
    NodesTSIterator &operator++();
    // TODO(Superjomn) current implementation just compare the first
    // element, need to compare the graph and all the elements in the queue and
    // set.
    NodesTSIterator &operator=(const NodesTSIterator &other);
    bool operator==(const NodesTSIterator &other);
    bool operator!=(const NodesTSIterator &other) { return !(*this == other); }
    Node *operator->();

   private:
    std::vector<Node *> sorted_;
    size_t cursor_{0};
  };

  explicit GraphTraits(DataFlowGraph *graph) : graph_(graph) {}

  // default use BFS to visit the nodes.
  iterator_range<NodesBFSIterator> nodes() {
    return iterator_range<NodesBFSIterator>(nodes_bfs_begin(), nodes_bfs_end());
  }
  iterator_range<NodesBFSIterator> nodes_in_BFS() {
    return iterator_range<NodesBFSIterator>(nodes_bfs_begin(), nodes_bfs_end());
  }
  iterator_range<NodesDFSIterator> nodes_in_DFS() {
    return iterator_range<NodesDFSIterator>(nodes_dfs_begin(), nodes_dfs_end());
  }
  iterator_range<NodesTSIterator> nodes_in_TS() {
    return iterator_range<NodesTSIterator>(nodes_ts_begin(), nodes_ts_end());
  }

 private:
  NodesBFSIterator nodes_bfs_begin() {
    return NodesBFSIterator(graph_->inputs);
  }
  NodesBFSIterator nodes_bfs_end() { return NodesBFSIterator(); }

  NodesDFSIterator nodes_dfs_begin() {
    return NodesDFSIterator(graph_->inputs);
  }
  NodesDFSIterator nodes_dfs_end() { return NodesDFSIterator(); }

  NodesTSIterator nodes_ts_begin() { return NodesTSIterator(graph_->inputs); }
  NodesTSIterator nodes_ts_end() { return NodesTSIterator(); }

 private:
  DataFlowGraph *graph_;
};

// Extract the inputs and outputs of a graph. The inputs and outputs of a
// sub-graph is the inputs nodes and output nodes that doesn't inside the
// sub-graph.
std::pair<std::vector<Node *>, std::vector<Node *>>
ExtractInputAndOutputOfSubGraph(std::vector<Node *> &graph);  // NOLINT

void FilterRedundantOutputOfSubGraph(DataFlowGraph *graph);
}  // namespace analysis
}  // namespace inference
}  // namespace paddle
