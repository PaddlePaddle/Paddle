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

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"

namespace paddle {
namespace framework {
namespace ir {

constexpr char kGraphToProgramVarsToRemove[] =
    "__graph_to_program_vars_to_remove__";
constexpr char kGraphToProgramSortKind[] = "__graph_to_program_sort_kind__";

// Compare nodes via node id.
class Graph;

struct NodeComp {
  bool operator()(ir::Node *const &node1, ir::Node *const &node2) const {
    return node1->id() < node2->id();
  }
};

// Test if the graph contains circle.
bool HasCircle(const Graph &graph);

// Check if the var desc of node is consistency.
// The graph may have the same name node, for example, parameter
// is the input of operator and it also is the output of optimizer.
// For the persistable variable, the var_desc of the nodes with
// the same node name should be equal.
bool VarDescIsConsistency(const Graph &graph);

// Find All Circles for debugging,
// store all subgraph in circles.
bool FindCircleSubGraph(const Graph &graph,
                        std::vector<std::vector<ir::Node *>> *circles);

size_t GraphNum(const Graph &graph);

// Topology Sort the operations in the graph from inputs to outputs.
// `graph` cannot contain circle.
std::vector<ir::Node *> TopologySortOperations(const Graph &graph);

// Check whether the topological order of graph ops is unique
bool IsTopologySortOperationsUnique(const Graph &graph);

// Topological sort, but try to DFS.
std::vector<ir::Node *> TopologyDfsSortOperations(const Graph &graph);

// Different kinds to sort the operators in a graph to a sequence.
enum class SortKind {
  // Topological Search
  TS = 0,
  // Topological and Depth First Search
  TDFS
};

// Several kinds of topological sort.
std::vector<Node *> TopologyVariantSort(const Graph &graph, SortKind sort_kind);

// Clean the nodes that doesn't connect to others.
void CleanIndividualNodes(Graph *graph);

// Build an in-link adjacency list of operations for the `graph`.
template <class NodeComparator = ir::NodeComp>
std::map<ir::Node *, std::set<ir::Node *, NodeComparator>, NodeComparator>
BuildOperationAdjList(const Graph &graph) {
  std::map<ir::Node *, std::set<ir::Node *, NodeComparator>, NodeComparator>
      adj_list;

  for (auto &n : graph.Nodes()) {
    if (!n->IsOp()) continue;
    if (adj_list.find(n) == adj_list.end()) {
      adj_list[n] = std::set<ir::Node *, NodeComparator>();
    }
    for (auto &var : n->inputs) {
      for (auto &adj_n : var->inputs) {
        PADDLE_ENFORCE_EQ(adj_n->NodeType(),
                          ir::Node::Type::kOperation,
                          common::errors::InvalidArgument(
                              "Node(%s)'s type(%d) must be kOperation type.",
                              adj_n->Name(),
                              static_cast<int>(adj_n->NodeType())));
        VLOG(4) << "adj " << adj_n->Name() << reinterpret_cast<void *>(adj_n)
                << " -> " << n->Name() << reinterpret_cast<void *>(n)
                << "  via " << var->Name() << reinterpret_cast<void *>(var);
        adj_list[n].insert(adj_n);
      }
    }
  }
  return adj_list;
}

template <typename T>
std::vector<T *> FilterByNodeWrapper(const Graph &graph) {
  std::vector<T *> ret;
  for (ir::Node *n : graph.Nodes()) {
    if (n->IsWrappedBy<T>()) ret.push_back(&n->Wrapper<T>());
  }
  return ret;
}

std::vector<ir::Node *> TopologySortGraphByDescOrder(const Graph &graph);

void GraphToProgram(const Graph &graph,
                    ProgramDesc *p_program,
                    const SortKind *sort_kind = nullptr);

std::vector<std::vector<std::vector<ir::Node::Dep>>> GetOpDependencies(
    const ProgramDesc &program);

}  // namespace ir
}  // namespace framework
}  // namespace paddle
