// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#pragma once
#include <string>
#include <vector>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/stmt.h"

namespace cinn {
namespace ir {
namespace analyzer {

// Dep detail: RAW | WAW | WAR
enum class DepKind { DEP, NO_DEP };

// Var or Tensor name
struct DepData {
  std::string name;
};

struct StmtCompare {
  bool operator()(const ir::stmt::StmtRef& a,
                  const ir::stmt::StmtRef& b) const {
    return a.get() < b.get();
  }
};

struct DepDataCompare {
  bool operator()(const DepData& a, const DepData& b) const {
    return a.name < b.name;
  }
};

/**
 * DataDependencyGraph is a graph data structure where graph nodes are
 * stmts in a block, and edges are data dependences between the nodes.
 *
 * This graph can be used in scenarios where pass may change the order of
 * stmts which containing store/load operation. Using this graph to analyze
 * the dependences of stmts and keep the topological order.

 * Examples:
 * const auto &dep_graph = DataDependencyGraph(stmts);
 * DepKind res = dep_graph.HasDependency(stmt1, stmt2);
 */
class DataDependencyGraph {
 public:
  explicit DataDependencyGraph(const std::vector<ir::stmt::StmtRef>& stmts)
      : stmts_(stmts) {
    BuildGraphByStmts();
  }

  // Returns DepKind::DEP if there is a path in the data dependency graph from
  // node src to node dst. Returns DepKind::NO_DEP otherwise. src and dst, are
  // expected to be from the same block.
  DepKind HasDependency(const ir::stmt::StmtRef& src,
                        const ir::stmt::StmtRef& dst) const;

  // Node represents a node in the graph. A Node is a stmt which contains
  // loads/stores.
  struct Node {
    // The unique identifier of this node in the graph.
    unsigned id;
    // The top-level statement which is (or contains) a load/store.
    stmt::StmtRef stmt;
    // Set of load data.
    std::set<DepData, DepDataCompare> loads;
    // Set of store data.
    std::set<DepData, DepDataCompare> stores;

    Node() = default;
    Node(unsigned id, const ir::stmt::StmtRef& stmt);
  };

  struct Edge {
    // The id of the node at the other end of the edge.
    // If this edge is stored in Edge = Node.in_edges_[i], then
    // 'Node.in_edges_[i].id' is the identifier of the source node of the edge.
    // If this edge is stored in Edge = Node.out_edges_[i], then
    // 'Node.out_edges_[i].id' is the identifier of the dest node of the edge.
    unsigned id;
    // The DepInfo of this edge represents a dependence, each data represents a
    // dependence between graph nodes.
    std::map<DepData, std::vector<DepKind>, DepDataCompare> dep_info;
  };

  void Print(int log_level = 0) const;

 private:
  // Initializes the dependence graph based on stmts in block.
  void BuildGraphByStmts();

  // Returns true iff there is an edge from node src_id to node dst_id. Returns
  // false otherwise.
  bool HasEdge(unsigned src_id, unsigned dst_id);

  // Adds an edge from node src_id to node dst_id for dep_info.
  void AddEdge(
      unsigned src_id,
      unsigned dst_id,
      const std::map<DepData, std::vector<DepKind>, DepDataCompare>& dep_info);

  // Removes an edge from node src_id to node dst_id for dep_data.
  void RemoveEdge(unsigned src_id, unsigned dst_id, const DepData& dep_data);

  // The next unique identifier to use for newly created graph nodes.
  unsigned next_node_id_ = 0;

  std::vector<ir::stmt::StmtRef> stmts_;
  std::map<ir::stmt::StmtRef, unsigned, StmtCompare> stmt_to_node_ids_;
  std::unordered_map<unsigned, Node> nodes_;

  // Map from node id to list of edges.
  std::unordered_map<unsigned, std::vector<Edge>> in_edges_;
  std::unordered_map<unsigned, std::vector<Edge>> out_edges_;
};

std::ostream& operator<<(std::ostream& os, const DepKind& dep_kind);

}  // namespace analyzer
}  // namespace ir
}  // namespace cinn
