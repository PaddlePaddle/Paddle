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

#include "paddle/cinn/ir/ir_analyzer/data_dependency_graph.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/common/bfs_walker.h"

namespace cinn {
namespace ir {
namespace analyzer {

std::ostream& operator<<(std::ostream& os, const DepKind& dep_kind) {
  if (dep_kind == DepKind::DEP) {
    os << " {DepKind : DEP} ";
  } else if (dep_kind == DepKind::NO_DEP) {
    os << " {DepKind : NO_DEP} ";
  } else {
    os << "Not support DepKind for output!";
  }
  return os;
}

// MemRefCollector walks stmts collects load and store.
class MemRefCollector : public ir::stmt::StmtVisitor<>,
                        public ir::IRMutator<const ir::Expr*> {
 public:
  void VisitStmt(const ir::stmt::StmtRef& stmt) {
    ir::stmt::StmtVisitor<>::VisitStmt(stmt);
  }

  std::set<DepData, DepDataCompare> GetLoads() { return loads_; }
  std::set<DepData, DepDataCompare> GetStores() { return stores_; }

 private:
#define __(stmt__) void VisitStmt(const ir::stmt::stmt__& stmt) override;
  NODETY_FORALL_STMT(__)
#undef __

  void Visit(const ir::Load* op, const ir::Expr* expr) override {
    auto tensor_node = op->tensor.As<ir::_Tensor_>();
    loads_.insert({tensor_node->buffer->name});
    ir::IRMutator<const ir::Expr*>::Visit(op, expr);
  }

  void Visit(const ir::_Var_* op, const ir::Expr* expr) override {
    const std::unordered_set<std::string> gpu_axis = {"blockIdx.x",
                                                      "blockIdx.y",
                                                      "blockIdx.z",
                                                      "threadIdx.x",
                                                      "threadIdx.y",
                                                      "threadIdx.z"};
    if (op->is_symbolic_constant) return;
    if (gpu_axis.count(op->name)) return;
    loads_.insert({op->name});
    ir::IRMutator<const ir::Expr*>::Visit(op, expr);
  }

  void Visit(const ir::Call* op, const ir::Expr* expr) override {
    for (auto write_arg : op->write_args) {
      if (write_arg.As<ir::_Var_>()) {
        stores_.insert({write_arg.As<ir::_Var_>()->name});
      } else if (write_arg.As<ir::Load>()) {
        auto load_node = write_arg.As<ir::Load>();
        auto tensor_node = load_node->tensor.As<ir::_Tensor_>();
        stores_.insert({tensor_node->buffer->name});
      } else {
        VLOG(6) << "Not support type in write arguments: \n" << write_arg;
      }
    }
    ir::IRMutator<const ir::Expr*>::Visit(op, expr);
  }

  std::set<DepData, DepDataCompare> loads_;
  std::set<DepData, DepDataCompare> stores_;
};

void MemRefCollector::VisitStmt(const ir::stmt::Let& stmt) {
  if (stmt->symbol().As<ir::_Var_>())
    stores_.insert({stmt->symbol().As<ir::_Var_>()->name});
  ir::IRMutator<const ir::Expr*>::Visit(&stmt->body(), &stmt->body());
}

void MemRefCollector::VisitStmt(const ir::stmt::Store& stmt) {
  auto tensor_node = stmt->tensor().As<ir::_Tensor_>();
  if (tensor_node->buffer.get()) {
    stores_.insert({tensor_node->buffer->name});
  }
  ir::IRMutator<const ir::Expr*>::Visit(&stmt->value(), &stmt->value());
  for (std::size_t i = 0; i < stmt->indices().size(); i++) {
    ir::IRMutator<const ir::Expr*>::Visit(&stmt->indices()[i],
                                          &stmt->indices()[i]);
  }
}

void MemRefCollector::VisitStmt(const ir::stmt::IfThenElse& stmt) {
  ir::IRMutator<const ir::Expr*>::Visit(&stmt->condition(), &stmt->condition());
  ir::stmt::StmtVisitor<>::VisitBlock(stmt->true_case());
  if (stmt->false_case().defined())
    ir::stmt::StmtVisitor<>::VisitBlock(stmt->false_case());
}

void MemRefCollector::VisitStmt(const ir::stmt::For& stmt) {
  ir::IRMutator<const ir::Expr*>::Visit(&stmt->min(), &stmt->min());
  ir::IRMutator<const ir::Expr*>::Visit(&stmt->extent(), &stmt->extent());
  ir::stmt::StmtVisitor<>::VisitBlock(stmt->body());
}

void MemRefCollector::VisitStmt(const ir::stmt::Schedule& stmt) {
  ir::stmt::StmtVisitor<>::VisitBlock(stmt->body());
}

void MemRefCollector::VisitStmt(const ir::stmt::Evaluate& stmt) {
  ir::IRMutator<const ir::Expr*>::Visit(&stmt->value(), &stmt->value());
}

void MemRefCollector::VisitStmt(const ir::stmt::Alloc& stmt) {}
void MemRefCollector::VisitStmt(const ir::stmt::Free& stmt) {}

DataDependencyGraph::Node::Node(unsigned id, const ir::stmt::StmtRef& stmt)
    : id(id), stmt(stmt) {
  MemRefCollector collector;
  collector.VisitStmt(stmt);
  loads = collector.GetLoads();
  stores = collector.GetStores();
}

DepKind DataDependencyGraph::HasDependency(const ir::stmt::StmtRef& src,
                                           const ir::stmt::StmtRef& dst) const {
  PADDLE_ENFORCE_GT(stmt_to_node_ids_.count(src),
                    0,
                    ::common::errors::InvalidArgument(
                        "stmt_to_node_ids_ should contain stmt src"));
  PADDLE_ENFORCE_GT(stmt_to_node_ids_.count(dst),
                    0,
                    ::common::errors::InvalidArgument(
                        "stmt_to_node_ids_ should contain stmt dst"));
  auto src_id = stmt_to_node_ids_.at(src);
  auto dst_id = stmt_to_node_ids_.at(dst);

  // Run BFS traversal to check if src and dst are reachable.
  DepKind res = DepKind::NO_DEP;
  ::common::BfsWalker<unsigned> bfs_walker(
      [&](unsigned id, const std::function<void(unsigned)> Visit) {
        // Skip if node has no out edges, or have been found already.
        if (out_edges_.count(id) != 0 && res == DepKind::NO_DEP) {
          for (const auto& edge : out_edges_.at(id)) {
            Visit(edge.id);
          }
        }
      });
  bfs_walker(src_id, [&](unsigned id) {
    if (id == dst_id) res = DepKind::DEP;
  });
  return res;
}

void DataDependencyGraph::BuildGraphByStmts() {
  auto BuildNodes = [&]() {
    for (auto& stmt : stmts_) {
      Node node(next_node_id_++, stmt);
      stmt_to_node_ids_.insert({stmt, node.id});
      nodes_.insert({node.id, node});
    }
  };

  auto GetDepInfo = [&](const unsigned src_id, const unsigned dst_id)
      -> std::map<DepData, std::vector<DepKind>, DepDataCompare> {
    auto src = nodes_[src_id];
    auto dst = nodes_[dst_id];
    std::map<DepData, std::vector<DepKind>, DepDataCompare> dep_info;
    for (auto store : src.stores) {
      // RAW
      if (dst.loads.count(store)) {
        dep_info[store].push_back(DepKind::DEP);
      }
      // WAW
      if (dst.stores.count(store)) {
        dep_info[store].push_back(DepKind::DEP);
      }
    }
    for (auto load : src.loads) {
      // WAR
      if (dst.stores.count(load)) {
        dep_info[load].push_back(DepKind::DEP);
      }
    }
    return dep_info;
  };

  auto BuildEdges = [&]() {
    for (unsigned i = 0; i < nodes_.size(); ++i) {
      for (unsigned j = i + 1; j < nodes_.size(); ++j) {
        auto dep_info = GetDepInfo(i, j);
        if (!dep_info.empty()) {
          AddEdge(i, j, dep_info);
        }
      }
    }
  };

  BuildNodes();
  BuildEdges();
}

void DataDependencyGraph::Print(int log_level) const {
  VLOG(log_level) << "DataDependencyGraph";
  VLOG(log_level) << "Nodes size: " << nodes_.size();
  for (const auto& [id, node] : nodes_) {
    auto stmt = node.stmt;
    VLOG(log_level) << "Node" << id << " stmt: " << stmt;
    for (auto load : node.loads) {
      VLOG(log_level) << "Load: " << load.name;
    }
    for (auto store : node.stores) {
      VLOG(log_level) << "Store: " << store.name;
    }
    auto it = in_edges_.find(id);
    if (it != in_edges_.end()) {
      for (const auto& e : it->second)
        for (const auto& value : e.dep_info)
          for (const auto& dep_kind : value.second)
            VLOG(log_level) << "In Edge: \n"
                            << nodes_.at(e.id).stmt << " dst: \n"
                            << nodes_.at(id).stmt << " DepData:\n"
                            << value.first.name << " DepKind: " << dep_kind;
    }
    it = out_edges_.find(id);
    if (it != out_edges_.end()) {
      for (const auto& e : it->second)
        for (const auto& value : e.dep_info)
          for (const auto& dep_kind : value.second)
            VLOG(log_level) << "Out Edge: \n"
                            << nodes_.at(id).stmt << " dst: \n"
                            << nodes_.at(e.id).stmt << " DepData:\n"
                            << value.first.name << " DepKind: " << dep_kind;
    }
  }
}

bool DataDependencyGraph::HasEdge(unsigned src_id, unsigned dst_id) {
  auto CheckEdges = [&](const unsigned id, const std::vector<Edge>& edges) {
    for (const auto edge : edges) {
      if (edge.id == id) return true;
    }
    return false;
  };

  if (out_edges_.count(src_id == 0) || in_edges_.count(dst_id) == 0) {
    return false;
  }
  return CheckEdges(dst_id, out_edges_[src_id]) &&
         CheckEdges(src_id, in_edges_[dst_id]);
}

void DataDependencyGraph::AddEdge(
    unsigned src_id,
    unsigned dst_id,
    const std::map<DepData, std::vector<DepKind>, DepDataCompare>& dep_info) {
  if (!HasEdge(src_id, dst_id)) {
    out_edges_[src_id].push_back({dst_id, dep_info});
    in_edges_[dst_id].push_back({src_id, dep_info});
  }
}

}  // namespace analyzer
}  // namespace ir
}  // namespace cinn
