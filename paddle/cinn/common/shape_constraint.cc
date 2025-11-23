// Copyright (c) 2025 CINN Authors. All Rights Reserved.
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
#include "paddle/cinn/common/shape_constraint.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/utils/ir_compare.h"

namespace cinn {
namespace common {
ShapeConstraintManager& ShapeConstraintManager::Instance() {
  thread_local static ShapeConstraintManager instance;
  return instance;
}
void ShapeConstraintManager::Init(
    const symbol::ConstraintsManager& constraint) {
  InitEqualExprs(constraint.equals());
  InitBroadcastableExprs(constraint.broadcastables());
}

void ShapeConstraintManager::InitEqualExprs(
    const ::common::UnionFindSet<symbol::DimExpr>& equal_dim_exprs) {
  equal_dim_exprs_ = equal_dim_exprs;
  // need to convert DimExpr to IndexExpr
  DimExprConverter cvt;
  ::common::UnionFindSet<ir::IndexExpr, IndexExprDirectCompare> transfer_set;
  equal_dim_exprs.VisitCluster(
      [&transfer_set, &cvt](const std::vector<symbol::DimExpr>& cluster) {
        ir::IndexExpr first_elem;
        for (size_t i = 0; i < cluster.size(); ++i) {
          if (i == 0) {
            first_elem = cvt.ConvertToIrExpr(cluster[i]).as_index();
          } else {
            transfer_set.Union(cvt.ConvertToIrExpr(cluster[i]).as_index(),
                               first_elem);
          }
        }
      });
  equal_exprs_ = std::move(transfer_set);
}

void ShapeConstraintManager::InitEqualExprs(
    const ::common::UnionFindSet<ir::IndexExpr, IndexExprDirectCompare>&
        equal_exprs) {
  equal_exprs_ = equal_exprs;
}
bool ShapeConstraintManager::IsEqual(const ir::IndexExpr& lhs,
                                     const ir::IndexExpr& rhs) {
  return equal_exprs_.HasSameRoot(lhs, rhs);
}

bool ShapeConstraintManager::IsBroadcastable(const ir::IndexExpr& lhs,
                                             const ir::IndexExpr& rhs) {
  return broadcastable_exprs_.HasEdge(lhs, rhs);
}

bool ShapeConstraintManager::IsBroadcastable(
    const std::vector<ir::IndexExpr>& vec) {
  for (size_t i = 0; i < vec.size(); ++i) {
    for (size_t j = i + 1; j < vec.size(); ++j) {
      if (!IsBroadcastable(vec[i], vec[j])) return false;
    }
  }
  return true;
}

void ShapeConstraintManager::InitBroadcastableExprs(
    const std::unordered_set<symbol::Broadcastable<symbol::DimExpr>>&
        broadcastable_dim_exprs) {
  auto AddEdgesForOperands =
      [this](const std::vector<symbol::DimExpr>& operands) {
        for (size_t i = 0; i < operands.size(); ++i) {
          for (size_t j = i + 1; j < operands.size(); ++j) {
            broadcastable_exprs_.AddEdge(operands[i], operands[j]);
          }
        }
      };
  auto AddEdgesBetweenOperands =
      [this](const std::vector<symbol::DimExpr>& lhs_operands,
             const std::vector<symbol::DimExpr>& rhs_operands) {
        for (auto& lhs_op : lhs_operands) {
          for (auto& rhs_op : rhs_operands) {
            broadcastable_exprs_.AddEdge(lhs_op, rhs_op);
          }
        }
      };
  for (auto& broadcastable_dim_expr : broadcastable_dim_exprs) {
    auto [lhs, rhs] = *broadcastable_dim_expr;
    if (lhs.isa<symbol::Broadcast<symbol::DimExpr>>() &&
        rhs.isa<symbol::Broadcast<symbol::DimExpr>>()) {
      auto lhs_operands =
          lhs.dyn_cast<symbol::Broadcast<symbol::DimExpr>>().operands.vector();
      auto rhs_operands =
          rhs.dyn_cast<symbol::Broadcast<symbol::DimExpr>>().operands.vector();
      AddEdgesForOperands(lhs_operands);
      AddEdgesForOperands(rhs_operands);
      AddEdgesBetweenOperands(lhs_operands, rhs_operands);
    } else if (lhs.isa<symbol::Broadcast<symbol::DimExpr>>()) {
      auto lhs_operands =
          lhs.dyn_cast<symbol::Broadcast<symbol::DimExpr>>().operands.vector();
      AddEdgesForOperands(lhs_operands);
      AddEdgesBetweenOperands(lhs_operands, {rhs});
    } else if (rhs.isa<symbol::Broadcast<symbol::DimExpr>>()) {
      auto rhs_operands =
          rhs.dyn_cast<symbol::Broadcast<symbol::DimExpr>>().operands.vector();
      AddEdgesForOperands(rhs_operands);
      AddEdgesBetweenOperands({lhs}, rhs_operands);
    } else {
      broadcastable_exprs_.AddEdge(lhs, rhs);
    }
  }
}

std::ostream& operator<<(std::ostream& stream,
                         const ShapeConstraintManager& constraints_manager) {
  stream << "----------Equal Constraints Expr Clusters---------" << std::endl;
  constraints_manager.equal_exprs_.VisitCluster([&](const auto& cluster) {
    stream << "  {" << std::endl;
    for (const auto& expr : cluster) {
      stream << "  " << expr << std::endl;
    }
    stream << "  }" << std::endl;
  });
  stream << "--------------------------------------------------" << std::endl;
  stream << "--------Equal Constraints DimExpr Clusters--------" << std::endl;
  constraints_manager.equal_dim_exprs_.VisitCluster([&](const auto& cluster) {
    stream << "  {" << std::endl;
    for (const auto& dim_expr : cluster) {
      stream << "  " << dim_expr << std::endl;
    }
    stream << "  }" << std::endl;
  });
  stream << "--------------------------------------------------" << std::endl;
  stream << "--------Broadcastable Constraints Exprs-----------" << std::endl;
  for (auto& broadcastable_dim_expr :
       constraints_manager.broadcastable_exprs_.GetBroadcastableExprs()) {
    stream << broadcastable_dim_expr.first << " -> ";
    for (auto& expr : broadcastable_dim_expr.second) {
      stream << expr << ", ";
    }
    stream << std::endl;
  }
  stream << "--------------------------------------------------" << std::endl;

  return stream;
}

}  // namespace common
}  // namespace cinn
