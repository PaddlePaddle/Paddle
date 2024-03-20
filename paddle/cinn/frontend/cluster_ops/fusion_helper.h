// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/frontend/cluster_ops/group_pattern.h"
#include "paddle/cinn/frontend/cluster_ops/pattern_utils.h"

namespace cinn::frontend::cluster_ops {

class StmtFusionHelper {
 public:
  explicit StmtFusionHelper(const std::vector<const pir::Operation*>& ops,
                   const ShardableAxesInferer& shardable_axes_inferer);

  GroupPattern FuseToGroupPattern();

 private:
  std::vector<StmtPattern> ConvertToStmtPatternVec();
  void SortStmtPatterns(std::vector<StmtPattern>* stmt_patterns);

  std::optional<ErrorGroupPattern> Fuse_IS_x_IS_2_IS(
      std::vector<StmtPattern>* stmt_patterns);

  std::optional<ErrorGroupPattern> Fuse_PS_x_PS_2_PS(
      std::vector<StmtPattern>* stmt_patterns);

  struct FusePolicy_IS_x_PS_2_PS {
    bool FuseCondition(const StmtPattern& upstream,
                              const StmtPattern& downstream);
    std::variant<StmtPattern, ErrorGroupPattern> MergePattern(
        const StmtPattern& upstream, const StmtPattern& downstream);
    std::variant<StmtPattern, ErrorGroupPattern> MergePatternImpl(
        const IS& upstream, const PS& downstream);
    ShardableAxesSignature MergeShardableAxesSignature(
        const IS& upstream, const PS& downstream);
  };

  std::optional<ErrorGroupPattern> Fuse_IS_x_PS_2_PS(
      std::vector<StmtPattern>* stmt_patterns);
  struct FusePolicy_IS_x_R_2_R {
    bool FuseCondition(const StmtPattern& upstream,
                              const StmtPattern& downstream);
    std::variant<StmtPattern, ErrorGroupPattern> MergePattern(
        const StmtPattern& upstream, const StmtPattern& downstream);
    std::variant<StmtPattern, ErrorGroupPattern> MergePatternImpl(
        const IS& upstream, const R& downstream);
  };

  std::optional<ErrorGroupPattern> Fuse_IS_x_R_2_R(
      std::vector<StmtPattern>* stmt_patterns);

  struct FusePolicy_PS_x_R_2_R {
    bool FuseCondition(const StmtPattern& upstream,
                              const StmtPattern& downstream);
    std::variant<StmtPattern, ErrorGroupPattern> MergePattern(
        const StmtPattern& upstream, const StmtPattern& downstream);
    std::variant<StmtPattern, ErrorGroupPattern> MergePatternImpl(
        const PS& upstream, const R& downstream);
  };

  std::optional<ErrorGroupPattern> Fuse_PS_x_R_2_R(
      std::vector<StmtPattern>* stmt_patterns);
  StmtPattern ConvertToStmtPattern(const pir::Operation* op);

  IS ConvertToIS(const pir::Operation* op);

  R ConvertReductionOpToReductionPattern(const pir::Operation* op);

  PS ConvertOpToPS(const pir::Operation* op);
  using StmtPtr4OpT =
      std::function<std::optional<StmtPattern*>(const pir::Operation*)>;
  StmtPtr4OpT MakeStmtFinderFromOp(std::vector<StmtPattern>* stmts);


  template <typename IsChozenPatternT, typename ConstructPatternT>
  std::optional<ErrorGroupPattern> MultiFuse(
      const IsChozenPatternT& IsChozenPattern,
      const ConstructPatternT& ConstructPattern,
      std::vector<StmtPattern>* stmts) {
    const auto StmtFinder = MakeStmtFinderFromOp(stmts);
    const auto VisitInputStmt = [&](const StmtPattern* stmt,
                                    const StmtVisitor& DoEach) {
      VisitStmtOp(*stmt, [&](const auto* op) {
        op_topo_.VisitInputOp(op, [&](const pir::Operation* input) {
          if (const auto& input_stmt = StmtFinder(input)) {
            if (IsChozenPattern(*input_stmt.value())) {
              DoEach(input_stmt.value());
            }
          }
        });
      });
    };
    const auto VisitOutputStmt = [&](const StmtPattern* stmt,
                                     const StmtVisitor& DoEach) {
      VisitStmtOp(*stmt, [&](const auto* op) {
        op_topo_.VisitOutputOp(op, [&](const pir::Operation* output) {
          if (const auto& output_stmt = StmtFinder(output)) {
            if (IsChozenPattern(*output_stmt.value())) {
              DoEach(output_stmt.value());
            }
          }
        });
      });
    };
    const auto IsSinkPattern = [&](const StmtPattern* stmt) {
      if (!IsChozenPattern(*stmt)) return false;
      std::size_t num_injective_src_outputs = 0;
      VisitOutputStmt(stmt, [&](const auto& consumer) {
        num_injective_src_outputs += IsChozenPattern(*consumer);
      });
      return num_injective_src_outputs == 0;
    };
    const auto Cmp = [&](const auto* lhs, const auto* rhs) {
      return this->GetOrderValue4Op(lhs) < this->GetOrderValue4Op(rhs);
    };
    common::BfsWalker<const StmtPattern*> reverse_walker(VisitInputStmt);
    const auto& GetAllUpstreamOps = [&](const StmtPattern* stmt_ptr) {
      std::vector<const pir::Operation*> visited_ops;
      reverse_walker(stmt_ptr, [&](const StmtPattern* node) {
        VisitStmtOp(*node, [&](const auto* op) { visited_ops.push_back(op); });
      });
      std::sort(visited_ops.begin(), visited_ops.end(), Cmp);
      return visited_ops;
    };

    std::vector<StmtPattern> ret_stmts = [&] {
      std::vector<StmtPattern> ret_stmts;
      ret_stmts.reserve(stmts->size());
      for (const auto& stmt : *stmts) {
        if (!IsChozenPattern(stmt)) {
          ret_stmts.push_back(stmt);
        } else {
          // do nothing.
        }
      }
      return ret_stmts;
    }();
    for (auto& stmt : *stmts) {
      if (!IsSinkPattern(&stmt)) continue;
      ret_stmts.emplace_back(ConstructPattern(GetAllUpstreamOps(&stmt)));
    }
    *stmts = ret_stmts;
    return std::nullopt;
  }

  struct StmtIterPair {
    std::list<StmtPattern*>::iterator upstream_iter;
    std::list<StmtPattern*>::iterator downstream_iter;
  };

  bool IsConnected(const StmtPtr4OpT& StmtFinder,
                   const StmtPattern* upstream,
                   const StmtPattern* downstream);

  template <typename FuseTargetConditionT>
  std::optional<StmtIterPair> FindConnetedPattenPairWithCondition(
      const StmtPtr4OpT& StmtFinder,
      std::list<StmtPattern*>* stmt_ptrs,
      const FuseTargetConditionT& FuseTargetCondition) {
    for (auto dst_iter = stmt_ptrs->begin(); dst_iter != stmt_ptrs->end();
         ++dst_iter) {
      for (auto src_iter = stmt_ptrs->begin(); src_iter != stmt_ptrs->end();
           ++src_iter) {
        if (src_iter == dst_iter) continue;
        if (!IsConnected(StmtFinder, *src_iter, *dst_iter)) continue;
        if (FuseTargetCondition(**src_iter, **dst_iter)) {
          return StmtIterPair{
              .upstream_iter = src_iter,
              .downstream_iter = dst_iter,
          };
        }
      }
    }
    return std::nullopt;
  }


  template <typename FusionPolicy>
  std::optional<ErrorGroupPattern> FuseFilteredStmtPatterns(
      std::vector<StmtPattern>* stmt_patterns);

  ShardableAxesSignature GetShardableAxesSignature(const OpTopo& op_topo);

 private : 
  std::vector<const pir::Operation*> ops_;
  ShardableAxesInferer shardable_axes_inferer_;
  OpTopo op_topo_;
  std::function<bool(const pir::Operation*)> IsInThisOpList;
  std::function<bool(const pir::Operation*)> IsInjectiveSource;
  std::function<size_t(const pir::Operation*)> GetOrderValue4Op;
};

}  // namespace cinn::frontend::cluster_ops
