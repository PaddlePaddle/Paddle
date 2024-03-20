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
  std::vector<StmtPattern> ConvertToStmtsPattern();
  void SortStmtPatterns(std::vector<StmtPattern>* stmt_patterns);

  std::optional<ErrorGroupPattern> Fuse_IS_x_IS_2_IS(
      std::vector<StmtPattern>* stmt_patterns);

  std::optional<ErrorGroupPattern> Fuse_PS_x_PS_2_PS(
      std::vector<StmtPattern>* stmt_patterns);

  struct FusePolicy_IS_x_PS_2_PS {
    static bool FuseCondition(const StmtPattern& upstream,
                              const StmtPattern& downstream);
    static std::variant<StmtPattern, ErrorGroupPattern> MergePattern(
        const StmtPattern& upstream, const StmtPattern& downstream);
    static std::variant<StmtPattern, ErrorGroupPattern> MergePatternImpl(
        const IS& upstream, const PS& downstream);
    static ShardableAxesSignature MergeShardableAxesSignature(
        const IS& upstream, const PS& downstream);
  };

  std::optional<ErrorGroupPattern> Fuse_IS_x_PS_2_PS(
      std::vector<StmtPattern>* stmt_patterns);
  struct FusePolicy_IS_x_R_2_R {
    static bool FuseCondition(const StmtPattern& upstream,
                              const StmtPattern& downstream);
    static std::variant<StmtPattern, ErrorGroupPattern> MergePattern(
        const StmtPattern& upstream, const StmtPattern& downstream);
    static std::variant<StmtPattern, ErrorGroupPattern> MergePatternImpl(
        const IS& upstream, const R& downstream);
  };

  std::optional<ErrorGroupPattern> Fuse_IS_x_R_2_R(
      std::vector<StmtPattern>* stmt_patterns);

  struct FusePolicy_PS_x_R_2_R {
    static bool FuseCondition(const StmtPattern& upstream,
                              const StmtPattern& downstream);
    static std::variant<StmtPattern, ErrorGroupPattern> MergePattern(
        const StmtPattern& upstream, const StmtPattern& downstream);
    static std::variant<StmtPattern, ErrorGroupPattern> MergePatternImpl(
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
  static StmtPtr4OpT MakeStmtFinderFromOp(std::vector<StmtPattern>* stmts);

  template <typename IsChozenPatternT, typename ConstructPatternT>
  std::optional<ErrorGroupPattern> MultiFuse(
      const IsChozenPatternT& IsChozenPattern,
      const ConstructPatternT& ConstructPattern,
      std::vector<StmtPattern>* stmts);

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
      const FuseTargetConditionT& FuseTargetCondition);

  template <typename FusionPolicy>
  std::optional<ErrorGroupPattern> FuseFilteredStmtPatterns(
      std::vector<StmtPattern>* stmt_patterns)

  ShardableAxesSignature GetShardableAxesSignature(const OpTopo& op_topo)

 private : 
  std::vector<const pir::Operation*> ops_;
  ShardableAxesInferer shardable_axes_inferer_;
  OpTopo op_topo_;
  std::function<bool(const pir::Operation*)> IsInThisOpList;
  std::function<bool(const pir::Operation*)> IsInjectiveSource;
  std::function<size_t(const pir::Operation*)> GetOrderValue4Op;
};

}  // namespace cinn::frontend::cluster_ops
