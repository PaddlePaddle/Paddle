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

#include "paddle/cinn/frontend/cluster_ops/cluster_policy.h"
#include "paddle/cinn/frontend/cluster_ops/common_utils.h"
#include "paddle/cinn/frontend/cluster_ops/group_pattern.h"
#include "paddle/cinn/frontend/cluster_ops/pattern_utils.h"
#include "paddle/cinn/frontend/cluster_ops/shardable_axes_provider.h"

namespace cinn::frontend::cluster_ops {

struct LoopAlignableStmtsPattern {
  std::vector<api::StmtPattern<FrontendPattern>> stmts;
};

struct ClusteringResult {
  std::vector<LoopAlignableStmtsPattern> loop_alignable_list;
};

class ClusteringEngine {
 public:
  ClusteringEngine(const std::vector<const pir::Operation*>& ops,
                   const ShardableAxesInferer& shardable_axes_inferer,
                   const std::shared_ptr<ClusteringPolicy>& clustering_policy);

  ClusteringResult ClusterOps();

 private:
  void SortStmtsList(
      std::vector<std::vector<const StmtPattern*>>* stmt_ptrs,
      const std::function<size_t(const pir::Operation*)>& OrderValue4Op);

  template <typename DoEachComponentT>
  void VisitConnectedComponent(
      const common::BfsWalker<const StmtPattern*>& walker,
      const std::vector<StmtPattern>& stmt_patterns,
      const DoEachComponentT& DoEachComponent);

  common::BfsWalker<const StmtPattern*> MakeAcyclicSameClusterBfsWalker(
      const std::vector<StmtPattern>& stmt_patterns);

  using IsAcyclicConnectedT =
      std::function<bool(const StmtPattern* src, const StmtPattern* dst)>;
  using ClusterRoot4StmtT =
      std::function<const StmtPattern*(const StmtPattern*)>;

  IsAcyclicConnectedT MakePredicatorIsAcyclicConnected(
      const common::TopoWalker<const StmtPattern*>& walker,
      const std::vector<StmtPattern>& stmt_patterns,
      const ClusterRoot4StmtT& ClusterRoot4Stmt);

  struct TopoClosure {
    std::list<const StmtPattern*> sources;
    std::list<const StmtPattern*> sinks;
    std::unordered_set<const StmtPattern*> stmts;
  };

  using IsReachableT =
      std::function<bool(const StmtPattern* src, const StmtPattern* dst)>;

  using TopoClosure4RootStmtT =
      std::function<std::optional<const TopoClosure*>(const StmtPattern*)>;

  using AllTopClosureUpstreams4StmtT =
      std::function<const std::set<const StmtPattern*>*(const StmtPattern*)>;

  AllTopClosureUpstreams4StmtT MakeAllTopClosureUpstreams4Stmt(
      const common::TopoWalker<const StmtPattern*>& entire_topo_walker,
      const std::vector<StmtPattern>& stmt_patterns,
      const ClusterRoot4StmtT& ClusterRoot4Stmt);

  TopoClosure4RootStmtT MakeTopoClosure4RootStmt(
      const common::TopoWalker<const StmtPattern*>& entire_topo_walker,
      const std::vector<StmtPattern>& stmt_patterns,
      const ClusterRoot4StmtT& ClusterRoot4Stmt);

  std::unordered_set<const StmtPattern*> CollectSubGraphAllStmts(
      const common::TopoWalker<const StmtPattern*>& entire_topo_walker,
      const IsReachableT& IsReachable,
      const std::list<const StmtPattern*> sources,
      const std::list<const StmtPattern*> sinks);

  template <typename DoEachStmtAndTopoClosureUpstreamsT>
  void VisitStmtTopoClosureUpstreams(
      const common::TopoWalker<const StmtPattern*>& entire_topo_walker,
      const TopoClosure& topo_closure,
      const DoEachStmtAndTopoClosureUpstreamsT&
          DoEachStmtAndTopoClosureUpstreams);

  IsReachableT MakeIsReachable(
      const common::TopoWalker<const StmtPattern*>& walker,
      const std::vector<StmtPattern>& stmt_patterns);

  std::function<const StmtPattern*(const StmtPattern*)> MakeClusterRoot4Stmt(
      const common::TopoWalker<const StmtPattern*>& topo_walker,
      const std::vector<StmtPattern>& stmt_patterns);

  template <typename DoEachComponentT>
  void VisitClusterStmts(const common::TopoWalker<const StmtPattern*>& walker,
                         const std::vector<StmtPattern>& stmt_patterns,
                         const DoEachComponentT& DoEachComponent);

  template <typename DoEachComponentT>
  void VisitInferedClusterStmts(
      const common::TopoWalker<const StmtPattern*>& entire_topo_walker,
      const std::vector<const StmtPattern*>& stmt_ptrs,
      const DoEachComponentT& DoEachComponent);

  const std::vector<const pir::Operation*> ops_;
  const std::shared_ptr<ClusteringPolicy> clustering_policy_;
  ShardableAxesInferer shardable_axes_inferer_;
  const OpTopo op_topo_;
};

}  // namespace cinn::frontend::cluster_ops
