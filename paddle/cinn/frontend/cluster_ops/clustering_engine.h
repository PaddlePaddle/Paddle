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
#include "paddle/cinn/frontend/cluster_ops/shardable_axes_inferer.h"
#include "paddle/cinn/frontend/cluster_ops/shardable_axes_provider.h"

namespace cinn::frontend::cluster_ops {

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
      const DoEachComponentT& DoEachComponent) {
    VLOG(4) << "Step 2, Searching Connected Componenet";
    std::unordered_set<const StmtPattern*> visited;
    for (const auto& start : stmt_patterns) {
      VLOG(2) << "Choose BFS start StmtPattern: \n"
              << StmtPatternDebugStr(start);
      if (visited.count(&start)) continue;
      std::vector<const StmtPattern*> component;
      walker(&start, [&](const auto* stmt) {
        component.push_back(stmt);
        CHECK(visited.emplace(stmt).second);
      });
      DoEachComponent(component);
    }
    VLOG(4) << "Step 2 Finished";
  }

  ShardableAxes4ValueT MakeInferedShardableAxes4Value(
      const std::vector<const StmtPattern*>& stmt_ptrs);

  common::BfsWalker<const StmtPattern*> MakeAcyclicSameClusterBfsWalker(
      const std::vector<StmtPattern>& stmt_patterns);

  using ClusterRoot4StmtT =
      std::function<const StmtPattern*(const StmtPattern*)>;

  using IsAcyclicConnectedT =
      std::function<bool(const StmtPattern* src, const StmtPattern* dst)>;

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
          DoEachStmtAndTopoClosureUpstreams) {
    const auto IsInTopoClosure = [&](const auto* stmt) {
      return topo_closure.stmts.count(stmt) > 0;
    };
    using NodeVisitor = std::function<void(const StmtPattern*)>;
    auto VisitInput = [&](const auto* stmt, const NodeVisitor& Visit) {
      entire_topo_walker.VisitPrevNodes(stmt, [&](const auto* input) {
        if (IsInTopoClosure(input)) {
          Visit(input);
        }
      });
    };
    auto VisitOutput = [&](const auto* stmt, const NodeVisitor& Visit) {
      entire_topo_walker.VisitNextNodes(stmt, [&](const auto* output) {
        if (IsInTopoClosure(output)) {
          Visit(output);
        }
      });
    };
    common::TopoWalker<const StmtPattern*> closure_walker(VisitInput,
                                                          VisitOutput);
    const auto& sources = topo_closure.sources;
    std::unordered_map<const StmtPattern*, std::set<const StmtPattern*>>
        stmt2all_topo_closure_upstreams;
    closure_walker(sources.begin(), sources.end(), [&](const auto* stmt) {
      auto* stmt_upstreams = &stmt2all_topo_closure_upstreams[stmt];
      VisitInput(stmt, [&](const auto* input) {
        stmt_upstreams->insert(input);
        const auto& input_upstreams = stmt2all_topo_closure_upstreams[input];
        stmt_upstreams->insert(input_upstreams.begin(), input_upstreams.end());
      });
      const auto* const_stmt_upstreams = stmt_upstreams;
      DoEachStmtAndTopoClosureUpstreams(stmt, *const_stmt_upstreams);
    });
  }

  IsReachableT MakeIsReachable(
      const common::TopoWalker<const StmtPattern*>& walker,
      const std::vector<StmtPattern>& stmt_patterns);

  std::function<const StmtPattern*(const StmtPattern*)> MakeClusterRoot4Stmt(
      const common::TopoWalker<const StmtPattern*>& topo_walker,
      const std::vector<StmtPattern>& stmt_patterns);

  template <typename DoEachComponentT>
  void VisitClusterStmts(const common::TopoWalker<const StmtPattern*>& walker,
                         const std::vector<StmtPattern>& stmt_patterns,
                         const DoEachComponentT& DoEachComponent) {
    std::vector<const StmtPattern*> stmt_ptrs = [&] {
      std::vector<const StmtPattern*> stmt_ptrs;
      stmt_ptrs.reserve(stmt_patterns.size());
      for (const auto& stmt : stmt_patterns) {
        stmt_ptrs.push_back(&stmt);
      }
      return stmt_ptrs;
    }();
    std::unordered_set<const StmtPattern*> visited;
    while (!stmt_ptrs.empty()) {
      VisitInferedClusterStmts(walker, stmt_ptrs, [&](const auto& component) {
        for (const auto* stmt_ptr : component) {
          CHECK(visited.emplace(stmt_ptr).second);
        }
        DoEachComponent(component);
      });
      stmt_ptrs = [&] {
        std::vector<const StmtPattern*> remainders;
        remainders.reserve(stmt_ptrs.size());
        for (const auto* stmt_ptr : stmt_ptrs) {
          if (visited.count(stmt_ptr)) continue;
          remainders.push_back(stmt_ptr);
        }
        return remainders;
      }();
    }
  }

  template <typename DoEachComponentT>
  void VisitInferedClusterStmts(
      const common::TopoWalker<const StmtPattern*>& entire_topo_walker,
      const std::vector<const StmtPattern*>& stmt_ptrs,
      const DoEachComponentT& DoEachComponent) {
    const auto ShardableAxes4Value = MakeInferedShardableAxes4Value(stmt_ptrs);
    const auto Fusible = [&](const auto* src, const auto* dst) {
      return clustering_policy_->IsEdgeFusible(ShardableAxes4Value, *src, *dst);
    };
    using NodeVisitor = std::function<void(const StmtPattern*)>;
    const auto VisitNext = [&](const StmtPattern* stmt,
                               const NodeVisitor& DoEach) {
      entire_topo_walker.VisitPrevNodes(stmt, [&](const auto* prev) {
        if (Fusible(prev, stmt)) {
          DoEach(prev);
        }
      });
      entire_topo_walker.VisitNextNodes(stmt, [&](const auto* next) {
        if (Fusible(stmt, next)) {
          DoEach(next);
        }
      });
    };
    common::BfsWalker<const StmtPattern*> cluster_walker(VisitNext);
    std::unordered_set<const StmtPattern*> visited;
    for (const auto* start : stmt_ptrs) {
      if (visited.count(start)) continue;
      if (!clustering_policy_->CanActAsSink(ShardableAxes4Value, *start))
        continue;
      std::vector<const StmtPattern*> collected_component;
      cluster_walker(start, [&](const auto* stmt_ptr) {
        collected_component.push_back(stmt_ptr);
        CHECK(visited.emplace(stmt_ptr).second);
      });
      DoEachComponent(collected_component);
    }
    CHECK(!visited.empty())
        << "no StmtPattern visited. please check if "
           "clustering_policy_->CanActAsSink() returns false all the time.";
  }

  const std::vector<const pir::Operation*> ops_;
  const std::shared_ptr<ClusteringPolicy> clustering_policy_;
  ShardableAxesInferer shardable_axes_inferer_;
  const OpTopo op_topo_;
};

}  // namespace cinn::frontend::cluster_ops
