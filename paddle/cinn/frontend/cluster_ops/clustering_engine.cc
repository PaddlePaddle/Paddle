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

#include "paddle/cinn/frontend/cluster_ops/clustering_engine.h"
#include "paddle/cinn/frontend/cluster_ops/fusion_helper.h"

namespace cinn::frontend::cluster_ops {

ClusteringEngine::ClusteringEngine(
    const std::vector<const pir::Operation*>& ops,
    const ShardableAxesInferer& shardable_axes_inferer,
    const std::shared_ptr<ClusteringPolicy>& clustering_policy)
    : ops_(ops),
      op_topo_(OpTopo::Make(ops)),
      shardable_axes_inferer_(shardable_axes_inferer),
      clustering_policy_(clustering_policy) {}

ClusteringResult ClusteringEngine::ClusterOps() {
  const std::vector<StmtPattern> stmt_patterns = [&] {
    GroupPattern raw_parsed =
        StmtFusionHelper(ops_, shardable_axes_inferer_).FuseToGroupPattern();
    CHECK(!std::holds_alternative<ErrorGroupPattern>(raw_parsed))
        << std::get<ErrorGroupPattern>(raw_parsed).error_string;
    CHECK(std::holds_alternative<std::vector<StmtPattern>>(raw_parsed));
    return std::get<std::vector<StmtPattern>>(raw_parsed);
  }();
  VLOG(4) << "- After Raw Parsing, the number of StmtPatterns is "
          << stmt_patterns.size();
  VLOG(4) << "- Making Acyclic Same Cluster Bfs Walker";
  common::BfsWalker<const StmtPattern*> walker =
      MakeAcyclicSameClusterBfsWalker(stmt_patterns);
  auto OrderValue4Op = MakeTopoOrderFinderOfOp(ops_);

  std::vector<std::vector<const StmtPattern*>> stmts_list;
  VisitConnectedComponent(walker, stmt_patterns, [&](auto stmt_ptrs) {
    SortStmtPtrs(&stmt_ptrs, OrderValue4Op);
    stmts_list.push_back(stmt_ptrs);
  });

  VLOG(4) << "- Sort Stmts List";
  SortStmtsList(&stmts_list, OrderValue4Op);
  VLOG(4) << "- Make Clustering Result";
  return clustering_policy_->MakeClusteringResult(stmts_list);
}

void ClusteringEngine::SortStmtsList(
    std::vector<std::vector<const StmtPattern*>>* stmt_ptrs,
    const std::function<size_t(const pir::Operation*)>& OrderValue4Op) {
  auto GetOrderValue = [&](const std::vector<const StmtPattern*>& stmts) {
    CHECK(!stmts.empty());
    return OrderValue4Op(GetStmtSoleSinkOp(*stmts.back()));
  };
  auto Cmp = [&](const auto& lhs, const auto& rhs) {
    return GetOrderValue(lhs) < GetOrderValue(rhs);
  };
  std::sort(stmt_ptrs->begin(), stmt_ptrs->end(), Cmp);
}

common::BfsWalker<const StmtPattern*>
ClusteringEngine::MakeAcyclicSameClusterBfsWalker(
    const std::vector<StmtPattern>& stmt_patterns) {
  VLOG(4) << "-- Make Topo Walker";
  const auto entire_topo_walk = MakeTopoWalker(op_topo_, stmt_patterns);
  VLOG(4) << "-- Make ClusterRoot for Stmt";
  const auto ClusterRoot4Stmt =
      MakeClusterRoot4Stmt(entire_topo_walk, stmt_patterns);
  const auto IsInSameCluster = [=](const auto* lhs, const auto* rhs) {
    return ClusterRoot4Stmt(lhs) == ClusterRoot4Stmt(rhs);
  };
  VLOG(4) << "-- Make Is Acyclic Connected Predicator";
  const auto IsAcyclicConnected = MakePredicatorIsAcyclicConnected(
      entire_topo_walk, stmt_patterns, ClusterRoot4Stmt);
  using NodeVisitor = std::function<void(const StmtPattern*)>;
  const auto VisitAcyclicClusterNext = [=](const StmtPattern* stmt,
                                           const NodeVisitor& DoEach) {
    entire_topo_walk.VisitPrevNodes(stmt, [&](const StmtPattern* input) {
      VLOG(4) << "Walker || Checking Connected with PreNode:";
      VLOG(4) << "Walker || Base Node is:\n" << StmtPatternDebugStr(*stmt);
      VLOG(4) << "Walker || Pre Node is:\n" << StmtPatternDebugStr(*input);

      bool in_same_cluster = IsInSameCluster(stmt, input);
      VLOG(4) << "Walker || In Same Cluster: " << in_same_cluster;

      bool is_acyclic_connected = IsAcyclicConnected(stmt, input);
      VLOG(4) << "Walker || Is Acyclic Connected: " << is_acyclic_connected;

      if (!in_same_cluster || !is_acyclic_connected) return;
      DoEach(input);
    });
    entire_topo_walk.VisitNextNodes(stmt, [&](const StmtPattern* output) {
      VLOG(4) << "Walker || Checking Connected with NextNode:";
      VLOG(4) << "Walker || Base Node is:\n" << StmtPatternDebugStr(*stmt);
      VLOG(4) << "Walker || Next Node is:\n" << StmtPatternDebugStr(*output);

      bool in_same_cluster = IsInSameCluster(stmt, output);
      VLOG(4) << "Walker || In Same Cluster: " << in_same_cluster;

      bool is_acyclic_connected = IsAcyclicConnected(stmt, output);
      VLOG(4) << "Walker || Is Acyclic Connected: " << is_acyclic_connected;

      if (!in_same_cluster || !is_acyclic_connected) return;
      DoEach(output);
    });
  };
  return common::BfsWalker<const StmtPattern*>(VisitAcyclicClusterNext);
}

ShardableAxes4ValueT ClusteringEngine::MakeInferedShardableAxes4Value(
    const std::vector<const StmtPattern*>& stmt_ptrs) {
  const OpSetPtr ops = [&] {
    auto ops = std::make_shared<OpSet>();
    for (const auto* stmt_ptr : stmt_ptrs) {
      VisitStmtOp(*stmt_ptr, [&](const auto* op) { ops->insert(op); });
    }
    return ops;
  }();
  auto value2shardable_axes = shardable_axes_inferer_.InferShardableAxes(ops);
  return [map = std::move(value2shardable_axes)](
             pir::Value value) -> std::optional<const ShardableAxes*> {
    const auto& iter = map.find(value);
    if (iter == map.end()) return std::nullopt;
    return &iter->second;
  };
}

ClusteringEngine::IsAcyclicConnectedT
ClusteringEngine::MakePredicatorIsAcyclicConnected(
    const common::TopoWalker<const StmtPattern*>& walker,
    const std::vector<StmtPattern>& stmt_patterns,
    const ClusteringEngine::ClusterRoot4StmtT& ClusterRoot4Stmt) {
  VLOG(4) << "MakePredicatorIsAcyclicConnected";
  const auto AllTopClosureUpstreams4Stmt =
      MakeAllTopClosureUpstreams4Stmt(walker, stmt_patterns, ClusterRoot4Stmt);
  const auto IsSrcAcyclicConnectedToDst = [&](const auto* src,
                                              const auto* dst) {
    // return true if there exist no other clusters's node in
    // all_topo_closure_upstreams(dst) - all_topo_closure_upstreams(src)
    const auto* src_upstreams = AllTopClosureUpstreams4Stmt(src);
    const auto* dst_upstreams = AllTopClosureUpstreams4Stmt(dst);
    std::vector<const StmtPattern*> diff_stmts;
    std::set_difference(dst_upstreams->begin(),
                        dst_upstreams->end(),
                        src_upstreams->begin(),
                        src_upstreams->end(),
                        std::back_inserter(diff_stmts));
    const auto* cluster_root = ClusterRoot4Stmt(src);
    CHECK_EQ(cluster_root, ClusterRoot4Stmt(dst));
    for (const auto* diff_stmt : diff_stmts) {
      if (ClusterRoot4Stmt(diff_stmt) != cluster_root) return false;
    }
    return true;
  };
  using Src2AcyclicConnectedDst =
      std::map<const StmtPattern*, std::set<const StmtPattern*>>;
  Src2AcyclicConnectedDst src2acyclic_connected_dst;
  for (const auto& stmt : stmt_patterns) {
    const auto* src = &stmt;
    auto* acyclic_connected_dst = &src2acyclic_connected_dst[src];
    walker.VisitNextNodes(src, [&](const auto* dst) {
      if (!(acyclic_connected_dst->count(dst) == 0)) return;
      if (!(ClusterRoot4Stmt(src) == ClusterRoot4Stmt(dst))) return;
      if (IsSrcAcyclicConnectedToDst(src, dst)) {
        acyclic_connected_dst->insert(dst);
      }
    });
  }
  return [map = std::move(src2acyclic_connected_dst)](const StmtPattern* src,
                                                      const StmtPattern* dst) {
    const auto& iter = map.find(src);
    if (iter == map.end()) return false;
    return iter->second.count(dst) > 0;
  };
}

ClusteringEngine::AllTopClosureUpstreams4StmtT
ClusteringEngine::MakeAllTopClosureUpstreams4Stmt(
    const common::TopoWalker<const StmtPattern*>& entire_topo_walker,
    const std::vector<StmtPattern>& stmt_patterns,
    const ClusteringEngine::ClusterRoot4StmtT& ClusterRoot4Stmt) {
  const auto TopoClosure4RootStmt = MakeTopoClosure4RootStmt(
      entire_topo_walker, stmt_patterns, ClusterRoot4Stmt);
  using NodeVisitor = std::function<void(const StmtPattern*)>;
  std::unordered_map<const StmtPattern*, std::set<const StmtPattern*>>
      stmt2all_topo_closure_upstreams;
  for (const auto& stmt_pattern : stmt_patterns) {
    if (stmt2all_topo_closure_upstreams.count(&stmt_pattern)) continue;
    const auto* cluster_root = ClusterRoot4Stmt(&stmt_pattern);
    const auto& topo_closure = TopoClosure4RootStmt(cluster_root);
    CHECK(topo_closure.has_value());
    VisitStmtTopoClosureUpstreams(
        entire_topo_walker,
        *topo_closure.value(),
        [&](const auto* stmt, const auto& all_topo_closure_upstreams) {
          if (ClusterRoot4Stmt(stmt) != cluster_root) return;
          CHECK(stmt2all_topo_closure_upstreams
                    .emplace(stmt, all_topo_closure_upstreams)
                    .second);
        });
  }
  return [map = std::move(stmt2all_topo_closure_upstreams)](
             const StmtPattern* stmt) {
    const auto iter = map.find(stmt);
    if (iter == map.end()) {
      static const std::set<const StmtPattern*> empty;
      return &empty;
    }
    return &iter->second;
  };
}

ClusteringEngine::TopoClosure4RootStmtT
ClusteringEngine::MakeTopoClosure4RootStmt(
    const common::TopoWalker<const StmtPattern*>& entire_topo_walker,
    const std::vector<StmtPattern>& stmt_patterns,
    const ClusteringEngine::ClusterRoot4StmtT& ClusterRoot4Stmt) {
  using NodeVisitor = std::function<void(const StmtPattern*)>;
  auto VisitClusterInput = [&](const StmtPattern* stmt,
                               const NodeVisitor& DoEach) {
    entire_topo_walker.VisitPrevNodes(stmt, [&](const auto* input) {
      if (ClusterRoot4Stmt(stmt) == ClusterRoot4Stmt(input)) {
        DoEach(input);
      }
    });
  };
  auto IsClusterSource = [&](const auto* stmt) {
    size_t num_inputs = 0;
    VisitClusterInput(stmt, [&](const auto*) { ++num_inputs; });
    return num_inputs == 0;
  };
  auto VisitClusterOutput = [&](const StmtPattern* stmt,
                                const NodeVisitor& DoEach) {
    entire_topo_walker.VisitNextNodes(stmt, [&](const auto* output) {
      if (ClusterRoot4Stmt(stmt) == ClusterRoot4Stmt(output)) {
        DoEach(output);
      }
    });
  };
  auto IsClusterSink = [&](const auto* stmt) {
    size_t num_outputs = 0;
    VisitClusterOutput(stmt, [&](const auto*) { ++num_outputs; });
    return num_outputs == 0;
  };
  auto VisitClusterNext = [&](const StmtPattern* stmt,
                              const NodeVisitor& DoEach) {
    VisitClusterInput(stmt, DoEach);
    VisitClusterOutput(stmt, DoEach);
  };
  common::BfsWalker<const StmtPattern*> cluster_bfs_walker(VisitClusterNext);
  const auto IsReachable = MakeIsReachable(entire_topo_walker, stmt_patterns);
  std::unordered_map<const StmtPattern*, TopoClosure> root_stmt2topo_closure;
  for (const auto& stmt_pattern : stmt_patterns) {
    const auto* cluster_root = ClusterRoot4Stmt(&stmt_pattern);
    if (cluster_root != &stmt_pattern) continue;
    CHECK(!(root_stmt2topo_closure.count(cluster_root)));
    auto* topo_closure = &root_stmt2topo_closure[cluster_root];
    cluster_bfs_walker(cluster_root, [&](const auto* stmt) {
      if (IsClusterSource(stmt)) {
        topo_closure->sources.push_back(stmt);
      }
      if (IsClusterSink(stmt)) {
        topo_closure->sinks.push_back(stmt);
      }
    });
    topo_closure->stmts = CollectSubGraphAllStmts(entire_topo_walker,
                                                  IsReachable,
                                                  topo_closure->sources,
                                                  topo_closure->sinks);
  }
  return [map = std::move(root_stmt2topo_closure)](
             const StmtPattern* stmt) -> std::optional<const TopoClosure*> {
    const auto iter = map.find(stmt);
    if (iter == map.end()) return std::nullopt;
    return &iter->second;
  };
}

std::unordered_set<const StmtPattern*>
ClusteringEngine::CollectSubGraphAllStmts(
    const common::TopoWalker<const StmtPattern*>& entire_topo_walker,
    const ClusteringEngine::IsReachableT& IsReachable,
    const std::list<const StmtPattern*> sources,
    const std::list<const StmtPattern*> sinks) {
  auto IsConnectedToOneSource = [&](const auto* stmt) {
    for (const auto* source : sources) {
      if (IsReachable(source, stmt)) return true;
    }
    return false;
  };
  using NodeVisitor = std::function<void(const StmtPattern*)>;
  auto VisitInput = [&](const StmtPattern* stmt, const NodeVisitor& DoEach) {
    entire_topo_walker.VisitPrevNodes(stmt, [&](const auto* input) {
      if (IsConnectedToOneSource(input)) {
        DoEach(input);
      }
    });
  };
  auto IsConnectedToOneSink = [&](const auto* stmt) {
    for (const auto* sink : sinks) {
      if (IsReachable(stmt, sink)) return true;
    }
    return false;
  };
  auto VisitOutput = [&](const StmtPattern* stmt, const NodeVisitor& DoEach) {
    entire_topo_walker.VisitNextNodes(stmt, [&](const auto* output) {
      if (IsConnectedToOneSink(output)) {
        DoEach(output);
      }
    });
  };
  auto VisitNext = [&](const StmtPattern* stmt, const NodeVisitor& DoEach) {
    VisitInput(stmt, DoEach);
    VisitOutput(stmt, DoEach);
  };
  std::unordered_set<const StmtPattern*> ret;
  common::BfsWalker<const StmtPattern*> bfs_walker(VisitNext);
  bfs_walker(sources.begin(), sources.end(), [&](const auto* stmt) {
    ret.insert(stmt);
  });
  return ret;
}

ClusteringEngine::IsReachableT ClusteringEngine::MakeIsReachable(
    const common::TopoWalker<const StmtPattern*>& walker,
    const std::vector<StmtPattern>& stmt_patterns) {
  const auto& sources = [&] {
    std::list<const StmtPattern*> sources;
    const auto IsSource = [&](const auto* stmt) {
      size_t num_upstreams = 0;
      walker.VisitPrevNodes(stmt, [&](const auto*) { ++num_upstreams; });
      return num_upstreams == 0;
    };
    for (const auto& stmt : stmt_patterns) {
      if (IsSource(&stmt)) {
        sources.push_back(&stmt);
      }
    }
    return sources;
  }();

  std::unordered_map<const StmtPattern*, std::set<const StmtPattern*>>
      stmt2upstreams;
  walker(sources.begin(), sources.end(), [&](const auto* stmt) {
    (void)stmt2upstreams[stmt];
    walker.VisitPrevNodes(stmt, [&](const auto* upstream) {
      stmt2upstreams[stmt].insert(upstream);
    });
  });
  return [map = std::move(stmt2upstreams)](const StmtPattern* src,
                                           const StmtPattern* dst) {
    if (src == dst) return true;
    const auto iter = map.find(dst);
    if (iter == map.end()) return false;
    return iter->second.count(src) > 0;
  };
}

std::function<const StmtPattern*(const StmtPattern*)>
ClusteringEngine::MakeClusterRoot4Stmt(
    const common::TopoWalker<const StmtPattern*>& topo_walker,
    const std::vector<StmtPattern>& stmt_patterns) {
  VLOG(4) << "MakeClusterRoot4Stmt";
  std::unordered_map<const StmtPattern*, const StmtPattern*> stmt2cluster_root;
  VisitClusterStmts(topo_walker, stmt_patterns, [&](const auto& stmt_ptrs) {
    CHECK(!stmt_ptrs.empty());
    const auto* root = *stmt_ptrs.begin();
    for (const auto* stmt_ptr : stmt_ptrs) {
      CHECK(stmt2cluster_root.emplace(stmt_ptr, root).second);
    }
  });
  return [map = std::move(stmt2cluster_root)](const StmtPattern* stmt) {
    const auto& iter = map.find(stmt);
    CHECK(iter != map.end());
    return iter->second;
  };
}
}  // namespace cinn::frontend::cluster_ops
