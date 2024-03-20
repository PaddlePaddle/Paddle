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

#include "paddle/cinn/frontend/cluster_ops/cluster_engine.h"

namespace cinn::frontend::cluster_ops {
class ClusteringEngine {
 public:
  ClusteringEngine(const std::vector<const pir::Operation*>& ops,
                   const ShardableAxesInferer& shardable_axes_inferer,
                   const std::shared_ptr<ClusteringPolicy>& clustering_policy)
      : ops_(ops),
        op_topo_(OpTopo::Make(ops)),
        shardable_axes_inferer_(shardable_axes_inferer),
        clustering_policy_(clustering_policy) {}

  ClusteringResult ClusterOps() {
    VLOG(4) << "- Raw Parsing";
    const std::vector<StmtPattern> stmt_patterns = [&] {
      GroupPattern raw_parsed =
          StmtFusionHelper(ops_, shardable_axes_inferer_).FuseToGroupPattern();
      CHECK(!std::holds_alternative<ErrorGroupPattern>(raw_parsed))
          << std::get<ErrorGroupPattern>(raw_parsed).error_string;
      CHECK(std::holds_alternative<std::vector<StmtPattern>>(raw_parsed));
      return std::get<std::vector<StmtPattern>>(raw_parsed);
    }();

    common::BfsWalker<const StmtPattern*> walker =
        MakeAcyclicSameClusterBfsWalker(stmt_patterns);

    VLOG(4) << "- Making Acyclic Same Cluster Bfs Walker";
    std::vector<std::vector<const StmtPattern*>> stmts_list;
    VLOG(4) << "- Visit Connect Component";

    auto OrderValue4Op = MakeTopoOrderFinderOfOp(ops_);
    VisitConnectedComponent(walker, stmt_patterns, [&](auto stmt_ptrs) {
      SortStmtPtrs(&stmt_ptrs, OrderValue4Op);
      stmts_list.push_back(stmt_ptrs);
    });

    VLOG(4) << "- Sort Stmts List";
    SortStmtsList(&stmts_list, OrderValue4Op);
    VLOG(4) << "- Make Clustering Result";
    return clustering_policy_->MakeClusteringResult(stmts_list);
  }

 private:
  void SortStmtsList(
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

  template <typename DoEachComponentT>
  void VisitConnectedComponent(
      const common::BfsWalker<const StmtPattern*>& walker,
      const std::vector<StmtPattern>& stmt_patterns,
      const DoEachComponentT& DoEachComponent) {
    std::unordered_set<const StmtPattern*> visited;
    for (const auto& start : stmt_patterns) {
      if (visited.count(&start)) continue;
      std::vector<const StmtPattern*> component;
      walker(&start, [&](const auto* stmt) {
        component.push_back(stmt);
        CHECK(visited.emplace(stmt).second);
      });
      DoEachComponent(component);
    }
  }

  common::BfsWalker<const StmtPattern*> MakeAcyclicSameClusterBfsWalker(
      const std::vector<StmtPattern>& stmt_patterns) {
    const auto entire_topo_walk = MakeTopoWalker(op_topo_, stmt_patterns);
    const auto ClusterRoot4Stmt =
        MakeClusterRoot4Stmt(entire_topo_walk, stmt_patterns);
    const auto IsInSameCluster = [=](const auto* lhs, const auto* rhs) {
      return ClusterRoot4Stmt(lhs) == ClusterRoot4Stmt(rhs);
    };
    const auto IsAcyclicConnected = MakePredicatorIsAcyclicConnected(
        entire_topo_walk, stmt_patterns, ClusterRoot4Stmt);
    using NodeVisitor = std::function<void(const StmtPattern*)>;
    const auto VisitAcyclicClusterNext = [=](const StmtPattern* stmt,
                                             const NodeVisitor& DoEach) {
      entire_topo_walk.VisitPrevNodes(stmt, [&](const StmtPattern* input) {
        if (!IsInSameCluster(input, stmt)) return;
        if (!IsAcyclicConnected(input, stmt)) return;
        DoEach(input);
      });
      entire_topo_walk.VisitNextNodes(stmt, [&](const StmtPattern* output) {
        if (!IsInSameCluster(stmt, output)) return;
        if (!IsAcyclicConnected(stmt, output)) return;
        DoEach(output);
      });
    };
    return common::BfsWalker<const StmtPattern*>(VisitAcyclicClusterNext);
  }

  using IsAcyclicConnectedT =
      std::function<bool(const StmtPattern* src, const StmtPattern* dst)>;
  using ClusterRoot4StmtT =
      std::function<const StmtPattern*(const StmtPattern*)>;

  IsAcyclicConnectedT MakePredicatorIsAcyclicConnected(
      const common::TopoWalker<const StmtPattern*>& walker,
      const std::vector<StmtPattern>& stmt_patterns,
      const ClusterRoot4StmtT& ClusterRoot4Stmt) {
    const auto AllTopClosureUpstreams4Stmt = MakeAllTopClosureUpstreams4Stmt(
        walker, stmt_patterns, ClusterRoot4Stmt);
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
    return [map = std::move(src2acyclic_connected_dst)](
               const StmtPattern* src, const StmtPattern* dst) {
      const auto& iter = map.find(src);
      if (iter == map.end()) return false;
      return iter->second.count(dst) > 0;
    };
  }

  struct TopoClosure {
    std::list<const StmtPattern*> sources;
    std::list<const StmtPattern*> sinks;
    std::unordered_set<const StmtPattern*> stmts;
  };

  using TopoClosure4RootStmtT =
      std::function<std::optional<const TopoClosure*>(const StmtPattern*)>;

  using AllTopClosureUpstreams4StmtT =
      std::function<const std::set<const StmtPattern*>*(const StmtPattern*)>;

  AllTopClosureUpstreams4StmtT MakeAllTopClosureUpstreams4Stmt(
      const common::TopoWalker<const StmtPattern*>& entire_topo_walker,
      const std::vector<StmtPattern>& stmt_patterns,
      const ClusterRoot4StmtT& ClusterRoot4Stmt) {
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

  TopoClosure4RootStmtT MakeTopoClosure4RootStmt(
      const common::TopoWalker<const StmtPattern*>& entire_topo_walker,
      const std::vector<StmtPattern>& stmt_patterns,
      const ClusterRoot4StmtT& ClusterRoot4Stmt) {
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

  using IsReachableT =
      std::function<bool(const StmtPattern* src, const StmtPattern* dst)>;

  std::unordered_set<const StmtPattern*> CollectSubGraphAllStmts(
      const common::TopoWalker<const StmtPattern*>& entire_topo_walker,
      const IsReachableT& IsReachable,
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

  std::function<const StmtPattern*(const StmtPattern*)> MakeClusterRoot4Stmt(
      const common::TopoWalker<const StmtPattern*>& topo_walker,
      const std::vector<StmtPattern>& stmt_patterns) {
    std::unordered_map<const StmtPattern*, const StmtPattern*>
        stmt2cluster_root;
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

} // namespace cinn::frontend::cluster_ops