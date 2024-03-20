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

#include "paddle/cinn/frontend/group_pattern_util.h"

#include <algorithm>
#include <optional>
#include <typeinfo>
#include <variant>

#include "paddle/cinn/common/bfs_walker.h"
#include "paddle/cinn/common/topo_walker.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

namespace cinn::frontend {

namespace cluster_ops {

struct OpTopo {
  OpSetPtr ops;

  static OpTopo Make(const std::vector<const pir::Operation*>& ops) {
    auto ops_set = std::make_shared<OpSet>(ops.begin(), ops.end());
    return OpTopo{
        .ops = ops_set,
    };
  }

  template <typename OpVisitorT>
  void VisitInputOp(const pir::Operation* op, const OpVisitorT& DoEach) const {
    if (this->ops->count(op) == 0) return;
    for (int i = 0; i < op->num_operands(); ++i) {
      const auto* input_op = op->operand_source(i).defining_op();
      if (this->ops->count(input_op) == 0) continue;
      DoEach(input_op);
    }
  }

  template <typename OpVisitorT>
  void VisitOutputOp(const pir::Operation* op, const OpVisitorT& DoEach) const {
    for (int i = 0; i < op->num_results(); ++i) {
      pir::Value output = op->result(i);
      for (auto consumer_it = output.use_begin();
           consumer_it != output.use_end();
           ++consumer_it) {
        const auto* consumer_op = consumer_it->owner();
        if (consumer_op->isa<pir::YieldOp>()) continue;
        if (this->ops->count(consumer_op) == 0) continue;
        DoEach(consumer_op);
      }
    }
  }
};

int GetOutputShardableAxesResultIdx(const pir::Operation* op) { return 0; }

OpPatternKind GetOpPatternKind(const ::pir::Operation* node) {
  return hlir::framework::pir::CompatibleInfo::OpKind(*node);
}

bool IsGeneralInjective(const pir::Operation* op) {
  hlir::framework::OpPatternKind op_pattern_kind = GetOpPatternKind(op);
  return op_pattern_kind == hlir::framework::kElementWise ||
         op_pattern_kind == hlir::framework::kBroadcast ||
         op_pattern_kind == hlir::framework::kInjective;
}

bool IsISPattern(const StmtPattern& pattern) {
  return std::holds_alternative<IS>(pattern);
}

bool IsPSPattern(const StmtPattern& pattern) {
  return std::holds_alternative<PS>(pattern);
}

bool IsRPattern(const StmtPattern& pattern) {
  return std::holds_alternative<R>(pattern);
}

std::list<const pir::Operation*> GetSinks(const OpSet& ops) {
  const auto IsSink = [&](const pir::Operation* op) {
    for (int i = 0; i < op->num_results(); ++i) {
      pir::Value output = op->result(i);
      for (auto consumer_it = output.use_begin();
           consumer_it != output.use_end();
           ++consumer_it) {
        const auto* consumer_op = consumer_it->owner();
        if (consumer_op->isa<pir::YieldOp>()) continue;
        if (ops.count(consumer_op) > 0) return false;
      }
    }
    return true;
  };
  std::list<const pir::Operation*> sinks;
  for (const auto* op : ops) {
    if (IsSink(op)) {
      sinks.push_back(op);
    }
  }
  return sinks;
}

const pir::Operation* GetSoleSink(const OpSet& ops) {
  const auto& sinks = GetSinks(ops);
  CHECK_EQ(sinks.size(), 1);
  return *sinks.begin();
}

template <typename DoEachT>
void VisitStmtOpImpl(const IS& injective_source, const DoEachT& DoEach) {
  for (const auto* op : injective_source.ops) {
    DoEach(op);
  }
}

template <typename DoEachT>
void VisitStmtOpImpl(const PS& partial_shardable, const DoEachT& DoEach) {
  for (const auto* op : partial_shardable.ops) {
    DoEach(op);
  }
}

template <typename DoEachT>
void VisitStmtOpImpl(const R& reduce, const DoEachT& DoEach) {
  std::visit(adt::match{
                 [](const std::monostate&) {
                   // do nothing.
                 },
                 [&](const IS& injective_source) {
                   VisitStmtOpImpl(injective_source, DoEach);
                 },
                 [&](const PS& partial_shardable) {
                   VisitStmtOpImpl(partial_shardable, DoEach);
                 },
             },
             reduce.input);
  DoEach(reduce.reduce_op_pattern.reduce_op);
}

template <typename DoEachT>
void VisitStmtOp(const StmtPattern& stmt, const DoEachT& DoEach) {
  std::visit([&](const auto& impl) { VisitStmtOpImpl(impl, DoEach); }, stmt);
}

common::TopoWalker<const pir::Operation*> GetOpsReversedTopoWalker(
    const OpTopo& op_topo) {
  const auto VisitUpStreamInOps = [op_topo](const pir::Operation* op,
                                            const OpVisitor& DoEach) {
    op_topo.VisitInputOp(op, DoEach);
  };
  const auto VisitDownStreamInOps = [op_topo](const pir::Operation* op,
                                              const OpVisitor& DoEach) {
    op_topo.VisitOutputOp(op, DoEach);
  };
  common::TopoWalker<const pir::Operation*> reversed_walker(
      VisitDownStreamInOps, VisitUpStreamInOps);
  return reversed_walker;
}

size_t GetRank(pir::Value value) {
  return value.type().dyn_cast<pir::DenseTensorType>().dims().size();
}

std::vector<int64_t> GetReduceAxes(const pir::Operation* reduce_op) {
  const size_t input_rank = GetRank(reduce_op->operand_source(0));
  const auto& attr_val = reduce_op->attributes().at("dim");
  CHECK(attr_val.isa<::pir::ArrayAttribute>());
  const auto& axis_attr = attr_val.dyn_cast<::pir::ArrayAttribute>();
  std::vector<int64_t> reduce_axes;
  for (int i = 0; i < axis_attr.size(); ++i) {
    int64_t axis = axis_attr.at(i).dyn_cast<::pir::Int64Attribute>().data();
    if (axis < 0) {
      axis += input_rank;
    }
    CHECK_GE(axis, 0);
    CHECK_LT(axis, input_rank);
    reduce_axes.push_back(axis);
  }
  return reduce_axes;
}

bool GetReduceOpKeepDims(const pir::Operation* reduce_op) {
  const auto& attr_val = reduce_op->attributes().at("keep_dim");
  CHECK(attr_val.isa<::pir::BoolAttribute>());
  return attr_val.dyn_cast<::pir::BoolAttribute>();
}

std::function<size_t(const pir::Operation*)> MakeTopoOrderFinderOfOp(
    const std::vector<const pir::Operation*>& ops) {
  std::unordered_map<const pir::Operation*, size_t> op2order_in_block;
  size_t order = 0;
  for (const pir::Operation* op : ops) {
    op2order_in_block[op] = ++order;
  }
  return [map = std::move(op2order_in_block)](const pir::Operation* op) {
    const auto& iter = map.find(op);
    CHECK(iter != map.end());
    return iter->second;
  };
}

pir::Value GetStmtBigestShapeValueImpl(const IS& injective_source) {
  const auto* sink_op = injective_source.sole_sink;
  const int result_idx = GetOutputShardableAxesResultIdx(sink_op);
  return sink_op->result(result_idx);
}

pir::Value GetStmtBigestShapeValueImpl(const R& reduce_pattern) {
  const auto* sink_op = reduce_pattern.reduce_op_pattern.reduce_op;
  CHECK_EQ(sink_op->num_operands(), 1);
  return sink_op->operand_source(0);
}

pir::Value GetStmtBigestShapeValueImpl(const PS& partial_shardable) {
  const auto* sink_op = partial_shardable.sole_sink;
  const int result_idx = GetOutputShardableAxesResultIdx(sink_op);
  return sink_op->result(result_idx);
}

pir::Value GetStmtBigestShapeValue(const StmtPattern& stmt) {
  return std::visit(
      [&](const auto& impl) { return GetStmtBigestShapeValueImpl(impl); },
      stmt);
}

const pir::Operation* GetStmtSoleSinkImpl(const IS& injective_source) {
  return injective_source.sole_sink;
}

const pir::Operation* GetStmtSoleSinkImpl(const PS& partial_shardable) {
  return partial_shardable.sole_sink;
}

const pir::Operation* GetStmtSoleSinkImpl(const R& reduce) {
  return reduce.reduce_op_pattern.reduce_op;
}

const pir::Operation* GetStmtSoleSinkOp(const StmtPattern& stmt) {
  return std::visit([](const auto& impl) { return GetStmtSoleSinkImpl(impl); },
                    stmt);
}

void SortStmtPtrs(
    std::vector<const StmtPattern*>* stmt_ptrs,
    const std::function<size_t(const pir::Operation*)>& OrderValue4Op) {
  auto GetOrderValue4Stmt = [&](const StmtPattern* stmt) {
    const auto* sink_op = GetStmtSoleSinkOp(*stmt);
    return OrderValue4Op(sink_op);
  };
  const auto Cmp = [&](const auto* lhs, const auto* rhs) {
    const auto& lhs_order = GetOrderValue4Stmt(lhs);
    const auto& rhs_order = GetOrderValue4Stmt(rhs);
    return lhs_order < rhs_order;
  };
  std::sort(stmt_ptrs->begin(), stmt_ptrs->end(), Cmp);
}

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

  using ShardableAxes4ValueT =
      std::function<std::optional<const ShardableAxes*>(pir::Value)>;
  ShardableAxes4ValueT MakeInferedShardableAxes4Value(
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

  common::TopoWalker<const StmtPattern*> MakeTopoWalker(
      const OpTopo& op_topo, const std::vector<StmtPattern>& stmt_patterns) {
    using StmtPtrs = std::vector<const StmtPattern*>;
    using Op2OwnerStmtPtrs =
        std::unordered_map<const pir::Operation*, StmtPtrs>;
    auto op2owner_stmt_ptr = std::make_shared<Op2OwnerStmtPtrs>();
    for (const auto& stmt : stmt_patterns) {
      VisitStmtOp(stmt, [&](const pir::Operation* op) {
        (*op2owner_stmt_ptr)[op].push_back(&stmt);
      });
    }
    using NodeVisitor = std::function<void(const StmtPattern*)>;
    auto VisitInput = [=](const StmtPattern* stmt, const NodeVisitor& DoEach) {
      VisitStmtOp(*stmt, [&](const auto* op) {
        op_topo.VisitInputOp(op, [&](const auto* input_op) {
          const auto& owners_iter = op2owner_stmt_ptr->find(input_op);
          if (owners_iter == op2owner_stmt_ptr->end()) return;
          if (owners_iter->second.size() != 1) return;
          const auto* owner_stmt = *owners_iter->second.begin();
          if (owner_stmt == stmt) return;
          DoEach(owner_stmt);
        });
      });
    };
    auto VisitOutput = [=](const StmtPattern* stmt, const NodeVisitor& DoEach) {
      const auto* sink = GetStmtSoleSinkOp(*stmt);
      op_topo.VisitOutputOp(sink, [&](const pir::Operation* op) {
        const auto& owners_iter = op2owner_stmt_ptr->find(op);
        if (owners_iter == op2owner_stmt_ptr->end()) return;
        for (const StmtPattern* stmt : owners_iter->second) {
          DoEach(stmt);
        }
      });
    };
    const auto& TryPushBack = [](const auto* stmt, auto* stmts) {
      if (std::find(stmts->begin(), stmts->end(), stmt) == stmts->end()) {
        stmts->push_back(stmt);
      }
    };
    using EdgeCache =
        std::unordered_map<const StmtPattern*, std::vector<const StmtPattern*>>;
    auto stmt2inputs = std::make_shared<EdgeCache>();
    auto stmt2outputs = std::make_shared<EdgeCache>();
    for (const auto& stmt : stmt_patterns) {
      (void)(*stmt2inputs)[&stmt];
      VisitInput(&stmt, [&](const auto* input) {
        TryPushBack(input, &(*stmt2inputs)[&stmt]);
      });
      (void)(*stmt2outputs)[&stmt];
      VisitOutput(&stmt, [&](const auto* output) {
        TryPushBack(output, &(*stmt2outputs)[&stmt]);
      });
    }

    auto VisitCachedInput = [stmt2inputs](const auto* stmt,
                                          const NodeVisitor& DoEach) {
      const auto& map = (*stmt2inputs);
      const auto& iter = map.find(stmt);
      if (iter == map.end()) return;
      for (const auto* input : iter->second) {
        DoEach(input);
      }
    };
    auto VisitCachedOutput = [stmt2outputs](const auto* stmt,
                                            const NodeVisitor& DoEach) {
      const auto& map = (*stmt2outputs);
      const auto& iter = map.find(stmt);
      if (iter == map.end()) return;
      for (const auto* output : iter->second) {
        DoEach(output);
      }
    };
    return common::TopoWalker<const StmtPattern*>(VisitCachedInput,
                                                  VisitCachedOutput);
  }

  const std::vector<const pir::Operation*> ops_;
  const std::shared_ptr<ClusteringPolicy> clustering_policy_;
  ShardableAxesInferer shardable_axes_inferer_;
  const OpTopo op_topo_;
};

}  // namespace cluster_ops

ClusteringResult ClusterOps(
    const std::vector<const pir::Operation*>& ops,
    const std::shared_ptr<ShardableAxesProvider>& shardable_axes_provider,
    const std::shared_ptr<ClusteringPolicy>& clustering_policy) {
  VLOG(4) << "Initializing Inferer";
  ShardableAxesInferer inferer(shardable_axes_provider);
  VLOG(4) << "Initializing Clustering Engine";
  ClusteringEngine engine(ops, inferer, clustering_policy);
  VLOG(4) << "Engine calls ClusterOps()";
  return engine.ClusterOps();
}
}  // namespace cinn::frontend
