// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/adt/generate_map_expr.h"
#include "paddle/cinn/adt/anchor_sd_equation_context.h"
#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/equation_solver.h"
#include "paddle/cinn/adt/igroup.h"
#include "paddle/cinn/adt/index_expr_infer_context.h"
#include "paddle/cinn/adt/kgroup.h"
#include "paddle/cinn/adt/naive_op_equation_context.h"
#include "paddle/cinn/adt/partition_op_stmts.h"
#include "paddle/cinn/adt/schedule_descriptor.h"

#include "glog/logging.h"

namespace cinn::adt {

namespace {

using LoopDescriptor4IterVarT =
    std::function<const LoopDescriptor&(const equation::IterVar&)>;

using AnchorTensor = equation::Variable;
using FakeOpPlaceHolders = List<equation::FakeOpPlaceHolder>;

m_expr::Op MakeOp(const hlir::framework::Node* op) { return {op}; }

template <typename DoEachT>
void VisitEachInputTensor(const hlir::framework::Node* op,
                          const DoEachT& DoEach) {
  for (const auto& graph_edge : op->inlinks_in_order()) {
    DoEach(graph_edge->source()->safe_as<hlir::framework::NodeData>());
  }
}

List<m_expr::Arg> MakeOpStmtInputList(const hlir::framework::Node* op,
                                      const hlir::framework::Graph* graph) {
  List<m_expr::Arg> ret{};

  VisitEachInputTensor(op, [&](const auto* tensor) {
    ret->emplace_back(adapter::Tensor{tensor, graph});
  });

  return ret;
}

template <typename DoEachT>
void VisitEachOutputTensor(const hlir::framework::Node* op,
                           const DoEachT& DoEach) {
  for (const auto& graph_edge : op->outlinks_in_order()) {
    DoEach(graph_edge->sink()->safe_as<hlir::framework::NodeData>());
  }
}

List<m_expr::Arg> MakeOpStmtOutputList(const hlir::framework::Node* op,
                                       const hlir::framework::Graph* graph) {
  List<m_expr::Arg> ret{};

  VisitEachOutputTensor(op, [&](const auto* tensor) {
    ret->emplace_back(adapter::Tensor{tensor, graph});
  });

  return ret;
}

template <typename DoEachT>
void VisitEachOpStmt(
    const std::shared_ptr<hlir::framework::Graph::Group>& group,
    const DoEachT& DoEach) {
  for (const auto* op : group.nodes) {
    // Tuple<Op, In<List<Arg>>, Out<List<Arg>>>
    DoEach(m_expr::OpStmt{MakeOp(op),
                          MakeOpStmtInputList(op, group->graph_),
                          MakeOpStmtOutputList(op, group->graph_)});
  }
}

List<m_expr::OpStmt> MakeOpStmts(
    const std::shared_ptr<hlir::framework::Graph::Group>& group) {
  List<m_expr::OpStmt> ret{};

  VisitEachOpStmt(group,
                  [&](const auto& op_stmt) { ret->emplace_back(op_stmt); });

  return ret;
}

template <typename DoEachT>
void PartitionIGroupOpStmts(const List<m_expr::OpStmt>& op_stmts,
                            const DoEachT& DoEach) {
  const auto& EquationCtx4OpStmt =
      config::GenerateContext4LocalOpStmt(op_stmts);
  const auto& igroup_specs =
      partition::PartitionOpStmts(EquationCtx4OpStmt, op_stmts);
  for (const auto& igroup_spec : igroup_specs) {
    DoEach(igroup_spec);
  }
}

std::shared_ptr<IGroup> MakeIGroup(const partition::AnchorGroup& igroup_spec) {
  CHECK(IsEquationSolvable(igroup_spec));
  return std::make_shared<IGroup>(igroup_spec.op_stmts,
                                  igroup_spec.anchor_index,
                                  igroup_spec.EquationCtx4OpStmt);
}

std::vector<std::shared_ptr<IGroup>> GenerateIGroups(
    const std::shared_ptr<hlir::framework::Graph::Group>& group) {
  std::vector<std::shared_ptr<IGroup>> ret{};

  List<m_expr::OpStmt> op_stmts = MakeOpStmts(group);

  PartitionIGroupOpStmts(op_stmts, [&](const auto& igroup_spec) {
    ret.push_back(MakeIGroup(igroup_spec));
  });

  return ret;
}

std::shared_ptr<KGroup> GenerateKGroups(
    const std::shared_ptr<hlir::framework::Graph::Group>& group,
    const std::vector<std::shared_ptr<IGroup>>& igroups) {
  CHECK_EQ(igroups.size(), 1);
  return std::make_shared<KGroup>(group, igroups);
}

equation::Equations MakeSdEquations(const std::shared_ptr<IGroup>& igroup,
                                    const ScheduleDescriptor& sd) {
  equation::config::AnchorSdEquationContext ctx{sd->size(),
                                                igroup->anchor_index()};
  igroup->set_anchor_sd_equation_ctx(ctx);

  return igroup->anchor_sd_equation_ctx().value().equations();
}

equation::GraphView GenerateSdEquationGraphView(
    const std::shared_ptr<IGroup>& igroup, const ScheduleDescriptor& sd) {
  equation::Equations equations = MakeSdEquations(igroup, sd);

  return equation::Graph(equations).GetGraphView();
}

equation::GraphView MakeEquationGraphView(const std::shared_ptr<IGroup>& igroup,
                                          const ScheduleDescriptor& sd) {
  return GenerateSdEquationGraphView(igroup, sd);
}

using TensorIndex = equation::Variable;
using TensorIndexExpr = equation::Value;

std::unordered_map<equation::Variable, const equation::Value>
MakeSdIterator2LoopDescriptor(const IGroup& igroup,
                              const ScheduleDescriptor& sd) {
  std::unordered_map<equation::Variable, const equation::Value> ret{};
  CHECK_EQ(igroup.loop_iterators()->size(), sd->size());

  for (std::size_t i = 0; i < sd->size(); ++i) {
    CHECK(ret.emplace(igroup.loop_iterators()->at(i), sd->at(i)).second);
  }

  return ret;
}

std::function<const TensorIndexExpr&(const m_expr::Tensor&)>
MakeGetterTensorIndexExpr(const std::shared_ptr<IGroup>& igroup,
                          const equation::GraphView& sd_equation_graph_view,
                          const ScheduleDescriptor& sd) {
  equation::GraphView igroup_view = igroup->GetDefaultGraphView();
  equation::GraphView merged_view = igroup_view.Merge(sd_equation_graph_view);

  const auto& init_var2value = MakeSdIterator2LoopDescriptor(*igroup, sd);
  auto ctx = std::make_shared<equation::IndexExprInferContext>(init_var2value);

  const std::vector<equation::Variable> starts{
      igroup->loop_iterators()->begin(), igroup->loop_iterators()->end()};
  equation::value::SolveEquations(merged_view, starts, ctx.get());
  return [ctx, igroup](const m_expr::Tensor& tensor) {
    // All indexes of same tensor have the same Value.
    const auto index = igroup->GetIndexes(tensor).at(0);
    return ctx->GetValue(index);
  };
}

LoopDescriptor4IterVarT MakeGetterLoopDescriptor4IterVar(
    const LoopIterators& loop_iters, const ScheduleDescriptor& sd) {
  CHECK_EQ(loop_iters->size(), sd->size());
  using Cache = std::unordered_map<equation::IterVar, LoopDescriptor>;
  const auto& sd_iter2sd = std::make_shared<Cache>();
  for (std::size_t i = 0; i < loop_iters->size(); ++i) {
    CHECK(sd_iter2sd->emplace(loop_iters->at(i), sd->at(i)).second);
  }
  return [sd_iter2sd](const auto& sd_iter) { return sd_iter2sd->at(sd_iter); };
}

using LoopIteratorsAndMapIrList = std::pair<LoopIterators, m_ir::MapIrList>;

List<LoopIteratorsAndMapIrList> GroupByFirstLoopIterators(
    const m_ir::MapIrList& map_irs) {
  CHECK(!map_irs->empty());

  const auto& VisitSkipPosition = [&](const auto& DoEach) {
    for (std::size_t i = 1; i < map_irs->size(); ++i) {
      const auto& prev_loop_iters = map_irs->at(i - 1).loop_iters_list()->at(0);
      const auto& next_loop_iters = map_irs->at(i).loop_iters_list()->at(0);
      if (prev_loop_iters != next_loop_iters) {
        DoEach(i);
      }
    }
  };

  const auto& VisitRangeWithSameFirstLoopIterators = [&](const auto& DoEach) {
    std::size_t begin = 0;
    VisitSkipPosition([&](std::size_t end) {
      DoEach(begin, end);
      begin = end;
    });
    DoEach(begin, map_irs->size());
  };

  const auto& GetFirstLoopIterators = [&](std::size_t begin) {
    return map_irs->at(begin).loop_iters_list()->at(0);
  };

  const auto& MakeMapIrList = [&](std::size_t begin, std::size_t end) {
    MapIrList map_ir_list{};
    for (std::size_t i = begin; i < end; ++i) {
      map_ir_list->emplace_back(map_irs->at(i));
    }
    return map_ir_list;
  };

  const auto& MakeLoopIteratorsAndMapIrList = [&](std::size_t begin,
                                                  std::size_t end) {
    return LoopIteratorsAndMapIrList{GetFirstLoopIterators(begin),
                                     MakeMapIrList(begin, end)};
  };

  List<m_ir::MapIrList> ret{};
  VisitRangeWithSameFirstLoopIterators([&](std::size_t begin, std::size_t end) {
    ret->emplace_back(MakeLoopIteratorsAndMapIrList(begin, end));
  });

  return ret;
}

m_expr::MapStmt<m_expr::Stmt> MakeMapStmt(
    const m_ir::MapIrList& map_irs,
    const LoopDescriptor4IterVarT& LoopDescriptor4IterVar);

void CheckFirstLoopAllSame(const m_ir::MapIrList& map_irs) {
  if (map_irs->empty()) {
    return;
  }
  for (std::size_t i = 1; i < map_irs->size(); ++i) {
    const auto& prev_loop_iters = map_irs->at(i - 1).loop_iters_list()->at(0);
    const auto& next_loop_iters = map_irs->at(i).loop_iters_list()->at(0);
    CHECK(prev_loop_iters == next_loop_iters);
  }
}

LoopIteratorsAndMapIrList GetStrippedMapIrs(const m_ir::MapIrList& map_irs) {
  CheckFirstLoopAllSame(map_irs);
  m_ir::MapIrList ret_map_irs{};
  for (const auto& map_ir : *map_irs) {
    const auto& op_stmts = map_ir.op_stmts();
    const auto& origin_loops = map_ir.loop_iters_list();
    List<LoopIterators> loop_iters_list{std::next(origin_loops->begin()),
                                        origin_loops->end()};
    ret_map_irs->emplace_back(MapIr{op_stmts, loop_iters_list});
  }
  return {map_irs->at(0).loop_iters_list()->at(0), ret_map_irs};
}

List<m_expr::Stmt> MakeStmtList(const m_ir::MapIrList& map_irs) {
  const auto& grouped_map_irs = GroupByFirstLoopIterators(map_irs);
  List<m_expr::Stmt> ret{};

  const auto& CollectOpStmts = [&](const auto& inner_map_irs) {
    for (const auto& map_ir : inner_map_irs) {
      CHECK(map_ir.loop_iters_list()->empty());
      for (const auto& op_stmt : map_ir.op_stmts()) {
        ret->emplace_back(op_stmt);
      }
    }
  };

  for (const auto& [loop_iters, inner_map_irs] : grouped_map_irs) {
    if (loop_iters.empty()) {
      CollectOpStmts(inner_map_irs);
    } else {
      ret->emplace_back({MakeMapStmt(inner_map_irs)});
    }
  }

  return ret;
}

m_expr::MapStmt<m_expr::Stmt> MakeMapStmt(
    const m_ir::MapIrList& map_irs,
    const LoopDescriptor4IterVarT& LoopDescriptor4IterVar) {
  const auto& [first_loop_iters, first_stripped_map_irs] =
      GetStrippedMapIrs(map_irs);

  return m_expr::MapStmt<m_expr::Stmt>{
      MakeScheduleDescriptor(first_loop_iters, LoopDescriptor4IterVar),
      MakeStmtList(first_stripped_map_irs)};
}

m_expr::Tensor GetAnchorTensor(const std::shared_ptr<IGroup>& igroup) {
  return igroup->anchor_tensor();
}

template <typename DoEachT>
void VisitInputTensor(const hlir::framework::Graph::Group& group,
                      const DoEachT& DoEach) {
  for (const auto* node_data : group->GetInputNodeDatas()) {
    DoEach(node_data, group->graph_);
  }
}

template <typename DoEachT>
void VisitOutputTensor(const hlir::framework::Graph::Group& group,
                       const DoEachT& DoEach) {
  for (const auto& node_data : group->GetOutputNodeDatas()) {
    DoEach(node_data, group->graph_);
  }
}

List<m_expr::Tensor> MakeInputTensors(const std::shared_ptr<KGroup>& kgroup) {
  List<m_expr::Tensor> ret{};
  VisitInputTensor(*kgroup->cinn_group(),
                   [&](const auto* node_data, const auto* graph) {
                     ret->emplace_back(adapter::Tensor{node_data, graph});
                   });
  return ret;
}

List<m_expr::Tensor> MakeOutputTensors(const std::shared_ptr<KGroup>& kgroup) {
  List<m_expr::Tensor> ret{};
  VisitOutputTensor(*kgroup->cinn_group(),
                    [&](const auto* node_data, const auto* graph) {
                      ret->emplace_back(adapter::Tensor{node_data, graph});
                    });
  return ret;
}

m_expr::AnchoredMapStmt GenerateAnchoredMapStmt(
    const std::shared_ptr<IGroup>& igroup,
    const LoopIterators& loop_iters,
    const ScheduleDescriptor& sd,
    const m_expr::TensorIndexExpr4TensorT& TensorIndexExpr4Tensor) {
  const auto& LoopDescriptor4IterVar =
      MakeGetterLoopDescriptor4IterVar(loop_iters, sd);

  const auto& map_irs =
      m_ir::GenerateMapIrListForLoopFuse(igroup->op_stmts(),
                                         loop_iters,
                                         LoopDescriptor4IterVar,
                                         TensorIndexExpr4Tensor);

  // AnchoredMapStmt = (MapStmt Stmt, tAnchor Tensor, TensorIndexExpr4TensorT)
  return m_expr::AnchoredMapStmt{MakeMapStmt(map_irs, LoopDescriptor4IterVar),
                                 GetAnchorTensor(igroup),
                                 TensorIndexExpr4Tensor};
}

m_expr::AnchoredMapStmt GenerateAnchoredMapStmt(
    const std::shared_ptr<IGroup>& igroup, const ScheduleDescriptor& sd) {
  const auto& sd_equation_graph_view = MakeEquationGraphView(igroup, sd);

  const auto& TensorIndexExpr4Tensor =
      MakeGetterTensorIndexExpr(igroup, sd_equation_graph_view, sd);

  CHECK(igroup->anchor_sd_equation_ctx().has_value());
  const auto& schedule_iters =
      igroup->anchor_sd_equation_ctx().value().loop_iterators();

  return GenerateAnchoredMapStmt(
      igroup, schedule_iters, sd, TensorIndexExpr4Tensor);
}

List<m_expr::AnchoredMapStmt> MakeAnchoredMapStmts(
    const std::shared_ptr<KGroup>& kgroup) {
  List<m_expr::AnchoredMapStmt> ret{};
  for (const auto& igroup : kgroup->igroups()) {
    const auto& sd = kgroup->GetDefaultScheduleDescriptor(igroup);
    ret->emplace_back(GenerateAnchoredMapStmt(igroup, sd));
  }
  return ret;
}

m_expr::MapExpr GenerateMapExpr(const std::shared_ptr<KGroup>& kgroup) {
  // MapExpr = Kernel;
  // Kernel = ([AnchoredMapStmt], In [Tensor], Out [Tensor])
  return m_expr::MapExpr{MakeAnchoredMapStmts(kgroup),
                         MakeInputTensors(kgroup),
                         MakeOutputTensors(kgroup)};
}

}  // namespace

m_expr::MapExpr GenerateMapExpr(
    const std::shared_ptr<hlir::framework::Graph::Group>& group) {
  const auto& igroups = GenerateIGroups(group);

  const auto& kgroup = GenerateKGroups(group, igroups);

  return GenerateMapExpr(kgroup);
}

}  // namespace cinn::adt
