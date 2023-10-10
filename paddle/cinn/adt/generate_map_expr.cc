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
#include "paddle/cinn/adt/naive_bidirection_equation_generator.h"
#include "paddle/cinn/adt/naive_op_equation_context.h"
#include "paddle/cinn/adt/partition_op_stmts.h"
#include "paddle/cinn/adt/print_map_expr.h"
#include "paddle/cinn/adt/schedule_descriptor.h"
#include "paddle/cinn/adt/tree.h"
#include "paddle/cinn/runtime/flags.h"

#include "glog/logging.h"

PD_DECLARE_bool(cinn_enable_map_expr);

namespace cinn::adt {

template <>
struct TreeMerger<Stmt> {
  using TreeT = Stmt;
  using tree_type = TreeT;
  using inner_type = typename TreeTrait<TreeT>::inner_type;
  using leaf_type = typename TreeTrait<TreeT>::leaf_type;

  using inner_data_type = typename inner_type::value_type;
  std::function<inner_data_type(const leaf_type&)> GetInnerDataForLeaf;

  inner_type MakeInnerNode(const inner_data_type& inner_data,
                           const List<TreeT>& children) const {
    return MapStmt<Stmt>{inner_data, children};
  }

  using MergeResult = std::tuple<tCommon<inner_data_type>,
                                 tLhsRemainder<inner_data_type>,
                                 tRhsRemainder<inner_data_type>>;

  MergeResult MergeInnerValue(const inner_data_type& lhs,
                              const inner_data_type& rhs) const {
    inner_data_type common{};
    inner_data_type lhs_remainder{};
    inner_data_type rhs_remainder{};
    int min_size = std::min(lhs->size(), rhs->size());
    int idx = 0;
    for (; idx < min_size; ++idx) {
      if (lhs->at(idx) == rhs->at(idx)) {
        common->emplace_back(lhs->at(idx));
      } else {
        break;
      }
    }
    for (int lhs_idx = idx; lhs_idx < lhs->size(); ++lhs_idx) {
      lhs_remainder->emplace_back(lhs->at(lhs_idx));
    }
    for (int rhs_idx = idx; rhs_idx < rhs->size(); ++rhs_idx) {
      rhs_remainder->emplace_back(rhs->at(rhs_idx));
    }
    return MergeResult{common, lhs_remainder, rhs_remainder};
  }
};

namespace {

using LoopDescriptor4IterVarT = std::function<LoopDescriptor(const Iterator&)>;

using AnchorTensor = Variable;
using FakeOpPlaceHolders = List<FakeOpPlaceHolder>;

Op MakeOp(const hlir::framework::Node* op) { return {op}; }

template <typename DoEachT>
void VisitEachInputTensor(const hlir::framework::Node* op,
                          const DoEachT& DoEach) {
  for (const auto& graph_edge : op->inlinks_in_order()) {
    DoEach(graph_edge->source()->safe_as<hlir::framework::NodeData>());
  }
}

List<Arg> MakeOpStmtInputList(const hlir::framework::Node* op,
                              const hlir::framework::Graph* graph) {
  List<Arg> ret{};

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

List<Arg> MakeOpStmtOutputList(const hlir::framework::Node* op,
                               const hlir::framework::Graph* graph) {
  List<Arg> ret{};

  VisitEachOutputTensor(op, [&](const auto* tensor) {
    ret->emplace_back(adapter::Tensor{tensor, graph});
  });

  return ret;
}

template <typename DoEachT>
void VisitEachOpStmt(
    const std::shared_ptr<hlir::framework::Graph::Group>& group,
    const DoEachT& DoEach) {
  // Note
  for (const auto* op : group->nodes) {
    DoEach(OpStmt{MakeOp(op),
                  MakeOpStmtInputList(op, group->graph_),
                  MakeOpStmtOutputList(op, group->graph_)});
  }
}

hlir::framework::OpPatternKind GetOpPatternKind(
    const hlir::framework::Node* node) {
  static const hlir::framework::OpValueType<hlir::framework::OpPatternKind>&
      op_pattern_dict =
          hlir::framework::Operator::GetAttrs<hlir::framework::OpPatternKind>(
              "OpPattern");
  auto kind = op_pattern_dict[node->op()];
  return kind;
}

bool CollectRewritedReductionOpStmts(const OpStmt& op_stmt, List<OpStmt>* ret) {
  const auto& [op, inputs, outputs] = op_stmt.tuple();
  CHECK(op.Has<const hlir::framework::Node*>());
  if (GetOpPatternKind(op.Get<const hlir::framework::Node*>()) ==
      hlir::framework::OpPatternKind::kReduction) {
    tReduceInit<const hlir::framework::Node*> init_op{
        op.Get<const hlir::framework::Node*>()};
    (*ret)->emplace_back(OpStmt{init_op, List<Arg>{}, outputs});

    tReduceAcc<const hlir::framework::Node*> acc_op{
        op.Get<const hlir::framework::Node*>()};
    (*ret)->emplace_back(OpStmt{acc_op, inputs, outputs});
    return true;
  } else {
    return false;
  }
}

void CollectRewritedOpStmts(const OpStmt& op_stmt, List<OpStmt>* ret) {
  if (CollectRewritedReductionOpStmts(op_stmt, ret)) {
    return;
  }
  (*ret)->emplace_back(op_stmt);
}

List<OpStmt> MakeOpStmts(
    const std::shared_ptr<hlir::framework::Graph::Group>& group) {
  List<OpStmt> ret{};

  VisitEachOpStmt(group, [&](const auto& op_stmt) {
    CollectRewritedOpStmts(op_stmt, &ret);
  });

  return ret;
}

template <typename DoEachT>
void PartitionIGroupOpStmts(const List<OpStmt>& op_stmts,
                            const DoEachT& DoEach) {
  const auto& EquationCtx4OpStmt =
      config::GenerateContext4LocalOpStmt(op_stmts);
  auto direction_equation_generator =
      std::make_shared<NaiveBidirectionEquationGenerator>(op_stmts,
                                                          EquationCtx4OpStmt);
  const auto& igroup_specs = PartitionOpStmts(
      EquationCtx4OpStmt, op_stmts, direction_equation_generator);
  for (const auto& igroup_spec : igroup_specs) {
    DoEach(igroup_spec);
  }
}

std::shared_ptr<IGroup> MakeIGroup(const AnchorGroup& igroup_spec) {
  std::shared_ptr<const EquationFunctionConstantsProvider> constants_provider{
      new NaiveEquationFunctionConstantsProvider{
          igroup_spec.op_stmts, igroup_spec.EquationCtx4OpStmt}};
  std::shared_ptr<DirectionEquationGenerator> direction_equation_generator{
      new NaiveBidirectionEquationGenerator{igroup_spec.op_stmts,
                                            igroup_spec.EquationCtx4OpStmt}};
  CheckEquationSolvable(
      igroup_spec, constants_provider, direction_equation_generator);
  return std::make_shared<IGroup>(igroup_spec.op_stmts,
                                  igroup_spec.anchor_index,
                                  igroup_spec.EquationCtx4OpStmt,
                                  constants_provider);
}

std::vector<std::shared_ptr<IGroup>> GenerateIGroups(
    const std::shared_ptr<hlir::framework::Graph::Group>& group) {
  std::vector<std::shared_ptr<IGroup>> ret{};

  List<OpStmt> op_stmts = MakeOpStmts(group);

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

GraphView GenerateSdEquationGraphView(const std::shared_ptr<IGroup>& igroup,
                                      const ScheduleMesh& sched_mesh) {
  config::AnchorSdEquationContext ctx{sched_mesh, igroup->anchor_index()};
  igroup->set_anchor_sd_equation_ctx(ctx);

  Equations equations = igroup->anchor_sd_equation_ctx().value().equations();

  return Graph::New(equations)->GetGraphView();
}

using TensorIndexExpr = Value;

std::unordered_map<Variable, const Value> MakeSdIterator2Iterator(
    const IGroup& igroup) {
  std::unordered_map<Variable, const Value> ret{};

  for (std::size_t i = 0; i < igroup.loop_iterators()->size(); ++i) {
    CHECK(ret.emplace(igroup.loop_iterators()->at(i),
                      igroup.loop_iterators()->at(i))
              .second);
  }

  return ret;
}

std::function<TensorIndexExpr(const Tensor&)> MakeGetterTensorIndexExpr(
    const std::shared_ptr<IGroup>& igroup, const ScheduleMesh& sched_mesh) {
  const auto& sd_equation_graph_view =
      GenerateSdEquationGraphView(igroup, sched_mesh);

  GraphView igroup_view = igroup->GetDefaultGraphView();
  GraphView merged_view = igroup_view.Merge(sd_equation_graph_view);

  const auto& init_var2value = MakeSdIterator2Iterator(*igroup);
  auto ctx = std::make_shared<IndexExprInferContext>(
      init_var2value, igroup->constants_provider());

  std::vector<Variable> starts{};
  for (const auto& loop_iterator : *igroup->loop_iterators()) {
    starts.emplace_back(loop_iterator);
  }
  SolveEquations(merged_view, starts, ctx.get());
  return [ctx, igroup](const Tensor& tensor) {
    // All indexes of same tensor have the same Value.
    const auto index = igroup->GetIndexes(tensor).at(0);
    return ctx->GetValue(index);
  };
}

LoopDescriptor4IterVarT MakeGetterLoopDescriptor4IterVar(
    const LoopIterators& loop_iters, const LoopDescriptors& sd) {
  CHECK_EQ(loop_iters->size(), sd->size());
  using Cache = std::unordered_map<Iterator, LoopDescriptor>;
  const auto& sd_iter2sd = std::make_shared<Cache>();
  for (std::size_t i = 0; i < loop_iters->size(); ++i) {
    CHECK(sd_iter2sd->emplace(loop_iters->at(i), sd->at(i)).second);
  }
  return [sd_iter2sd](const auto& sd_iter) { return sd_iter2sd->at(sd_iter); };
}

TreeMerger<Stmt> MakeTreeMerger(const MapIr& map_ir) {
  using Cache = std::unordered_map<OpStmt, LoopIterators>;
  auto cache = std::make_shared<Cache>();
  for (const auto& op_stmt : *(map_ir.op_stmts())) {
    CHECK(cache->emplace(op_stmt, map_ir.loop_iterators()).second);
  }

  TreeMerger<Stmt> tree_merger{};
  tree_merger.GetInnerDataForLeaf =
      ([=](const OpStmt& op_stmt) -> LoopIterators {
        return cache->at(op_stmt);
      });
  return tree_merger;
}

MapStmt<Stmt> MakeMapStmt(const MapIrList& map_irs) {
  List<Stmt> stmts{};
  for (const auto& map_ir : *map_irs) {
    const TreeMerger<Stmt>& tree_merger = MakeTreeMerger(map_ir);
    MergeTrees(tree_merger, &stmts, map_ir.op_stmts());
  }
  CHECK_EQ(stmts->size(), 1);
  CHECK(stmts->at(0).Has<MapStmt<Stmt>>());
  return stmts->at(0).Get<MapStmt<Stmt>>();
}

Tensor GetAnchorTensor(const std::shared_ptr<IGroup>& igroup) {
  return igroup->anchor_tensor();
}

template <typename DoEachT>
void VisitInputTensor(const hlir::framework::Graph::Group& group,
                      const DoEachT& DoEach) {
  for (const auto* node_data : group.GetInputNodeDatas()) {
    DoEach(node_data, group.graph_);
  }
}

template <typename DoEachT>
void VisitOutputTensor(const hlir::framework::Graph::Group& group,
                       const DoEachT& DoEach) {
  for (const auto& node_data : group.GetOutputNodeDatas()) {
    DoEach(node_data, group.graph_);
  }
}

List<Tensor> MakeInputTensors(const std::shared_ptr<KGroup>& kgroup) {
  List<Tensor> ret{};
  VisitInputTensor(*kgroup->cinn_group(),
                   [&](const auto* node_data, const auto* graph) {
                     ret->emplace_back(adapter::Tensor{node_data, graph});
                   });
  return ret;
}

List<Tensor> MakeOutputTensors(const std::shared_ptr<KGroup>& kgroup) {
  List<Tensor> ret{};
  VisitOutputTensor(*kgroup->cinn_group(),
                    [&](const auto* node_data, const auto* graph) {
                      ret->emplace_back(adapter::Tensor{node_data, graph});
                    });
  return ret;
}

AnchoredMapStmt GenerateAnchoredMapStmt(
    const std::shared_ptr<IGroup>& igroup,
    const LoopIterators& loop_iters,
    const ScheduleMesh& sched_mesh,
    const LoopDescriptors& sd,
    const TensorIndexExpr4TensorT& TensorIndexExpr4Tensor) {
  const auto& LoopDescriptor4IterVar =
      MakeGetterLoopDescriptor4IterVar(loop_iters, sd);

  const auto& map_irs = GenerateMapIrListForLoopFuse(igroup->op_stmts(),
                                                     loop_iters,
                                                     LoopDescriptor4IterVar,
                                                     TensorIndexExpr4Tensor);
  return AnchoredMapStmt{MakeMapStmt(map_irs),
                         sched_mesh,
                         GetAnchorTensor(igroup),
                         TensorIndexExpr4Tensor,
                         LoopDescriptor4IterVar};
}

AnchoredMapStmt GenerateAnchoredMapStmt(const std::shared_ptr<IGroup>& igroup) {
  const auto& [sched_mesh, loop_types] =
      CreateOptimizedScheduleMesh(igroup->anchor_schedule_dims());

  const auto& sd = CreateScheduleDescriptor(sched_mesh, loop_types);

  const auto& TensorIndexExpr4Tensor =
      MakeGetterTensorIndexExpr(igroup, sched_mesh);

  const auto& schedule_iters = igroup->loop_iterators();

  return GenerateAnchoredMapStmt(
      igroup, schedule_iters, sched_mesh, sd, TensorIndexExpr4Tensor);
}

List<AnchoredMapStmt> MakeAnchoredMapStmts(
    const std::shared_ptr<KGroup>& kgroup) {
  List<AnchoredMapStmt> ret{};
  for (const auto& igroup : kgroup->igroups()) {
    ret->emplace_back(GenerateAnchoredMapStmt(igroup));
  }
  return ret;
}

MapExpr GenerateMapExpr(const std::shared_ptr<KGroup>& kgroup) {
  // MapExpr = Kernel;
  // Kernel = ([AnchoredMapStmt], In [Tensor], Out [Tensor])
  return MapExpr{MakeAnchoredMapStmts(kgroup),
                 MakeInputTensors(kgroup),
                 MakeOutputTensors(kgroup)};
}

}  // namespace

MapExpr GenerateMapExpr(
    const std::shared_ptr<hlir::framework::Graph::Group>& group) {
  const auto& igroups = GenerateIGroups(group);

  const auto& kgroup = GenerateKGroups(group, igroups);

  return GenerateMapExpr(kgroup);
}

namespace {}  // namespace

void TryGenerateMapExprFromGraph(
    const std::shared_ptr<cinn::hlir::framework::Graph>& graph) {
  if (!FLAGS_cinn_enable_map_expr) {
    return;
  }
  for (const auto& fusion_group : graph->fusion_groups) {
    const auto& map_expr = GenerateMapExpr(fusion_group);
    PrintMapExpr(map_expr, fusion_group->group_id);
  }
}

}  // namespace cinn::adt
