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
#include "paddle/cinn/adt/map_expr_ctx.h"
#include "paddle/cinn/adt/naive_bidirection_equation_generator.h"
#include "paddle/cinn/adt/naive_op_equation_context.h"
#include "paddle/cinn/adt/partition_op_stmts.h"
#include "paddle/cinn/adt/print.h"
#include "paddle/cinn/adt/schedule_descriptor.h"
#include "paddle/cinn/adt/tree.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"

#include "glog/logging.h"
#include "paddle/common/enforce.h"

PD_DECLARE_bool(cinn_enable_map_expr);
PD_DECLARE_bool(cinn_map_expr_enable_dynamic_shape);

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

using FakeOpPlaceHolders = List<FakeOpPlaceHolder>;

Op MakeOp(const ::pir::Operation* op) { return {op}; }

template <typename DoEachT>
void VisitEachInputTensor(const ::pir::Operation* op, const DoEachT& DoEach) {
  for (std::size_t i = 0; i < op->num_operands(); ++i) {
    DoEach(op->operand_source(i));
  }
}

bool HasDynamicShape(const ::pir::Value& tensor) {
  const auto& shape = hlir::framework::pir::CompatibleInfo::ValueShape(tensor);
  for (int dim : shape) {
    if (dim < 0) {
      PADDLE_ENFORCE_EQ(
          dim,
          -1UL,
          ::common::errors::InvalidArgument(
              "The dynamic shape dim should be -1, but got %d.", dim));
      return true;
    }
  }
  return false;
}

List<Arg> MakeOpStmtInputList(
    const ::pir::Operation* op,
    const hlir::framework::pir::OpLoweringGroup* group) {
  List<Arg> ret{};

  VisitEachInputTensor(op, [&](const ::pir::Value& tensor) {
    if (HasDynamicShape(tensor)) {
      ret->emplace_back(adapter::DynamicTensor{tensor, group});
    } else {
      ret->emplace_back(adapter::Tensor{tensor});
    }
  });

  return ret;
}

template <typename DoEachT>
void VisitEachOutputTensor(const ::pir::Operation* op, const DoEachT& DoEach) {
  for (std::size_t i = 0; i < op->num_results(); ++i) {
    DoEach(const_cast<::pir::Operation*>(op)->result(i));
  }
}

List<Arg> MakeOpStmtOutputList(
    const ::pir::Operation* op,
    const hlir::framework::pir::OpLoweringGroup* group) {
  List<Arg> ret{};

  VisitEachOutputTensor(op, [&](const ::pir::Value& tensor) {
    if (HasDynamicShape(tensor)) {
      ret->emplace_back(adapter::DynamicTensor{tensor, group});
    } else {
      ret->emplace_back(adapter::Tensor{tensor});
    }
  });

  return ret;
}

template <typename DoEachT>
void VisitEachOpStmt(
    const std::shared_ptr<hlir::framework::pir::OpLoweringGroup>& group,
    const DoEachT& DoEach) {
  for (const auto* op : group->ops()) {
    DoEach(OpStmt{MakeOp(op),
                  MakeOpStmtInputList(op, group.get()),
                  MakeOpStmtOutputList(op, group.get())});
  }
}

hlir::framework::OpPatternKind GetOpPatternKind(const ::pir::Operation* node) {
  return hlir::framework::pir::CompatibleInfo::OpKind(*node);
}

bool CollectRewrittenReductionOpStmts(const OpStmt& op_stmt,
                                      List<OpStmt>* ret) {
  const auto& [op, inputs, outputs] = op_stmt.tuple();
  PADDLE_ENFORCE_EQ(
      op.Has<const ::pir::Operation*>(),
      true,
      ::common::errors::InvalidArgument(
          "The op should have a value of type ::pir::Operation*"));
  if (GetOpPatternKind(op.Get<const ::pir::Operation*>()) ==
      hlir::framework::OpPatternKind::kReduction) {
    tReduceInit<const ::pir::Operation*> init_op{
        op.Get<const ::pir::Operation*>()};
    (*ret)->emplace_back(OpStmt{init_op, List<Arg>{}, outputs});

    tReduceAcc<const ::pir::Operation*> acc_op{
        op.Get<const ::pir::Operation*>()};
    (*ret)->emplace_back(OpStmt{acc_op, inputs, outputs});
    return true;
  } else {
    return false;
  }
}

void CollectRewrittenOpStmts(const OpStmt& op_stmt, List<OpStmt>* ret) {
  if (CollectRewrittenReductionOpStmts(op_stmt, ret)) {
    return;
  }
  (*ret)->emplace_back(op_stmt);
}

List<OpStmt> MakeOpStmts(
    const std::shared_ptr<hlir::framework::pir::OpLoweringGroup>& group) {
  List<OpStmt> ret{};

  VisitEachOpStmt(group, [&](const auto& op_stmt) {
    CollectRewrittenOpStmts(op_stmt, &ret);
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
  std::shared_ptr<DirectionEquationGenerator> direction_equation_generator{
      new NaiveBidirectionEquationGenerator{igroup_spec.op_stmts,
                                            igroup_spec.EquationCtx4OpStmt}};
  CheckEquationSolvable(igroup_spec, direction_equation_generator);
  return std::make_shared<IGroup>(igroup_spec.op_stmts,
                                  igroup_spec.anchor_index,
                                  igroup_spec.EquationCtx4OpStmt);
}

std::vector<std::shared_ptr<IGroup>> GenerateIGroups(
    const std::shared_ptr<hlir::framework::pir::OpLoweringGroup>& group) {
  std::vector<std::shared_ptr<IGroup>> ret{};

  List<OpStmt> op_stmts = MakeOpStmts(group);
  PADDLE_ENFORCE_EQ(
      !op_stmts->empty(),
      true,
      ::common::errors::InvalidArgument("The op_stmts should not be empty"));

  PartitionIGroupOpStmts(op_stmts, [&](const auto& igroup_spec) {
    ret.push_back(MakeIGroup(igroup_spec));
  });

  return ret;
}

std::shared_ptr<KGroup> GenerateKGroups(
    const std::shared_ptr<hlir::framework::pir::OpLoweringGroup>& group,
    const std::vector<std::shared_ptr<IGroup>>& igroups) {
  PADDLE_ENFORCE_EQ(
      igroups.size(),
      1UL,
      ::common::errors::InvalidArgument(
          "The size of igroups should be 1, but got %d.", igroups.size()));
  return std::make_shared<KGroup>(group, igroups);
}

GraphView GenerateSdEquationGraphView(const std::shared_ptr<IGroup>& igroup,
                                      const ScheduleMesh& sched_mesh) {
  config::AnchorSdEquationContext ctx{sched_mesh, igroup->anchor_index()};
  igroup->set_anchor_sd_equation_ctx(ctx);

  Equations equations = igroup->anchor_sd_equation_ctx().value().equations();

  return Graph<Variable, Equation>::New(equations)->GetGraphView();
}

using TensorIndexExpr = Value;

std::unordered_map<Variable, const Value> MakeSdIterator2Iterator(
    const IGroup& igroup) {
  std::unordered_map<Variable, const Value> ret{};

  for (std::size_t i = 0; i < igroup.loop_iterators()->size(); ++i) {
    PADDLE_ENFORCE_EQ(ret.emplace(igroup.loop_iterators()->at(i),
                                  igroup.loop_iterators()->at(i))
                          .second,
                      true,
                      ::common::errors::InvalidArgument(
                          "The loop iterator should be unique"));
  }

  return ret;
}

std::shared_ptr<IndexExprInferContext> SolveEquationsThenReturnCtx(
    const std::shared_ptr<IGroup>& igroup, const ScheduleMesh& sched_mesh) {
  const auto& sd_equation_graph_view =
      GenerateSdEquationGraphView(igroup, sched_mesh);

  GraphView igroup_view = igroup->GetDefaultGraphView();
  GraphView merged_view = igroup_view.Merge(sd_equation_graph_view);

  const auto& init_var2value = MakeSdIterator2Iterator(*igroup);
  auto ctx = std::make_shared<IndexExprInferContext>(init_var2value);

  std::vector<Variable> starts{};
  for (const auto& loop_iterator : *igroup->loop_iterators()) {
    starts.emplace_back(loop_iterator);
  }
  SolveEquations(merged_view, starts, ctx.get());
  return ctx;
}

std::function<TensorIndexExpr(const Tensor&)> MakeGetterTensorIndexExpr(
    const std::shared_ptr<IndexExprInferContext>& ctx,
    const std::shared_ptr<IGroup>& igroup) {
  return [ctx, igroup](const Tensor& tensor) {
    // All indexes of same tensor have the same Value.
    const auto& index = igroup->GetIndexes(tensor).at(0);
    return ctx->GetValue(index);
  };
}

TensorIteratorExpr4TensorT MakeGetterTensorIteratorExpr4Tensor(
    const std::shared_ptr<IndexExprInferContext>& ctx,
    const std::shared_ptr<IGroup>& igroup) {
  return [ctx, igroup](const Tensor& tensor) -> List<TensorIteratorExpr> {
    const auto& iterators = igroup->GetTensorIterators(tensor);
    List<TensorIteratorExpr> ret{};
    for (const auto& iterator : *iterators) {
      ret->emplace_back(ctx->GetValue(iterator));
    }
    return ret;
  };
}

LoopDescriptor4IterVarT MakeGetterLoopDescriptor4IterVar(
    const LoopIterators& loop_iters, const LoopDescriptors& sd) {
  PADDLE_ENFORCE_EQ(
      loop_iters->size(),
      sd->size(),
      ::common::errors::InvalidArgument(
          "The size of loop iterators and loop descriptors should be equal, "
          "but got loop iterators size = %d, loop descriptors size = %d.",
          loop_iters->size(),
          sd->size()));
  using Cache = std::unordered_map<Iterator, LoopDescriptor>;
  const auto& sd_iter2sd = std::make_shared<Cache>();
  for (std::size_t i = 0; i < loop_iters->size(); ++i) {
    PADDLE_ENFORCE_EQ(sd_iter2sd->emplace(loop_iters->at(i), sd->at(i)).second,
                      true,
                      ::common::errors::InvalidArgument(
                          "The loop iterator should be unique"));
  }
  return [sd_iter2sd](const auto& sd_iter) { return sd_iter2sd->at(sd_iter); };
}

TreeMerger<Stmt> MakeTreeMerger(const MapIr& map_ir) {
  using Cache = std::unordered_map<OpStmt, LoopIterators>;
  auto cache = std::make_shared<Cache>();
  for (const auto& op_stmt : *(map_ir.op_stmts())) {
    PADDLE_ENFORCE_EQ(
        cache->emplace(op_stmt, map_ir.loop_iterators()).second,
        true,
        ::common::errors::InvalidArgument("The op_stmt should be unique"));
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
  PADDLE_ENFORCE_EQ(
      stmts->size(),
      1UL,
      ::common::errors::InvalidArgument(
          "The size of stmts should be 1, but got %d.", stmts->size()));
  PADDLE_ENFORCE_EQ(stmts->at(0).Has<MapStmt<Stmt>>(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The stmts should have a value of type MapStmt<Stmt>"));
  return stmts->at(0).Get<MapStmt<Stmt>>();
}

Tensor GetAnchorTensor(const std::shared_ptr<IGroup>& igroup) {
  return igroup->anchor_tensor();
}

template <typename DoEachT>
void VisitInputTensor(const hlir::framework::pir::OpLoweringGroup& group,
                      const DoEachT& DoEach) {
  for (const ::pir::Value& node_data : group.GetInputOpValues()) {
    DoEach(node_data);
  }
}

template <typename DoEachT>
void VisitOutputTensor(const hlir::framework::pir::OpLoweringGroup& group,
                       const DoEachT& DoEach) {
  for (const ::pir::Value& node_data : group.GetOutputOpValues()) {
    DoEach(node_data);
  }
}

List<Tensor> MakeInputTensors(const std::shared_ptr<KGroup>& kgroup) {
  List<Tensor> ret{};
  VisitInputTensor(*kgroup->cinn_group(), [&](const ::pir::Value& node_data) {
    ret->emplace_back(adapter::Tensor{node_data});
  });
  return ret;
}

List<Tensor> MakeOutputTensors(const std::shared_ptr<KGroup>& kgroup) {
  List<Tensor> ret{};
  VisitOutputTensor(*kgroup->cinn_group(), [&](const ::pir::Value& node_data) {
    ret->emplace_back(adapter::Tensor{node_data});
  });
  return ret;
}

AnchoredMapStmt GenerateAnchoredMapStmt(
    const std::shared_ptr<IGroup>& igroup,
    const LoopIterators& loop_iters,
    const ScheduleMesh& sched_mesh,
    const LoopDescriptors& sd,
    const TensorIndexExpr4TensorT& TensorIndexExpr4Tensor,
    const TensorIteratorExpr4TensorT& TensorIteratorExpr4Tensor) {
  const auto& LoopDescriptor4IterVar =
      MakeGetterLoopDescriptor4IterVar(loop_iters, sd);

  const auto& map_irs = GenerateMapIrListForLoopFuse(
      igroup->op_stmts(), loop_iters, TensorIndexExpr4Tensor);
  return AnchoredMapStmt{MakeMapStmt(map_irs),
                         sched_mesh,
                         GetAnchorTensor(igroup),
                         TensorIndexExpr4Tensor,
                         TensorIteratorExpr4Tensor,
                         LoopDescriptor4IterVar};
}

AnchoredMapStmt GenerateAnchoredMapStmt(const std::shared_ptr<IGroup>& igroup) {
  const auto& [sched_mesh, loop_types] =
      CreateOptimizedScheduleMesh(igroup->anchor_schedule_dims());

  const auto& sd = CreateScheduleDescriptor(sched_mesh, loop_types);

  const auto& ctx = SolveEquationsThenReturnCtx(igroup, sched_mesh);
  const auto& TensorIndexExpr4Tensor = MakeGetterTensorIndexExpr(ctx, igroup);
  const auto& TensorIteratorExpr4Tensor =
      MakeGetterTensorIteratorExpr4Tensor(ctx, igroup);

  const auto& schedule_iters = igroup->loop_iterators();

  return GenerateAnchoredMapStmt(igroup,
                                 schedule_iters,
                                 sched_mesh,
                                 sd,
                                 TensorIndexExpr4Tensor,
                                 TensorIteratorExpr4Tensor);
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
    const std::shared_ptr<hlir::framework::pir::OpLoweringGroup>& group) {
  const auto& igroups = GenerateIGroups(group);

  const auto& kgroup = GenerateKGroups(group, igroups);

  return GenerateMapExpr(kgroup);
}

void TryGenerateMapExprFromGroup(
    const std::shared_ptr<hlir::framework::pir::OpLoweringGroup>&
        fusion_group) {
  if (!FLAGS_cinn_enable_map_expr) {
    return;
  }
  const auto& map_expr = GenerateMapExpr(fusion_group);
  VLOG(4) << "Generate MapExpr: \n"
          << ToTxtString(map_expr, fusion_group->group_id());
  fusion_group->set_map_expr_ctx(std::make_shared<MapExprCtx>(map_expr));
}

}  // namespace cinn::adt
