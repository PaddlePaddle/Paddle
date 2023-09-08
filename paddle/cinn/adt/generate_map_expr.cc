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
#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/group_partitioner.h"

namespace cinn::adt {

namespace {

using AnchorTensor = eqaution::Variable;
using FakeOpPlaceHolders = List<equation::FakeOpPlaceHolder>;

std::vector<std::uint64_t> MakeTensorRanks(const List<m_expr::Arg>& arg_lists) {
  std::vector<std::uint64_t> ret;

  for (const auto& arg : *arg_lists) {
    CHECK(arg.Has<adapter::Tensor>());
    ret.push_back(arg.Get<adapter::Tensor>().GetRank());
  }

  return ret;
}

void GenerateOpEquations(const m_expr::OpStmt& op_stmt,
                         equation::config::Context* ctx) {
  const auto& [op, inputs, outputs] = op_stmt;
  CHECK(op.Has<const hlir::framework::Node*>());
  const hlir::framework::Node* op_node = op.Get<const hlir::framework::Node*>();

  using GenerateEquationFunc =
      std::function<void(equation::config::Context * ctx)>;

  const auto& generate_equations =
      Operator::GetAttrs<GenerateEquationFunc>("generate_equations");
  const auto& iter = generate_equations.find(op_node->op());
  CHECK(iter != generate_equations.end());
  iter->second(ctx);
}

std::shared_ptr<equation::config::Context> MakeContextAndGenerateEquations(
    const m_expr::OpStmt& op_stmt) {
  const auto& [op, inputs, outputs] = op_stmt;
  const auto& ctx = std::make_shared<equation::config::Context>(
      MakeTensorRanks(inputs.value()), MakeTensorRanks(outputs.value()));

  GenerateOpEquations(op_stmt, ctx.get());

  return ctx;
}

std::function<std::shared_ptr<equation::config::Context>(const m_expr::OpStmt&)>
GenerateContext4LocalOpStmt(const List<m_expr::OpStmt>& op_stmts) {
  using OpStmt2EquationContext =
      std::unordered_map<m_expr::OpStmt,
                         std::shared_ptr<equation::config::Context>>;
  const auto& op_stmt2equation_ctx = std::make_shared<OpStmt2EquationContext>();

  for (const auto& op_stmt : op_stmts) {
    const auto& ctx = MakeContextAndGenerateEquations(op_stmt);
    CHECK(op_stmt2equation_ctx->emplace(op_stmt, ctx).second);
  }

  return [op_stmt2equation_ctx](const auto& op_stmt) {
    return op_stmt2equation_ctx->at(op_stmt);
  };
}

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
void VisitEachOpStmt(const cinn::hlir::framework::Graph::Group& group) {
  for (const auto* op : group.nodes) {
    // Tuple<Op, In<List<Arg>>, Out<List<Arg>>>
    DoEachT(m_expr::OpStmt{MakeOp(op),
                           MakeOpStmtInputList(op, group.graph_),
                           MakeOpStmtOutputList(op, group.graph_)});
  }
}

List<m_expr::OpStmt> MakeOpStmts(
    const cinn::hlir::framework::Graph::Group& group) {
  List<m_expr::OpStmt> ret{};

  VisitEachOpStmt(group,
                  [&](const auto& op_stmt) { ret->emplace_back(op_stmt); });

  return ret;
}

template <typename DoEachT>
void PartitionIGroupOpStmts(const List<m_expr::OpStmt>& op_stmts,
                            const DoEachT& DoEach) {
  const auto& EquationCtx4OpStmt = GenerateContext4LocalOpStmt(op_stmts);
  const auto& igroup_specs =
      partition::PartitionOpStmts(EquationCtx4OpStmt, op_stmts);
  for (const auto& igroup_spec : igroup_specs) {
    DoEach(igroup_spec);
  }
}

std::shared_ptr<IGroup> MakeIGroup(
    const AnchorIndex& anchor_index,
    const List<m_expr::OpStmt>& igroup_op_stmts,
    const std::function<std::shared_ptr<equation::config::Context>(
        const m_expr::OpStmt&)>& EquationCtx4OpStmt) {
  ADT_TODO();  // Non-Trivial
}

std::vector<std::shared_ptr<IGroup>> GenerateIGroups(
    const cinn::hlir::framework::Graph::Group& group) {
  std::vector<std::shared_ptr<IGroup>> ret{};

  List<m_expr::OpStmt> op_stmts = MakeOpStmts(group);

  PartitionIGroupOpStmts(op_stmts, [&](const auto& igroup_spec) {
    ret.push_back(MakeIGroup(igroup_spec.anchor_index,
                             igroup_spec.igroup_op_stmts,
                             igroup_spec.EquationCtx4OpStmt));
  });

  return ret;
}

std::shared_ptr<KGroup> GenerateKGroups(
    const cinn::hlir::framework::Graph::Group& group,
    const std::vector<std::shared_ptr<IGroup>>& igroups) {
  // @Yifan
  ADT_TODO();  // Trival code
}

std::pair<ScheduleIterators, equation::GraphView>
MakeSdIteratorsAndEquationGraphView(const std::shared_ptr<IGroup>& igroup,
                                    const ScheduleDescriptor& sd) {
  ADT_TODO();
}

template <typename DoEachT>
cinn::adt::m_expr::MapExpr MergeMapExpr(const std::shared_ptr<KGroup>& kgroup,
                                        const DoEachT& DoEach) {
  // @Yifan
  ADT_TODO();  // Trival code temporarily, consider multiple igroups later
}

using TensorIndex = Variable;
using TensorIndexExpr = Value;

std::functoin<const TensorIndexExpr&(const m_expr::Tensor&)>
MakeGetterTensorIndexExpr(const std::shared_ptr<IGroup>& igroup,
                          const eqaution::GraphView& sd_equation_graph_view) {
  equation::GraphView igroup_view = igroup->GetDefaultGraphView();
  equation::GraphView merged_view = igroup_view.Merge(sd_equation_graph_view);
  auto ctx = std::make_shared<equation::IndexExprInferContext>();
  eqaution::value::SolveEquations(
      merged_view, igroup->anchor_tensor(), ctx.get());
  return [ctx](const m_expr::Tensor& tensor) {
    const auto index = ctx->GetIndex(tensor);
    return ctx->GetValue(index);
  }
}

std::function<const m_expr::SchedulePolicy&(const equation::IterVar&)>
MakeGetterSchedulePolicy(const m_expr::ScheduleIterators& sd_iters,
                         const m_expr::ScheduleDescriptor& sd) {
  ADT_TODO();
}

template <typename DoEachT>
void VisitEachStmt(const m_ir::MapIRList& map_irs, const DoEachT& DoEach) {
  ADT_TODO();
}

m_expr::OpStmt MakeOpStmt(const m_ir::MapIR& map_ir,
                          std::size_t outter_layer_sd_size) {
  CHECK_EQ(map_ir.op_stmts().size(), 1);
  ADT_TODO();
}

ScheduleDescriptor MakeInnerScheduleDescriptor(
    const m_ir::MapIRList& map_irs, std::size_t outter_layer_sd_size) {
  ADT_TODO();
}

template <typename DoEachT>
void VisitEachOpStmt(const m_ir::MapIR& map_ir, const DoEachT& DoEach) {
  ADT_TODO();
}

List<m_expr::Stmt> MakeInnerLayerStmts(const std::shared_ptr<IGroup>& igroup,
                                       const m_ir::MapIR& map_ir) {
  List<m_expr::Stmt> ret;

  VisitEachOpStmt(map_ir,
                  [&](const auto& op_stmt) { ret->emplace_back(op_stmt); });

  return ret;
}

m_expr::MapStmt<m_expr::Stmt> MakeInnerLayerMapStmt(
    const std::shared_ptr<IGroup>& igroup,
    const m_ir::MapIR& map_ir,
    std::size_t outter_layer_sd_size) {
  const auto& inner_schedule_descriptor =
      MakeInnerScheduleDescriptor(map_irs, outter_layer_sd_size);

  return {inner_schedule_descriptor, MakeInnerLayerStmts(igroup, map_ir)};
}

m_expr::Stmt MakeOutterLayerStmt(const std::shared_ptr<IGroup>& igroup,
                                 const m_ir::MapIR& map_ir,
                                 std::size_t outter_layer_sd_size) {
  if (map_ir.op_stmts().size() == 1) {
    return MakeOpStmt(map_ir, outter_layer_sd_size);
  } else if (map_ir.op_stmts().size() > 1) {
    return MakeInnerLayerMapStmt(igroup, map_ir, outter_layer_sd_size);
  } else {
    LOG(FATAL) << "Not Supported";
  }
}

List<m_expr::Stmt> MakeOutterLayerStmts(const std::shared_ptr<IGroup>& igroup,
                                        const m_ir::MapIRList& map_irs,
                                        std::size_t outter_layer_sd_size) {
  List<m_expr::Stmt> ret;

  VisitEachStmt(map_irs, [&](const auto& map_ir) {
    ret->emplace_back(
        MakeOutterLayerStmt(igroup, map_ir, outter_layer_sd_size));
  });

  return ret;
}

ScheduleDescriptor MakeOutterScheduleDescriptor(
    const m_ir::MapIRList& map_irs) {
  ADT_TODO();
}

m_expr::MapStmt<m_expr::Stmt> MakeMapStmt(const std::shared_ptr<IGroup>& igroup,
                                          const m_ir::MapIRList& map_irs) {
  const auto& outter_schedule_descriptor =
      MakeOutterScheduleDescriptor(map_irs);

  return {outter_schedule_descriptor,
          MakeOutterLayerStmts(
              igroup, map_irs, outter_schedule_descriptor->size())};
}

m_expr::Tensor GetAnchorTensor(const std::shared_ptr<IGroup>& igroup) {
  ADT_TODO();
}

m_expr::AnchoredMapStmt MakeAnchoredMapStmt(
    const std::shared_ptr<IGroup>& igroup, const m_ir::MapIRList& map_irs) {
  return {MakeMapStmt(igroup, map_irs), GetAnchorTensor(igroup)};
}

List<m_expr::AnchoredMapStmt> MakeAnchoredMapStmts(
    const std::shared_ptr<IGroup>& igroup, const m_ir::MapIRList& map_irs) {
  return {MakeAnchoredMapStmt(igroup, map_irs)};
}

List<m_expr::Tensor> MakeInputTensors(const std::shared_ptr<IGroup>& igroup,
                                      const m_ir::MapIRList& map_irs) {
  ADT_TODO();
}

List<m_expr::Tensor> MakeOutputTensors(const std::shared_ptr<IGroup>& igroup,
                                       const m_ir::MapIRList& map_irs) {
  ADT_TODO();
}

m_expr::Kernel MakeKernel(const std::shared_ptr<IGroup>& igroup,
                          const m_ir::MapIRList& map_irs) {
  return {MakeAnchoredMapStmts(igroup, map_irs),
          MakeInputTensors(igroup, map_irs),
          MakeOutputTensors(igroup, map_irs)};
}

m_expr::MapExpr MakeMapExpr(
    const std::shared_ptr<IGroup>& igroup,
    const m_ir::MapIRList& map_irs,
    const std::functoin<const TensorIndexExpr&(const m_expr::Tensor&)>&
        GetTensorIndexes) {
  return {MakeKernel(igroup, map_irs), GetTensorIndexes};
}

m_expr::AnchoredMapStmt GenerateAnchoredMapStmt(
    const std::shared_ptr<IGroup>& igroup,
    const m_expr::ScheduleIterators& sd_iters,
    const m_expr::ScheduleDescriptor& sd,
    const std::functoin<const TensorIndexExpr&(const m_expr::Tensor&)>&
        GetTensorIndexes) {
  const auto& GetSchedulePolicy = MakeGetterSchedulePolicy(sd_iters, sd);

  const auto& map_irs = m_ir::GenerateClusterOpsForLoopFuse(
      igroup->op_stmts(), sd_iters, GetSchedulePolicy, GetTensorIndexes);

  return MakeMapExpr(igroup, map_irs, GetTensorIndexes);
}

m_expr::MapExpr GenerateMapExpr(const std::shared_ptr<IGroup>& igroup,
                                const m_expr::ScheduleDescriptor& sd) {
  const auto& [schedule_iters, sd_equation_graph_view] =
      MakeSdIteratorsAndEquationGraphView(igroup, sd);

  const auto& get_tensor_expr =
      MakeGetterTensorIndexExpr(igroup, sd_equation_graph_view);

  return GenerateAnchoredMapStmt(igroup, schedule_iters, sd, get_tensor_expr);
}

m_expr::MapExpr GenerateMapExpr(const std::shared_ptr<KGroup>& kgroup) {
  return MergeMapExpr(kgroup, [&](const std::shared_ptr<IGroup>& igroup) {
    auto sd = kgroup.GetDefaultScheduleDescriptor(igroup);
    return GenerateMapExpr(igroup, sd);
  });
}

}  // namespace

m_expr::MapExpr GenerateMapExpr(
    const cinn::hlir::framework::Graph::Group& group) {
  auto igroups = GenerateIGroups(group);

  auto kgroup = GenerateKGroups(group, igroups);

  return GenerateMapExpr(kgroup);
}

}  // namespace cinn::adt
