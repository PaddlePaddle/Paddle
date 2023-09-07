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

namespace cinn::adt {

namespace {

using AnchorTensor = eqaution::Variable;
using FakeOpPlaceHolders = List<equation::FakeOpPlaceHolder>;

std::shared_ptr<equation::Graph> GenerateEquationGraph(
    const cinn::hlir::framework::Graph::Group& group) {
  ADT_TODO();  // Trival code
}

std::unordered_map<AnchorTensor, FakeOpPlaceHolders> PartitionIGroups(
    const cinn::hlir::framework::Graph::Group& group,
    const equation::Graph& equations) {
  // Yifan
  ADT_TODO();  // Trival code
}

template <typename DoEachT>
std::vector<std::shared_ptr<IGroup>> AssembleIGroups(
    const cinn::hlir::framework::Graph::Group& group,
    const std::unordered_map<AnchorTensor, FakeOpPlaceHolders>&
        anchor_tensor2_igroup,
    const DoEachT& DoEach) {
  ADT_TODO();  // Trival code
}

List<m_expr::OpStmt> GenerateMapIROpStmts(
    const cinn::hlir::framework::Graph::Group& group,
    const std::shared_ptr<equation::Graph>& equation_graph,
    const AnchorTensor& anchor_tensor,
    const FakeOpPlaceHolders& fake_op_placeholders) {
  ADT_TODO();  // Trival code
}

equation::Equations GenerateMapIREquations(
    const cinn::hlir::framework::Graph::Group& group,
    const std::shared_ptr<equation::Graph>& equation_graph,
    const AnchorTensor& anchor_tensor,
    const FakeOpPlaceHolders& fake_op_placeholders) {
  ADT_TODO();  // Trival code
}

std::vector<std::shared_ptr<IGroup>> GenerateIGroups(
    const cinn::hlir::framework::Graph::Group& group,
    const std::shared_ptr<equation::Graph>& equation_graph,
    const std::unordered_map<AnchorTensor, FakeOpPlaceHolders>&
        anchor_tensor2_igroup) {
  return AssembleIGroups(
      group,
      anchor_tensor2_igroup,
      [&](const auto& anchor_tensor, const auto& fake_op_placeholders) {
        return std::pair{
            GenerateMapIROpStmts(
                group, equation_graph, anchor_tensor, fake_op_placeholders),
            GenerateMapIREquations(
                group, equation_graph, anchor_tensor, fake_op_placeholders)};
      });
}

std::vector<std::shared_ptr<IGroup>> GenerateIGroups(
    const cinn::hlir::framework::Graph::Group& group) {
  auto equation_graph = GenerateEquationGraph(group);

  auto anchor_tensor2_igroup = PartitionIGroups(group, equation_graph);

  return GenerateIGroups(group, equation_graph, anchor_tensor2_igroup);
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

m_expr::MapExpr GenerateMapExpr(
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

  return GenerateMapExpr(igroup, schedule_iters, sd, get_tensor_expr);
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
