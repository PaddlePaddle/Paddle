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
  ADT_TODO();  // Trival code
}

eqaution::Graph BuildSdSubGraph(const AnchorTensor& anchor_tensor,
                                const m_expr::ScheduleIterators& sd) {
  ADT_TODO();  // Trival code
}

equation::GraphView MakeSdEquationGraphView(
    const std::shared_ptr<IGroup>& igroup, const ScheduleIterators& sd_iters) {
  const eqaution::Graph& sub_graph =
      BuildSdSubGraph(igroup->anchor_tensor(), sd_iters);
  return sub_graph.GetWalker();
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

cinn::adt::m_expr::MapExpr GenerateMapExpr(
    const std::shared_ptr<IGroup>& igroup,
    const m_expr::ScheduleDescriptor& sd,
    const std::functoin<const TensorIndexExpr&(const m_expr::Tensor&)>&
        tensor_indexes) {
  // Non-trival here
  ADT_TODO();
}

cinn::adt::m_expr::MapExpr GenerateMapExpr(
    const std::shared_ptr<IGroup>& igroup,
    const m_expr::ScheduleDescriptor& sd) {
  auto sd_equation_graph_view = MakeSdEquationGraphView(igroup, sd);

  auto get_tensor_expr =
      MakeGetterTensorIndexExpr(igroup, sd_equation_graph_view);

  return GenerateMapExpr(igroup, sd, get_tensor_expr);
}

cinn::adt::m_expr::MapExpr GenerateMapExpr(
    const std::shared_ptr<KGroup>& kgroup) {
  return MergeMapExpr(kgroup, [&](const std::shared_ptr<IGroup>& igroup) {
    auto sd = kgroup.GetDefaultScheduleDescriptor(igroup);
    return GenerateMapExpr(igroup, sd);
  });
}

}  // namespace

cinn::adt::m_expr::MapExpr GenerateMapExpr(
    const cinn::hlir::framework::Graph::Group& group) {
  auto igroups = GenerateIGroups(group);

  auto kgroup = GenerateKGroups(group, igroups);

  return GenerateMapExpr(kgroup);
}

}  // namespace cinn::adt
