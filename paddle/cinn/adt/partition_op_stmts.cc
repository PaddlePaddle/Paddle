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

#include "paddle/cinn/adt/partition_op_stmts.h"

#include <algorithm>

namespace cinn::adt::partition {

using TensorIndex = Variable;

AnchorIndex PickThenEraseAnchorIndex(
    std::unordered_set<AnchorIndex>* candidate_anchor_indexes) {
  // Heuristic optimization will be added later
  // such as choosing the one with the biggest rank number as the anchor tensor
  // first
  const auto& ret = *candidate_anchor_indexes->begin();
  candidate_anchor_indexes->erase(candidate_anchor_indexes->begin());
  return ret;
}

std::vector<AnchorIndex> InitCandidateAnchorIndex(
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const List<m_expr::OpStmt>& op_stmts) {
  std::vector<AnchorIndex> ret{};
  for (const auto& op_stmt : *op_stmts) {
    const auto* equation_ctx = EquationCtx4OpStmt(op_stmt);
    eqaution_ctx->VisitEachTensorIndex(
        [&](const auto& tensor_index) { ret.push_back(tensor_index); });
  }
  return ret;
}

std::function<const m_expr::OpStmt&(const equation::FakeOpPlaceHolder)>
MakeGetterOpStmt4OpPlaceHolder(const EquationCtx4OpStmtT& EquationCtx4OpStmt,
                               const List<m_expr::OpStmt>& op_stmts) {
  ADT_TODO();  // Trivial Code
}

List<m_expr::OpStmt> FindVisitedOpStmts(
    const AnchorIndex& anchor_index,
    const equation::GraphView& equation_graph,
    const std::function<const m_expr::OpStmt&(
        const equation::FakeOpPlaceHolder)>& OpStmt4OpPlaceHolder) {
  ADT_TODO();  // Trivial Code
}

equation::GraphView MakeGlobalEquationGraphForPartition(
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const List<m_expr::OpStmt>& op_stmts) {
  ADT_TODO();
}

template <typename DoEachT>
void VisitTensorIndex(const IGroupSpec& igroup_spec, const DoEachT& DoEach) {
  const auto& igroup_op_stmts = igroup_spec.igroup_op_stmts;
  const auto& EquationCtx4OpStmt = igroup_spec.EquationCtx4OpStmt;
  for (const auto& igroup_op_stmt : *igroup_op_stmts) {
    const auto* ctx = EquationCtx4OpStmt(igroup_op_stmt);
    ctx->VisitEachTensorIndex(DoEach);
  }
}

void CleanSmallIGroupSpecs(
    const IGroupSpec& igroup_spec,
    std::unordered_map<AnchorIndex, IGroupSpec>* anchor_index2igroup_spec) {
  VisitTensorIndex(igroup_spec, [&](const auto& tensor_index) {
    anchor_index2igroup_spec->erase(tensor_index);
  });
}

void UpdataAnchorIndex2IGroupSpec(
    const IGroupSpec& igroup_spec,
    std::unordered_map<AnchorIndex, IGroupSpec>* anchor_index2igroup_spec) {
  CleanSmallIGroupSpecs(igroup_spec, anchor_index2igroup_spec);

  CHECK(anchor_index2igroup_spec->emplace(igroup_spec.anchor_index, igroup_spec)
            .second);
}

void EraseCandidateAnchorIndexes(
    const IGroupSpec& igroup_spec,
    std::unordered_set<AnchorIndex>* candidate_anchor_indexes) {
  VisitTensorIndex(igroup_spec, [&](const auto& tensor_index) {
    candidate_anchor_indexes->erase(tensor_index);
  });
}

std::unordered_map<AnchorIndex, IGroupSpec> PartitionOpStmtsIntoIGroupSpecs(
    std::unordered_set<AnchorIndex>* candidate_anchor_indexes,
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const List<m_expr::OpStmt>& op_stmts) {
  std::unordered_map<AnchorIndex, IGroupSpec> anchor_index2igroup_spec;

  const auto& OpStmt4OpPlaceHolder =
      MakeGetterOpStmt4OpPlaceHolder(EquationCtx4OpStmt, op_stmts);

  const auto& equation_graph =
      MakeGlobalEquationGraphForPartition(EquationCtx4OpStmt, op_stmts);

  while (!candidate_anchor_indexes->empty()) {
    AnchorIndex anchor_tensor =
        PickThenEraseAnchorIndex(candidate_anchor_indexes);

    const auto& visited_op_stmts =
        FindVisitedOpStmts(anchor_tensor, equation_graph, OpStmt4OpPlaceHolder);
    CHECK(!visited_op_stmts->empty());

    IGroupSpec igroup_spec{anchor_tensor, visited_op_stmts, EquationCtx4OpStmt};
    UpdataAnchorIndex2IGroupSpec(igroup_spec, &anchor_index2igroup_spec);
    EraseCandidateAnchorIndexes(igroup_spec, candidate_anchor_indexes);
  }

  return anchor_index2igroup_spec;
}

std::vector<IGroupSpec> SortedIGroupSpecs(
    const std::unordered_map<AnchorIndex, IGroupSpec>& anchor_index2igroup_spec,
    const List<m_expr::OpStmt>& op_stmts) {
  ADT_TODO();
}

std::vector<IGroupSpec> PartitionOpStmts(
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const List<m_expr::OpStmt>& op_stmts) {
  std::unordered_set<AnchorIndex> candidate_anchor_indexes =
      InitCandidateAnchorIndex(EquationCtx4OpStmt, op_stmts);

  std::unordered_map<AnchorIndex, IGroupSpec> anchor_index2igroup_spec =
      PartitionOpStmtsIntoIGroupSpecs(
          &candidate_anchor_indexes, EquationCtx4OpStmt, op_stmts);

  return SortedIGroupSpecs(anchor_index2igroup_spec, op_stmts);
}

}  // namespace cinn::adt::partition
