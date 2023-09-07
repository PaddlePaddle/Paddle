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

Variable PickAnchorTensor(const std::vector<Variable>& candidate_index) {
  // Heuristic optimization will be added later
  // such as choosing the one with the biggest rank number as the anchor tensor
  // first
  return *(candidate_index.begin());
}

FakeOpPlaceHolders GenerateIGroup(const Graph& graph,
                                  const Variable& anchor_tensor) {
  FakeOpPlaceHolders igroup;
  EquationGraphTopoWalker<const Variable, const Function*> walker =
      graph.GetWalker();
  std::function<void(Variable)> variableVisitor =
      [&](const Variable& variable) {
        variable >> match{[&](const FakeOpPlaceHolder& fakeOpPlaceholder) {
          igroup->emplace_back(fakeOpPlaceholder);
        }};
      };
  walker(anchor_tensor, variableVisitor);
  return igroup;
}

bool IsContain(const FakeOpPlaceHolders& pre_igroup,
               const FakeOpPlaceHolders& igroup) {
  for (const auto& pre_op : *pre_igroup) {
    auto iter = std::find(igroup->begin(), igroup->end(), pre_op);
    if (iter == igroup->end()) {
      return false;
    }
  }
  return true;
}

void UpdateIGroupMap(
    const FakeOpPlaceHolders& igroup,
    const Variable& anchor_tensor,
    std::unordered_map<Variable, FakeOpPlaceHolders>* index2IGroup) {
  for (const auto& [pre_anchor_tensor, pre_igroup] : *index2IGroup) {
    if (pre_igroup->size() >= igroup->size()) {
      continue;
    }
    if (IsContain(pre_igroup, igroup)) {
      index2IGroup->erase(pre_anchor_tensor);
    }
  }
  index2IGroup->emplace(anchor_tensor, igroup);
}

void UpdateCandidateSet(const Graph& graph,
                        const FakeOpPlaceHolders& igroup,
                        const Variable& anchor_tensor,
                        std::vector<Variable>* candidate_index) {
  EquationGraphTopoWalker<const Variable, const Function*> walker =
      graph.GetWalker();

  std::function<void(Variable)> variableVisitor =
      [&](const Variable& variable) {
        variable >> match{[&](const Index& index) {
          auto iter = std::find(candidate_index->begin(),
                                candidate_index->end(),
                                Variable(index));
          if (iter != candidate_index->end()) {
            candidate_index->erase(iter);
          }
        }};
      };
  walker(anchor_tensor, variableVisitor);
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

std::unordered_map<AnchorIndex, IGroupSpec> PartitionOpStmtsIntoIGroupSpecs(
    std::vector<AnchorIndex>* candidate_anchor_index,
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const List<m_expr::OpStmt>& op_stmts) {
  ADT_TODO();
  while (!candidate_anchor_index.empty()) {
    Variable anchor_tensor = PickAnchorTensor(candidate_anchor_index);
    FakeOpPlaceHolders igroup = GenerateIGroup(graph, anchor_tensor);
    UpdateIGroupMap(igroup, anchor_tensor, &index2igroup);
    UpdateCandidateSet(graph, igroup, anchor_tensor, &candidate_anchor_index);
  }
}

std::vector<IGroupSpec> SortedIGroupSpecs(
    const std::unordered_map<AnchorIndex, IGroupSpec>& anchor_index2igroup_spec,
    const List<m_expr::OpStmt>& op_stmts) {
  ADT_TODO();
}

std::vector<IGroupSpec> PartitionOpStmts(
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const List<m_expr::OpStmt>& op_stmts) {
  std::vector<AnchorIndex> candidate_anchor_index =
      InitCandidateAnchorIndex(EquationCtx4OpStmt, op_stmts);

  std::unordered_map<AnchorIndex, IGroupSpec> anchor_index2igroup_spec =
      PartitionOpStmtsIntoIGroupSpecs(
          &candidate_anchor_index, EquationCtx4OpStmt, op_stmts);

  return SortedIGroupSpecs(anchor_index2igroup_spec, op_stmts);
}

}  // namespace cinn::adt::partition
