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

#pragma once

#include "paddle/cinn/adt/equation_graph.h"

namespace cinn::adt::equation {

using EquationIGroupOps = std::unordered_set<FakeOpPlaceHolder>;

std::unordered_set<Variable> InitCandidateIndex(const Graph& graph) {
  std::unordered_set<Variable> variables = graph.GetVariables();
  std::unordered_set<Variable> candidate_index;
  for (auto iter = variables.begin(); iter != variables.end(); ++iter) {
    *iter >> match{[&](const Index& index) {
      candidate_index.emplace(Variable(index));
    }};
  }
  return candidate_index;
}

Variable PickAnchorTensor(const std::unordered_set<Variable>& candidate_index) {
  // Heuristic optimization will be added later
  // such as choosing the one with the biggest rank number as the anchor tensor
  // first
  return *(candidate_index.begin());
}

EquationIGroupOps GenerateIGroup(const Graph& graph,
                                 const Variable& anchor_tensor) {
  EquationIGroupOps igroup;
  EquationGraphTopoWalker<const Variable, const Function*> walker =
      graph.GetWalker();
  std::function<void(Variable)> variableVisitor =
      [&](const Variable& variable) {
        variable >> match{[&](const FakeOpPlaceHolder& fakeOpPlaceholder) {
          igroup.emplace(fakeOpPlaceholder);
        }};
      };
  walker(anchor_tensor, variableVisitor);
  return igroup;
}

bool IsContain(const EquationIGroupOps& pre_igroup,
               const EquationIGroupOps& igroup) {
  for (auto iter = pre_igroup.begin(); iter != pre_igroup.end(); ++iter) {
    if (igroup.find(*iter) == igroup.end()) {
      return false;
    }
  }
  return true;
}

void UpdateIGroupMap(
    const EquationIGroupOps& igroup,
    const Variable& anchor_tensor,
    std::unordered_map<Index, EquationIGroupOps>* index2IGroup) {
  for (auto iter = index2IGroup->begin(); iter != index2IGroup->end(); ++iter) {
    if (iter->second.size() >= igroup.size()) {
      continue;
    }
    if (IsContain(iter->second, igroup)) {
      index2IGroup->erase(iter);
    }
  }
  index2IGroup->emplace(anchor_tensor, igroup);
}

void UpdateCandidateSet(const Graph& graph,
                        const EquationIGroupOps& igroup,
                        const Variable& anchor_tensor,
                        std::unordered_set<Variable>* candidate_index) {
  EquationGraphTopoWalker<const Variable, const Function*> walker =
      graph.GetWalker();
  std::function<void(Variable)> variableVisitor =
      [&](const Variable& variable) {
        variable >> match{[&](const Index& index) {
          if (candidate_index->find(Variable(index)) !=
              candidate_index->end()) {
            candidate_index->erase(Variable(index));
          }
        }};
      };
  walker(anchor_tensor, variableVisitor);
}

std::unordered_map<Index, EquationIGroupOps> PartitionGraph(
    const Graph& graph) {
  std::unordered_set<Variable> candidate_index = InitCandidateIndex(graph);
  std::unordered_map<Index, EquationIGroupOps> index2IGroup;
  while (!candidate_index.empty()) {
    Variable anchor_tensor = PickAnchorTensor(candidate_index);
    EquationIGroupOps igroup = GenerateIGroup(graph, anchor_tensor);
    UpdateIGroupMap(igroup, anchor_tensor, &index2IGroup);
    UpdateCandidateSet(graph, igroup, anchor_tensor, &candidate_index);
  }
  return index2IGroup;
}

}  // namespace cinn::adt::equation
