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

#include <algorithm>

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/equation_graph.h"

namespace cinn::adt::equation {

using FakeOpPlaceHolders = List<FakeOpPlaceHolder>;

std::vector<Variable> InitCandidateIndex(const Graph& graph) {
  std::unordered_set<Variable> variables = graph.GetVariables();
  std::vector<Variable> candidate_index;
  for (auto iter = variables.begin(); iter != variables.end(); ++iter) {
    *iter >> match{[&](const Index& index) {
      candidate_index.emplace_back(Variable(index));
    }};
  }
  return candidate_index;
}

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
  for (auto iter = pre_igroup->begin(); iter != pre_igroup->end(); ++iter) {
    if (find(igroup->begin(), igroup->end(), *iter) == igroup->end()) {
      return false;
    }
  }
  return true;
}

void UpdateIGroupMap(
    const FakeOpPlaceHolders& igroup,
    const Variable& anchor_tensor,
    std::unordered_map<Variable, FakeOpPlaceHolders>* index2IGroup) {
  for (auto iter = index2IGroup->begin(); iter != index2IGroup->end(); ++iter) {
    if (iter->second->size() >= igroup->size()) {
      continue;
    }
    if (IsContain(iter->second, igroup)) {
      index2IGroup->erase(iter);
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
          auto iter = find(candidate_index->begin(),
                           candidate_index->end(),
                           Variable(index));
          if (iter != candidate_index->end()) {
            candidate_index->erase(iter);
          }
        }};
      };
  walker(anchor_tensor, variableVisitor);
}

void TopoSort4IGroup(
    const cinn::hlir::framework::Graph::Group& group,
    std::unordered_map<Variable, FakeOpPlaceHolders>* index2IGroup) {
  std::vector<cinn::hlir::framework::Node*> sorted_ops = group.nodes;
  for (auto iter_index2IGroup = index2IGroup->begin();
       iter_index2IGroup != index2IGroup->end();
       ++iter_index2IGroup) {
    FakeOpPlaceHolders tmp_op_placeholder;
    for (auto iter_sorted_ops = sorted_ops.begin();
         iter_sorted_ops != sorted_ops.end();
         ++iter_sorted_ops) {
      if (find(iter_index2IGroup->second->begin(),
               iter_index2IGroup->second->end(),
               *iter_sorted_ops) != iter_index2IGroup->second->end()) {
        tmp_op_placeholder->emplace_back(*iter_sorted_ops);
      }
    }
    iter_index2IGroup->second = tmp_op_placeholder;
  }
}

std::unordered_map<Variable, FakeOpPlaceHolders> PartitionGraph(
    const cinn::hlir::framework::Graph::Group& group, const Graph& graph) {
  std::vector<Variable> candidate_index = InitCandidateIndex(graph);
  std::unordered_map<Variable, FakeOpPlaceHolders> index2IGroup;
  while (!candidate_index.empty()) {
    Variable anchor_tensor = PickAnchorTensor(candidate_index);
    FakeOpPlaceHolders igroup = GenerateIGroup(graph, anchor_tensor);
    UpdateIGroupMap(igroup, anchor_tensor, &index2IGroup);
    UpdateCandidateSet(graph, igroup, anchor_tensor, &candidate_index);
  }
  TopoSort4IGroup(group, &index2IGroup);
  return index2IGroup;
}

}  // namespace cinn::adt::equation
