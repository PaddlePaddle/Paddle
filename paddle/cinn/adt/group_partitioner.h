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
  std::unordered_set<Variable> candidate_indexes;
  for (auto iter_val = variables.begin(); iter_val != variables.end();
       ++iter_val) {
    *iter_val >> match{[&](const Index& index) {
      candidate_indexes.insert(Variable(index));
    }};
  }
  return candidate_indexes;
}

Variable Pick(const std::unordered_set<Variable>& candidate_indexes) {
  // Heuristic optimization will be added later
  // such as choosing the one with the biggest rank number as the anchor tensor
  // first
  return *(candidate_indexes.begin());
}

EquationIGroupOps Walk(const Graph& graph, const Variable& candidate_index) {
  EquationIGroupOps igroup;
  EquationGraphTopoWalker<const Variable, const Function*> walker =
      graph.GetWalker();
  std::function<void(Variable)> variableVisitor =
      [&](const Variable& variable) {
        variable >> match{[&](const FakeOpPlaceHolder& fakeOpPlaceholder) {
          igroup.emplace(fakeOpPlaceholder);
        }};
      };
  walker(candidate_index, variableVisitor);
  return igroup;
}

bool isContain(const EquationIGroupOps& pre_igroup,
               const EquationIGroupOps& igroup) {
  for (auto iter_pre_igroup = pre_igroup.begin();
       iter_pre_igroup != pre_igroup.end();
       ++iter_pre_igroup) {
    if (igroup.find(*iter_pre_igroup) == igroup.end()) {
      return false;
    }
  }
  return true;
}

void CleanAndAddSelectedSet(
    const EquationIGroupOps& igroup,
    const Variable& candidate_index,
    std::unordered_map<Index, EquationIGroupOps>* index2Igroup) {
  for (auto iter_index2Igroup = index2Igroup->begin();
       iter_index2Igroup != index2Igroup->end();
       ++iter_index2Igroup) {
    if (iter_index2Igroup->second.size() >= igroup.size()) {
      continue;
    }
    if (isContain(iter_index2Igroup->second, igroup)) {
      index2Igroup->erase(iter_index2Igroup);
    }
  }
  index2Igroup->emplace(candidate_index, igroup);
}

void CleanCandidateSet(const Graph& graph,
                       const EquationIGroupOps& igroup,
                       const Variable& candidate_index,
                       std::unordered_set<Variable>* candidate_indexes) {
  EquationGraphTopoWalker<const Variable, const Function*> walker =
      graph.GetWalker();
  std::function<void(Variable)> variableVisitor =
      [&](const Variable& variable) {
        variable >> match{[&](const Index& index) {
          if (candidate_indexes->find(Variable(index)) !=
              candidate_indexes->end()) {
            candidate_indexes->erase(Variable(index));
          }
        }};
      };
  walker(candidate_index, variableVisitor);
}
/*
PartitionGraph: [EquationAnchorIndex, EquationIGroupOps] <- EquationGraph
EquationAnchorIndex = tAnchor Index
EquationIGroupOps = tGroup [FakeOpPlaceHolder]

1. 对于每一个不重复的候选 tAnchor Variable，执行如下两句：
  1.1 收集该 tAnchor Variable 所遍历到的所有 tVisited Variable
  1.2 拿上述信息更新 tAnchor Variable 候选集，对上述 tVisited Variables 集合去重

一个 tAnchor Variable 对应一串 {tVisited Variable}，
candidate_anchor_tensors 保存候选的 tAnchor Variable
filtered_anchor_tensor 保存过滤出来的 tAnchor Variable
*/

// 1. 返回值的数据结构，需要明确，std::unordered_map<Index,
// std::vector<FakeOpPlaceHolder>>

std::unordered_map<Index, EquationIGroupOps> PartitionGraph(
    const Graph& graph) {
  std::unordered_set<Variable> candidate_indexes = InitCandidateIndex(graph);
  std::unordered_map<Index, EquationIGroupOps> index2Igroup;
  while (!candidate_indexes.empty()) {
    Variable candidate_index = Pick(candidate_indexes);
    EquationIGroupOps igroup = Walk(graph, candidate_index);
    CleanAndAddSelectedSet(igroup, candidate_index, &index2Igroup);
    CleanCandidateSet(graph, igroup, candidate_index, &candidate_indexes);
  }
  return index2Igroup;
}

}  // namespace cinn::adt::equation
