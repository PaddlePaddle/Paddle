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
using IGroupList = std::unordered_set<EquationIGroupOps>;

std::unordered_set<Index> InitCandidateIndex(const Graph& graph){
  //no methods now
  std::unordered_set<Variable> variables_ = graph.getVariables();

  std::unordered_set<Index> candidate_indexes;
  
  auto iter_val = variables_.begin();
  while(iter_val != variables_.end()){
    *iter_val >> match{
      [&](const Index& index){
        candidate_indexes.insert(index);
      }
    };
    iter_val++;
  }
  return candidate_indexes;
}

Index Pick(const std::unordered_set<Index>& candidate_indexes){
    //后续添加启发性优化
    return *candidate_indexes.begin();
}

EquationIGroupOps Walk(const Graph& graph, const Index& chosen_index){
    EquationIGroupOps igroup;
    
    EquationGraphTopoWalker<const Variable, const Function*> walker = graph.GetWalker();
    
    auto variableVisitor = [&](const Variable& variable){
        variable >> match{
            [&](const FakeOpPlaceHolder& fakeOpPlaceholder){
                igroup.insert(fakeOpPlaceholder);
            }
        };
    };
    std::array<Index, 1> arr = {chosen_index};
  
    walker(arr.begin(), arr.end(), variableVisitor);
    return igroup;
}

bool isContain(const EquationIGroupOps& pre_igroup, const EquationIGroupOps& igroup){
  auto iter_pre_igroup = pre_igroup.begin();
  while (iter_pre_igroup != pre_igroup.end()){
    if(igroup.find(*iter_pre_igroup) == igroup.end()){
      return false;
    }
    iter_pre_igroup++;
  }
  return true;
}


void CleanAndAddSelectedSet(IGroupList& pre_igroups, const EquationIGroupOps& igroup){

  auto iter_pre_igroup = pre_igroups.begin();
  while(iter_pre_igroup!= pre_igroups.end()){
    if(iter_pre_igroup->size() >= igroup.size()){
      continue;
    }
    if(isContain(*iter_pre_igroup, igroup)){
      pre_igroups.erase(iter_pre_igroup);
      pre_igroups.insert(igroup);
    }
  }
}

void CleanCandidateSet(const Graph& graph, const EquationIGroupOps& igroup, std::unordered_set<Index>* candidate_indexes, Index& candidate_index){
  EquationGraphTopoWalker<const Variable, const Function*> walker = graph.GetWalker();
  
  auto variableVisitor = [&](const Variable& variable){
    variable >> match{
      [&](const Index& index){
        if(candidate_indexes->find(index) != candidate_indexes->end()){
          candidate_indexes->erase(index);
        }
      }
    };
  };
  std::array<Index, 1> arr = {candidate_index};
  walker(arr.begin(), arr.end(), variableVisitor);
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

std::unordered_map<Index, std::vector<FakeOpPlaceHolder>> PartitionGraph(
    const Graph& graph) {
  std::unordered_set<Index> candidate_index = InitCandidateIndex(graph);
  std::unordered_set<Index> selected_anchor_index;

  IGroupList igroups;

  while (!candidate_index.empty()) {
    Index candidate = Pick(candidate_index);

    EquationIGroupOps igroup = Walk(graph, candidate);
    CleanAndAddSelectedSet(igroups, igroup);
    CleanCandidateSet(graph, igroup, &candidate_index, candidate);
  }
}

}  // namespace cinn::adt::equation
