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
  // TODO(yifan)

  std::unordered_set<Variable> candidate_index = InitCandidateIndex(graph);
  std::unordered_set<Index> selected_anchor_index;

  IGroupList igroups;

  while (!candidate_index.empty()) {
    Index candidate = Pick(candidate_index);

    EquationIGroupOps igroup = Walk(graph, candidate);
    CleanSelectedSet(igroups, igroup);
    AddToSelectedSet(igroups, igroup);
    CleanCandidateSet(graph, igroup, &candidate_index);
  }
}

}  // namespace cinn::adt::equation
