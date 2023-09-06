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

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/equation_graph.h"
#include "paddle/cinn/hlir/framework/graph.h"

namespace cinn::adt::equation {

using FakeOpPlaceHolders = List<FakeOpPlaceHolder>;

std::vector<Variable> InitCandidateIndex(const Graph& graph);

Variable PickAnchorTensor(const std::vector<Variable>& candidate_index);

FakeOpPlaceHolders GenerateIGroup(const Graph& graph,
                                  const Variable& anchor_tensor);

bool IsContain(const FakeOpPlaceHolders& pre_igroup,
               const FakeOpPlaceHolders& igroup);

void UpdateIGroupMap(
    const FakeOpPlaceHolders& igroup,
    const Variable& anchor_tensor,
    std::unordered_map<Variable, FakeOpPlaceHolders>* index2IGroup);

void UpdateCandidateSet(const Graph& graph,
                        const FakeOpPlaceHolders& igroup,
                        const Variable& anchor_tensor,
                        std::vector<Variable>* candidate_index);

void TopoSort4IGroup(
    const cinn::hlir::framework::Graph::Group& group,
    std::unordered_map<Variable, FakeOpPlaceHolders>* index2IGroup);

std::unordered_map<Variable, FakeOpPlaceHolders> PartitionGraph(
    const cinn::hlir::framework::Graph::Group& group, const Graph& graph);

}  // namespace cinn::adt::equation
