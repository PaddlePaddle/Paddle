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

TensorIndexGenerator: ScheduleLambda <- (EquationAnchorIndex, EquationIGroupOps,
tScheduleIterVar Name, ScheduleSize)

1. 以 EquationAnchorIndex 为起点，遍历整个 EquationGraph，得到从
EquationAnchorIndex 到 EquationIGroupOps 内每个 TensorIndex 的转换表达式
2. 建立 (tScheduleIterVar Name, ScheduleSize) 和 EquationAnchorIndex
之间的转换表达式 Schedule2AnchorIndex
3. 将 Schedule2AnchorIndex 代入 EquationIGroupOps 内每个 TensorIndex
的转换表达式，得到 ScheduleLambda

ADT 定义：
ScheduleLambda = TensorFlattenIndexLambda <- [tScheduleIterVar Name,
ScheduleSize] TensorFlattenIndexLambda = (tScheduleIterVar Name, tEach
TensorFlattenIndexExpr)

EquationAnchorIndex = tAnchor Index
EquationIGroupOps = tGroup [FakeOpPlaceHolder]

*/

TensorFlattenIndexLambda TensorIndexGenerator(
    EquationAnchorIndex anchor_index,
    EquationLeafIndex leaf_index,
    EquationSubGraph sub_graph,
    std::vector<tScheduleIterVar<Iterator>> sched_iter,
    std::vector<SchdeduleSize> sched_size) {
  // tSchedule Index -> tLeaf IndexExpr

  // tAnchorTensor Index -> [tLeaf IndexExpr]
  std::vector<TensorFlattenIndexExpr> tensor_exprs =
      GenerateTensorFlattenExprs(graph, anchor_index, igroup);

  // tScheduleDescriptor Index <-> tAnchorTensor Index
  Equations schedule2index =
      Schedule2AnchorIndex(anchor_index, sched_iter, sched_size);

  std::vector<ScheduleLambda> schedule_lambdas =
      SubstituteTensorExprs(&tensor_exprs, schedule2index);
}

}  // namespace cinn::adt::equation
