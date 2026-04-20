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

#include <memory>

#include <queue>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

#ifdef PADDLE_WITH_CINN
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#endif
#include "paddle/pir/include/core/builder.h"

namespace pir {
using OpClassifier = std::function<bool(const Operation&)>;
using GroupOpsVec = std::vector<Operation*>;

std::vector<GroupOpsVec> DetectSubGraphs(Block* block,
                                         const OpClassifier& classifier);

std::vector<Value> AnalysisOutputs(const GroupOpsVec& group_ops,
                                   bool at_least_one_output = true);
void ReplaceWithGroupOp(Block* block,
                        const GroupOpsVec& group_ops,
                        bool at_least_one_output = true);

Operation* FindInsertPoint(const GroupOpsVec& group_ops,
                           const std::vector<Value>& outputs);
void MoveUpstreamOpBeforeGroup(const GroupOpsVec& group_ops,
                               Block* block,
                               Operation* insert_point_op);

}  // namespace pir
