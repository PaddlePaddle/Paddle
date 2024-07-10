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

#include "paddle/cinn/hlir/framework/pir/op_lowering_group.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/tensor.h"

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {
using OpLoweringGroupPtr = std::shared_ptr<OpLoweringGroup>;

class PrettyNamer;

std::vector<::pir::Value*> GetAllNodeData(::pir::Operation* op);

std::vector<::pir::Operation*> GetConsumers(::pir::Operation* op);

bool IsConstOp(const ::pir::Operation* op);

std::vector<::pir::Operation*> GetConsumersInSet(
    ::pir::Operation* op, const std::unordered_set<::pir::Operation*>& ops);

::pir::Operation* FindGlobalReducer(
    const std::vector<::pir::Operation*>& ops_in_order);

::pir::Operation* FindNearestReducer(
    ::pir::Operation* op, const std::unordered_set<::pir::Operation*>& ops_set);

::pir::Operation* GetMasterToComputeAt(
    ::pir::Operation* op,
    PrettyNamer* pretty_name,
    const std::vector<::pir::Operation*>& ops_in_order,
    const std::unordered_set<::pir::Operation*>& ops_inline,
    const std::unordered_set<::pir::Operation*>& ops_set,
    const std::unordered_map<::pir::Operation*, ::pir::Operation*>&
        virtual_consumers);

std::unordered_set<::pir::Operation*> GetMasters(
    ::pir::Operation* op,
    PrettyNamer* pretty_name,
    const std::unordered_set<::pir::Operation*>& ops_inline,
    const std::unordered_set<::pir::Operation*>& ops_set);

void LoopAssignReduce(
    ir::IRSchedule& ir_sch,  // NOLINT
    ::pir::Operation* op,
    ::pir::Operation* reducer,
    PrettyNamer* pretty_name,
    const Target& target,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map,
    const std::unordered_map<std::string, ir::Tensor>& tmp_tensor_info);

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
