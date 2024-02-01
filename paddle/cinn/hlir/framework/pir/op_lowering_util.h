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

#include "paddle/cinn/hlir/framework/pir/group.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/tensor.h"

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {
using GroupPtr = std::shared_ptr<Group>;

class PrettyNamer;

std::unordered_map<::pir::Operation*, ::pir::Operation*> BuildVirtualConsumer(
    const GroupPtr& group);

std::vector<::pir::Value*> GetAllNodeData(::pir::Operation* op);

std::vector<::pir::Operation*> GetConsumers(::pir::Operation* op);

bool IsConstOp(const ::pir::Operation* op);

std::vector<::pir::Operation*> GetConsumersInSet(
    ::pir::Operation* op, const std::unordered_set<::pir::Operation*>& ops);

std::vector<::pir::Operation*> TopologicalOrder(
    const GroupPtr& group,
    const std::unordered_map<::pir::Operation*, ::pir::Operation*>&
        virtual_consumers);

std::vector<::pir::Operation*> BFSTopologicalOrderWithPriority(
    const GroupPtr& group,
    const std::unordered_map<::pir::Operation*, ::pir::Operation*>&
        virtual_consumers);

::pir::Operation* FindGlobalReducer(
    const std::vector<::pir::Operation*>& ops_in_order);

::pir::Operation* FindNearestReducer(
    ::pir::Operation* op, const std::unordered_set<::pir::Operation*>& ops_set);

bool CanbeInline(::pir::Operation* op,
                 ::pir::Operation* reducer,
                 PrettyNamer* pretty_name,
                 const std::vector<::pir::Operation*> consumers,
                 const std::unordered_set<::pir::Operation*> masters,
                 const GroupPtr& group,
                 const std::unordered_set<::pir::Operation*>& ops_set);

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

void LoopComputeAt(
    ir::IRSchedule& ir_sch,  // NOLINT
    ::pir::Operation* op,
    ::pir::Operation* master,
    PrettyNamer* pretty_name,
    const GroupPtr& group,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map,
    const std::unordered_map<std::string, ir::Tensor>& tmp_tensor_info);

void SyncThreadWithShared(
    ir::IRSchedule& ir_sch,  // NOLINT
    const GroupPtr& group,
    PrettyNamer* pretty_name,
    const std::unordered_set<::pir::Operation*>& ops_inline,
    const std::unordered_set<::pir::Operation*>& ops_set,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map);

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
