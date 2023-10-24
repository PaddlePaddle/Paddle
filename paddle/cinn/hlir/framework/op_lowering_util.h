// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include <queue>

#include "paddle/cinn/hlir/framework/op_lowering_impl.h"

namespace cinn {
namespace hlir {
namespace framework {

std::vector<NodeData*> GetInputNodeData(const Node* node);

ir::Tensor GetTensor(
    const NodeData* node_data,
    const absl::flat_hash_map<std::string, Type>& type_dict,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict);

std::vector<ir::Tensor> CollectInputTensor(
    const Node* node,
    const absl::flat_hash_map<std::string, Type>& type_dict,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict,
    std::vector<ir::Tensor>* func_args,
    std::unordered_map<std::string, ir::Tensor>* tensor_map);

std::unordered_map<Node*, Node*> BuildVirtualConsumer(
    const GroupPtr& group,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict);

NodeData* GetNodeData(const Node* node);

std::vector<NodeData*> GetAllNodeData(const Node* node);

std::vector<Node*> GetConsumers(const Node* node);

bool IsConstOp(const framework::Node* node);

std::vector<Node*> GetConsumersInSet(const Node* node,
                                     const std::unordered_set<Node*>& node_set);

std::vector<Node*> TopologicalOrder(
    const GroupPtr& group,
    const std::unordered_map<Node*, Node*>& virtual_consumers);

std::vector<Node*> BFSTopologicalOrderWithPriority(
    const GroupPtr& group,
    const std::unordered_map<Node*, Node*>& virtual_consumers,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict);

Node* FindGlobalReducer(const std::vector<Node*>& nodes_in_order);

Node* FindNearestReducer(const Node* node,
                         const std::unordered_set<Node*>& nodes_set);

bool CanbeInline(Node* node,
                 const std::vector<Node*> consumers,
                 const Node* reducer,
                 const std::unordered_set<Node*> masters,
                 const GroupPtr& group,
                 const std::unordered_set<Node*>& nodes_set,
                 const absl::flat_hash_map<std::string, shape_t>& shape_dict);

Node* GetMasterToComputeAt(
    Node* node,
    const std::vector<Node*>& nodes_in_order,
    const std::unordered_set<Node*>& nodes_inline,
    const std::unordered_set<Node*>& nodes_set,
    const std::unordered_map<Node*, Node*>& virtual_consumers,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict);

std::unordered_set<Node*> GetMasters(
    Node* node,
    const std::unordered_set<Node*>& nodes_inline,
    const std::unordered_set<Node*>& nodes_set);

void LoopAssignReduce(
    ir::IRSchedule& ir_sch,  // NOLINT
    const Node* node,
    const Node* reducer,
    const Target& target,
    const std::unordered_map<std::string, ir::Tensor>& tensor_map,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict);

void LoopComputeAt(
    ir::IRSchedule& ir_sch,  // NOLINT
    Node* node,
    const Node* master,
    const GroupPtr& group,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict,
    const std::unordered_map<std::string, ir::Tensor>& tensor_map);

void SyncThreadWithShared(
    ir::IRSchedule& ir_sch,  // NOLINT
    const GroupPtr& group,
    const std::unordered_set<Node*>& nodes_inline,
    const std::unordered_set<Node*>& nodes_set,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict,
    const std::unordered_map<std::string, ir::Tensor>& tensor_map);

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
