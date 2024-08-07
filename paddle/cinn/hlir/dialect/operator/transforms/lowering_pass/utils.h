// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/cinn/hlir/framework/pir/op_lowering_group.h"

namespace cinn::dialect::ir::details {
using OpLoweringGroup = cinn::hlir::framework::pir::OpLoweringGroup;
using OpLoweringGroupPtr = std::shared_ptr<OpLoweringGroup>;

std::vector<pir::Value> GetBlockOutsideInput(
    const std::vector<pir::Operation*>& op_list);

std::unordered_map<std::string, ::pir::Attribute> GetJitKernelAttr(
    const OpLoweringGroupPtr& group);

OpLoweringGroupPtr BuildOpLoweringGroup(pir::Operation* fusion_op_ptr);

void UpdateGroupShapeOrDataExprs(OpLoweringGroupPtr group);

}  // namespace cinn::dialect::ir::details
