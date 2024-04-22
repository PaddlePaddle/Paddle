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
#include "paddle/pir/include/dialect/shape/utils/shape_or_data_expr.h"

namespace cinn::dialect::ir::details {
using OpLoweringGroup = cinn::hlir::framework::pir::OpLoweringGroup;
using OpLoweringGroupPtr = std::shared_ptr<OpLoweringGroup>;

std::unordered_map<::pir::Value, symbol::ShapeOrDataDimExprs>
CreateGroupShapeOrDataExprs(
    const OpLoweringGroupPtr& group,
    pir::ShapeConstraintIRAnalysis& shape_analysis  // NOLINT
);

}  // namespace cinn::dialect::ir::details
