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

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/special_ops_fusion_rule.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"

namespace cinn {
namespace dialect {
namespace ir {

bool GatherNdFusionRule(const ::pir::Operation* consumer,
                        OpPatternKind producer_group_pattern) {
  if (producer_group_pattern == OpPatternKind::kReduction) {
    return false;
  }
  return true;
}

bool SliceFusionRule(const ::pir::Operation* consumer,
                     OpPatternKind producer_group_pattern) {
  if (producer_group_pattern == OpPatternKind::kReduction) {
    return false;
  }
  return true;
}

void SpecialOpsFusionRule::Init() {
  RegisterConsumerOpRule(paddle::dialect::GatherNdOp::name(),
                         &GatherNdFusionRule);
  RegisterConsumerOpRule(cinn::dialect::SliceOp::name(), &SliceFusionRule);
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
