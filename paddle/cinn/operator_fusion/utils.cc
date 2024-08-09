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

#include "paddle/cinn/operator_fusion/utils.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"

namespace cinn::fusion {

std::optional<std::pair<pir::Value, pir::Value>> GetBroadcastOpInputOuputValue(
    pir::Operation* op) {
  auto* mut_op = const_cast<pir::Operation*>(op);
  if (op->isa<paddle::dialect::ExpandOp>()) {
    auto expand_op = mut_op->dyn_cast<paddle::dialect::ExpandOp>();
    return std::make_pair(expand_op.x(), expand_op.out());
  } else if (op->isa<cinn::dialect::BroadcastOp>()) {
    auto broadcast_op = mut_op->dyn_cast<cinn::dialect::BroadcastOp>();
    return std::make_pair(broadcast_op.x(), broadcast_op.out());
  } else {
    PADDLE_THROW(::common::errors::Unimplemented("Unsupported broadcast op, %s",
                                                 op->name()));
  }
  return std::nullopt;
}

}  // namespace cinn::fusion
