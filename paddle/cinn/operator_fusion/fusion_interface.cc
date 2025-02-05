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

#include "paddle/cinn/operator_fusion/fusion_interface.h"
#include "paddle/cinn/operator_fusion/fusion_tracker/expr_utils.h"
#include "paddle/cinn/operator_fusion/fusion_tracker/interpreter.h"
#include "paddle/cinn/operator_fusion/utils.h"

COMMON_DECLARE_bool(enable_fusion_result_check);
namespace cinn::fusion {

std::vector<ir::Expr> OperationFusion(
    const std::vector<::pir::Operation*>& ops,
    const std::vector<ir::Expr>& op_compute_bodies,
    FusionTrackerPtr fusion_tracker_ptr) {
  std::vector<FusibleOp> initialized_lowered_op;
  for (int i = 0; i < ops.size(); i++) {
    auto fusible_op =
        cinn::hlir::framework::pir::trivial_fusion_detail::IsReduceBody(
            op_compute_bodies[i])
            ? FusibleOp(ReduceOp(op_compute_bodies[i]))
            : FusibleOp(TrivialOp(op_compute_bodies[i]));
    initialized_lowered_op.push_back(fusible_op);
  }

  auto interpreter =
      FusionInterpreter(fusion_tracker_ptr, ops, initialized_lowered_op);
  auto output = interpreter.Run();

  if (FLAGS_enable_fusion_result_check) {
    cinn::hlir::framework::pir::trivial_fusion_detail::CheckLoopAlignment(
        output);
  }

  VLOG(4) << "Fusion Result: output size is " << output.size();
  for (const auto& expr : output) {
    VLOG(4) << expr;
  }
  return output;
}

}  // namespace cinn::fusion
