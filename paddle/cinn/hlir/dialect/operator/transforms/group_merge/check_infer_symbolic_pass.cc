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

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/check_infer_symbolic_pass.h"

#include <string>
#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/dialect/shape/utils/dim_expr.h"
#include "paddle/utils/flags.h"

namespace {

std::vector<std::int64_t> GetOriginValueShape(pir::Value value) {
  const auto& dim = value.type().dyn_cast<::pir::DenseTensorType>().dims();
  return ::common::vectorize(dim);
}

std::vector<std::int64_t> GetTargetValueShape(
    pir::Value value, const pir::ShapeConstraintIRAnalysis& shape_analysis) {
  const auto& dynamic_shapes =
      shape_analysis.GetShapeOrDataForValue(value).shape();
  std::vector<std::int64_t> target_shapes{};
  for (const auto& dim_expr : dynamic_shapes) {
    CHECK(dim_expr.Has<std::int64_t>());
    target_shapes.push_back(dim_expr.Get<std::int64_t>());
  }
  return target_shapes;
}

void CompareOriginAndTargetValueShape(
    pir::Value value,
    const pir::ShapeConstraintIRAnalysis& shape_analysis,
    const pir::Operation& op) {
  std::vector<std::int64_t> origin_value_shape = GetOriginValueShape(value);
  std::vector<std::int64_t> target_value_shape =
      GetTargetValueShape(value, shape_analysis);
  CHECK(origin_value_shape.size() == target_value_shape.size())
      << op.name() << ": shape size is not equal\nthe origin shape is: "
      << origin_value_shape.size()
      << ", and the target shape is: " << target_value_shape.size();
  for (int i = 0; i < origin_value_shape.size(); ++i) {
    CHECK(origin_value_shape[i] == target_value_shape[i])
        << op.name()
        << ": origin shape is not equal to target shape\nthe origin shape[" << i
        << "] is: " << origin_value_shape[i] << ", and the target shape[" << i
        << "] is: " << target_value_shape[i];
  }
}

void CheckInferSymbolic(pir::ModuleOp module_op) {
  VLOG(4) << "CheckInferSymbolic Start!";
  const auto& shape_analysis =
      pir::ShapeAnalysisManager::Instance().Get(module_op.program());
  for (uint32_t i = 0; i < module_op->num_regions(); i++) {
    for (const auto& block : module_op->region(i)) {
      for (const auto& op : block) {
        for (std::size_t j = 0; j < op.num_operands(); ++j) {
          CompareOriginAndTargetValueShape(
              op.operand_source(j), shape_analysis, op);
        }
        for (std::size_t j = 0; j < op.num_results(); ++j) {
          CompareOriginAndTargetValueShape(op.result(j), shape_analysis, op);
        }
      }
    }
  }
  VLOG(4) << "CheckInferSymbolic Succeed!";
}

class CheckInferSymbolicPass : public pir::Pass {
 public:
  CheckInferSymbolicPass() : pir::Pass("check_infer_symbolic_pass", 1) {}

  void Run(pir::Operation* op) override {
    pir::ModuleOp module_op = op->dyn_cast<pir::ModuleOp>();
    CheckInferSymbolic(module_op);
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};

}  // namespace

namespace cinn {
namespace dialect {
namespace ir {

std::unique_ptr<::pir::Pass> CreateCheckInferSymbolicPass() {
  return std::make_unique<CheckInferSymbolicPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
