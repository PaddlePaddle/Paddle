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

#include "paddle/cinn/common/dim_expr_simplify.h"
#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/dialect/shape/utils/dim_expr.h"
#include "paddle/utils/flags.h"

namespace {

template <typename DoEachT>
void VisitEachValue(pir::ModuleOp module_op, const DoEachT& DoEach) {
  for (std::size_t i = 0; i < module_op->num_operands(); ++i) {
    DoEach(module_op->operand_source(i));
  }
  for (std::size_t i = 0; i < module_op->num_results(); ++i) {
    DoEach(module_op->result(i));
  }
}

std::vector<std::int64_t> GetOriginValueShape(pir::Value value) {
  auto& dim = value.type().dyn_cast<::pir::DenseTensorType>().dims();
  return ::common::vectorize(dim);
}

std::vector<std::int64_t> GetTargetValueShape(
    pir::Value value, pir::ShapeConstraintIRAnalysis* shape_analysis) {
  const auto& dynamic_shapes =
      shape_analysis->GetShapeOrDataForValue(value).shape();
  std::vector<std::int64_t> target_shapes{};
  for (const auto& dim_expr : dynamic_shapes) {
    CHECK(dim_expr.Has<std::int64_t>());
    target_shapes.push_back(dim_expr.Get<std::int64_t>());
  }
  return target_shapes;
}

void CheckInferSymbolic(pir::ModuleOp module_op) {
  pir::ShapeConstraintIRAnalysis* shape_analysis =
      &pir::ShapeAnalysisManager::Instance().Get(module_op.program());
  VisitEachValue(module_op, [&](pir::Value value) {
    auto origin_value_shape = GetOriginValueShape(value);
    auto target_value_shape = GetTargetValueShape(value, shape_analysis);
    VLOG(4) << "CheckInferSymbolic";
    CHECK(origin_value_shape == target_value_shape);
  });
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
