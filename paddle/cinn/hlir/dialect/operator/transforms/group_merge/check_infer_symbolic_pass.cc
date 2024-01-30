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

#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/dialect/shape/utils/dim_expr.h"
#include "paddle/utils/flags.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {

void PrintProgram(pir::ModuleOp m, std::string mgs) {
  std::ostringstream print_stream;
  print_stream << "\n\n";
  m.program()->Print(print_stream);
  print_stream << "\n\n";
  VLOG(4) << "===================== " << mgs << " =====================\n"
          << print_stream.str();
}

std::vector<std::int64_t> GetOriginValueShape(pir::Value value) {
  const auto& dim = value.type().dyn_cast<::pir::DenseTensorType>().dims();
  return ::common::vectorize(dim);
}

std::optional<std::vector<std::int64_t>> GetTargetValueShape(
    pir::Value value, const pir::ShapeConstraintIRAnalysis& shape_analysis) {
  if (!shape_analysis.HasShapeOrDataForValue(value)) {
    return std::nullopt;
  }
  const auto& dynamic_shapes =
      shape_analysis.GetShapeOrDataForValue(value).shape();
  std::vector<std::int64_t> target_shapes{};
  for (const auto& dim_expr_shape : dynamic_shapes) {
    CHECK(dim_expr_shape.Has<std::int64_t>());
    target_shapes.push_back(dim_expr_shape.Get<std::int64_t>());
  }
  return std::make_optional<std::vector<std::int64_t>>(target_shapes);
}

void CompareOriginAndTargetValueShape(
    pir::Value value,
    const pir::ShapeConstraintIRAnalysis& shape_analysis,
    const unsigned int count_op) {
  std::vector<std::int64_t> static_value_shape = GetOriginValueShape(value);
  std::vector<std::int64_t> dynamic_value_shape;
  std::optional<std::vector<std::int64_t>> dynamic_value_shape_opt =
      GetTargetValueShape(value, shape_analysis);
  if (dynamic_value_shape_opt.has_value()) {
    dynamic_value_shape = dynamic_value_shape_opt.value();
  }
  CHECK(static_value_shape.size() == dynamic_value_shape.size())
      << "in the above program, the " << count_op
      << "th op : the shape size is not equal\nthe static shape is: "
      << static_value_shape.size()
      << ", and the dynamic shape is: " << dynamic_value_shape.size();
  for (int i = 0; i < static_value_shape.size(); ++i) {
    CHECK(static_value_shape[i] == dynamic_value_shape[i])
        << "in the above program, the " << count_op
        << "th op : static shape is not equal to dynamic shape\nthe static "
           "shape["
        << i << "] is: " << static_value_shape[i] << ", and the dynamic shape["
        << i << "] is: " << dynamic_value_shape[i];
  }
}

void CheckInferSymbolic(pir::ModuleOp module_op) {
  VLOG(4) << "CheckInferSymbolic Start!";
  PrintProgram(module_op, "CheckInferSymbolic");
  unsigned int count_op = 0;
  const auto& shape_analysis =
      pir::ShapeAnalysisManager::Instance().Get(module_op.program());
  for (uint32_t i = 0; i < module_op->num_regions(); i++) {
    for (const auto& block : module_op->region(i)) {
      for (const auto& op : block) {
        count_op++;
        for (std::size_t j = 0; j < op.num_operands(); ++j) {
          CompareOriginAndTargetValueShape(
              op.operand_source(j), shape_analysis, count_op);
        }
        for (std::size_t j = 0; j < op.num_results(); ++j) {
          CompareOriginAndTargetValueShape(
              op.result(j), shape_analysis, count_op);
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

std::unique_ptr<::pir::Pass> CreateCheckInferSymbolicPass() {
  return std::make_unique<CheckInferSymbolicPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
