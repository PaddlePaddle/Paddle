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
#include "paddle/common/flags.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/pir/include/core/builtin_type.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {

std::string SprintShape(const std::vector<std::vector<std::int64_t>>& shapes) {
  std::string str;
  for (int i = 0; i < shapes.size(); i++) {
    str += "[";
    for (int j = 0; j < shapes[i].size(); j++) {
      str += std::to_string(shapes[i][j]);
      if (j != shapes[i].size() - 1) {
        str += ", ";
      }
    }
    str += "]";
    if (i != shapes.size() - 1) {
      str += ", ";
    }
  }
  return str;
}

void PrintProgram(pir::ModuleOp m, const std::string& mgs) {
  std::ostringstream print_stream;
  print_stream << "\n\n";
  m.program()->Print(print_stream);
  print_stream << "\n\n";
  VLOG(4) << "===================== " << mgs << " =====================\n"
          << print_stream.str();
}

std::vector<std::vector<std::int64_t>> GetStaticValueShape(pir::Value value) {
  std::vector<std::vector<std::int64_t>> static_shape;
  if (const pir::DenseTensorType& dense_tensor =
          value.type().dyn_cast<::pir::DenseTensorType>()) {
    static_shape.push_back(::common::vectorize(dense_tensor.dims()));
  } else if (const pir::VectorType vector_tensor =
                 value.type().dyn_cast<::pir::VectorType>()) {
    for (size_t i = 0; i < vector_tensor.size(); i++) {
      if (vector_tensor[i].isa<pir::DenseTensorType>()) {
        const pir::DenseTensorType& dense_tensor =
            vector_tensor[i].dyn_cast<::pir::DenseTensorType>();
        static_shape.push_back(::common::vectorize(dense_tensor.dims()));
      }
    }
  } else {
    IR_THROW("error:the value doesn't have DenseTensorType");
  }
  return static_shape;
}

std::vector<std::int64_t> GetShapeFromTensor(
    const symbol::TensorShapeOrDataDimExprs& tensor_shape_or_data) {
  std::vector<std::int64_t> dynamic_shape;
  for (const auto& dim_expr_shape : tensor_shape_or_data.shape()) {
    CHECK(dim_expr_shape.Has<std::int64_t>());
    dynamic_shape.push_back(dim_expr_shape.Get<std::int64_t>());
  }
  return dynamic_shape;
}

std::vector<std::vector<std::int64_t>> GetDynamicValueShape(
    pir::Value value, const pir::ShapeConstraintIRAnalysis& shape_analysis) {
  std::vector<std::vector<std::int64_t>> dynamic_shapes;
  if (!shape_analysis.HasShapeOrDataForValue(value)) {
    return dynamic_shapes;
  }
  symbol::ShapeOrDataDimExprs shape_or_data =
      shape_analysis.GetShapeOrDataForValue(value);
  auto lambdas = symbol::Overloaded{
      [&](const symbol::TensorShapeOrDataDimExprs& tensor_shape_or_data) {
        dynamic_shapes.push_back(GetShapeFromTensor(tensor_shape_or_data));
      },
      [&](const symbol::TensorListShapeOrDataDimExprs& tensor_list) {
        for (const auto& tensor_shape_or_data : tensor_list) {
          dynamic_shapes.push_back(GetShapeFromTensor(tensor_shape_or_data));
        }
      }};
  std::visit(lambdas, shape_or_data.variant());
  return dynamic_shapes;
}

void CompareStaticAndDynamicValueShape(
    pir::Value value,
    const pir::ShapeConstraintIRAnalysis& shape_analysis,
    int op_index,
    pir::ModuleOp module_op) {
  std::vector<std::vector<std::int64_t>> static_value_shape =
      GetStaticValueShape(value);
  std::vector<std::vector<std::int64_t>> dynamic_value_shape =
      GetDynamicValueShape(value, shape_analysis);
  if (static_value_shape != dynamic_value_shape) {
    VLOG(4) << "CheckInferSymbolic failed, in the following program, the "
            << op_index
            << "th op : the shape is not equal\nthe static shape is: "
            << SprintShape(static_value_shape) << ", and the dynamic shape is: "
            << SprintShape(dynamic_value_shape);
    PrintProgram(module_op, "CheckInferSymbolic");
  }
}

void CheckInferSymbolic(pir::ModuleOp module_op) {
  VLOG(4) << "CheckInferSymbolic start";
  int op_index = 0;
  const auto& shape_analysis =
      pir::ShapeAnalysisManager::Instance().Get(module_op.program());
  for (uint32_t i = 0; i < module_op->num_regions(); i++) {
    for (const auto& block : module_op->region(i)) {
      for (const auto& op : block) {
        for (std::size_t j = 0; j < op.num_operands(); ++j) {
          CompareStaticAndDynamicValueShape(
              op.operand_source(j), shape_analysis, op_index, module_op);
        }
        for (std::size_t j = 0; j < op.num_results(); ++j) {
          CompareStaticAndDynamicValueShape(
              op.result(j), shape_analysis, op_index, module_op);
        }
        op_index++;
      }
    }
  }
  VLOG(4) << "CheckInferSymbolic end";
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
