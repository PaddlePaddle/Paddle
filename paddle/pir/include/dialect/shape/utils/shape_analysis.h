// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/builtin_type_interfaces.h"
#include "paddle/pir/include/core/dll_decl.h"
#include "paddle/pir/include/core/utils.h"
#include "paddle/pir/include/dialect/shape/ir/shape_op.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_builder.h"
#include "paddle/pir/include/dialect/shape/utils/shape_or_data_expr.h"

namespace pir {

// The implementation is based on shape constraint ir.
class IR_API ShapeConstraintIRAnalysis {
 public:
  void Init();

  const std::string GetNextSymName();

  bool HasShapeOrDataForValue(Value val) const;

  const symbol::ShapeOrDataDimExprs& GetShapeOrDataForValue(Value val) const;

  void SetShapeOrDataForValue(Value val,
                              const symbol::ShapeOrDataDimExprs& shape_or_data);

  symbol::DimExprBuilder DimExprBuilder();

  // Used to debug
  void PrintShapeOrDatas() const;

  // Returns true if the two value have the same symbolic shape.
  bool IsShapeEqual(Value lhs, Value rhs) const;

  // Suppose:
  //    lhs_dim_idxs = {ld0, ld1, ...}
  //    rhs_dim_idxs = {rd0, rd1, ...}
  // Returns true if:
  //    lhs.shape[ld0] * lhs.shape[ld1] * ... ==
  //    rhs.shape[rd0] * rhs.shape[rd1] * ...
  bool IsProductEqual(Value lhs,
                      const std::vector<int>& lhs_dim_idxs,
                      Value rhs,
                      const std::vector<int>& rhs_dim_idxs) const;

  // Returns true if:
  //    lhs.shape[lhs_from] * ... lhs.shape[lhs_to-1] ==
  //    rhs.shape[rhs_from] * ... rhs.shape[rhs_to-1]
  bool IsProductEqual(Value lhs,
                      int lhs_from,
                      int lhs_to,
                      Value rhs,
                      int rhs_from,
                      int rhs_to) const;

  // Returns true if the two value have the same number elements.
  bool IsSameNumel(Value lhs, Value rhs) const;

  pir::PrintHooks PrintHook() const;

 private:
  ModuleOp m_;

  int64_t next_sym_idx_ = 0;

  std::unordered_map<Value, symbol::ShapeOrDataDimExprs>
      value_to_shape_or_data_;

  std::vector<symbol::DimExprConstraint> constraints_;
};

class IR_API ShapeAnalysisManager {
 public:
  static ShapeAnalysisManager& Instance();
  ShapeConstraintIRAnalysis& Get(pir::Program* program);

  ShapeAnalysisManager(const ShapeAnalysisManager&) = delete;
  ShapeAnalysisManager(ShapeAnalysisManager&&) = delete;
  ShapeAnalysisManager& operator=(const ShapeAnalysisManager&) = delete;

 private:
  ShapeAnalysisManager() {}
  std::unordered_map<uint64_t, ShapeConstraintIRAnalysis> tables_;
};

#define OP_DECLARE_INFER_SYMBOLIC_SHAPE(name) \
  bool name##OpInferSymbolicShape(            \
      pir::Operation* op, pir::ShapeConstraintIRAnalysis* shape_analysis);

}  // namespace pir
