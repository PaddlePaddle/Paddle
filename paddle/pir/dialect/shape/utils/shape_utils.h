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

#include "paddle/pir/dialect/shape/utils/shape_optimization_utils.h"
#include "paddle/pir/dialect/shape/utils/symbol_table.h"

namespace pir {

// Helper class to query and manipulate shape constraint IR on buffer level.
class IR_API ShapeAnalysis {
 public:
  virtual ~ShapeAnalysis() = default;

  // Returns true if the two value have the same symbolic shape.
  virtual bool IsShapeEqual(Value lhs, Value rhs) = 0;

  // Suppose:
  //    lhs_dim_idxs = {ld0, ld1, ...}
  //    rhs_dim_idxs = {rd0, rd1, ...}
  // Returns true if:
  //    lhs.shape[ld0] * lhs.shape[ld1] * ... ==
  //    rhs.shape[rd0] * rhs.shape[rd1] * ...
  virtual bool IsProductEqual(Value lhs,
                              std::vector<int> lhs_dim_idxs,
                              Value rhs,
                              std::vector<int> rhs_dim_idxs) = 0;

  // Returns true if:
  //    lhs.shape[lhs_from] * ... lhs.shape[lhs_to-1] ==
  //    rhs.shape[rhs_from] * ... rhs.shape[rhs_to-1]
  virtual bool IsProductEqual(
      Value lhs, int lhs_from, int lhs_to, Value rhs, int rhs_from, int rhs_to);

  // Returns true if the two value have the same number elements.
  virtual bool IsSameNumElements(Value lhs, Value rhs);
};

// A subclass to impement `ShapeAnalysis` on buffer level.
// The implementation is based on shape constraint ir.
class IR_API ShapeConstraintIRAnalysis : public ShapeAnalysis {
 public:
  explicit ShapeConstraintIRAnalysis(ModuleOp m);
  // Auto-save updated shape constriant ir when destroying.
  ~ShapeConstraintIRAnalysis();

  // Returns the `SymbolicDimMgr` this object holds.
  SymbolicDimMgr& symbolicDimMgr() { return mgr_; }
  const SymbolicDimMgr& symbolicDimMgr() const { return mgr_; }

  const std::vector<shape::SymbolicDimOp>&
  GetOrCreateSymbolicDimsForRankedValue(const Value& value);

  // Returns true if the two value have the same symbolic shape.
  bool IsShapeEqual(Value lhs, Value rhs) override;

  bool IsProductEqual(Value lhs,
                      std::vector<int> lhs_dim_idxs,
                      Value rhs,
                      std::vector<int> rhs_dim_idxs) override;

 private:
  // The operation this analysis runs on.
  ModuleOp m_;
  // The `SymbolicDimMgr` this analysis holds.
  SymbolicDimMgr mgr_;
  // Map a ranked memref value to an array of symbolicDims, each represents one
  // dimension size of the memref value.
  std::unordered_map<Value, std::vector<shape::SymbolicDimOp>>
      value_to_sym_dims_;

 public:
  explicit ShapeConstraintIRAnalysis(std::shared_ptr<pir::Program>&& program)
      : ShapeConstraintIRAnalysis(program->module_op()) {
    program_ = std::move(program);
  }

  explicit ShapeConstraintIRAnalysis(pir::IrContext* ctx)
      : ShapeConstraintIRAnalysis(std::make_shared<pir::Program>(ctx)) {}

 private:
  std::shared_ptr<pir::Program> program_;
};

class IR_API ShapeAnalysisManager {
 public:
  static ShapeAnalysisManager& Instance();
  ShapeConstraintIRAnalysis& Get(pir::Program* program);

 private:
  ShapeAnalysisManager() {}
  std::unordered_map<uint64_t, ShapeConstraintIRAnalysis> tables_;
};

}  // namespace pir
