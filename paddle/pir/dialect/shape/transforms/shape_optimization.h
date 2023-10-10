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
class ShapeComputationIRAnalysis {
 public:
  using func = std::function<bool(Operation* op)>;
  explicit ShapeComputationIRAnalysis(ModuleOp m,
                                      SymbolicDimMgr& mgr);  // NOLINT
  bool Run();

 private:
  bool RunOnRegion(Region* region, func fn);
  bool RunOnBlock(Block* block, func fn);
  bool RunOnOperation(Operation* op, func fn);

  bool BuildShapeOnOperation(Operation* op);
  bool BuildShapeOnValue(Value value);

  bool ApplyOpConstraint(Operation* op);
  bool ApplyIndexOpConstraint(Operation* op);
  bool ApplyTieShapeOpConstraint(Operation* op);

  bool initialized_ = false;
  ModuleOp m_;
  SymbolicDimMgr& mgr_;

  std::unordered_map<Value, SymbolicDim> value2SymDim_;

  // shape tensor is the 1D ranked tensor with int/index dtype.
  std::unordered_map<Value, std::vector<SymbolicDim>> shapeTensor2SymDims_;

  std::unordered_map<Value, std::vector<SymbolicDim>> rankedTensor2SymDims_;
};

}  // namespace pir
