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

#include "paddle/pir/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/dialect/shape/ir/shape_op.h"

namespace pir {
namespace dialect {
ShapeDialect::ShapeDialect(IrContext *context)
    : Dialect(name(), context, TypeId::get<ShapeDialect>()) {
  initialize();
}

void ShapeDialect::initialize() {
  RegisterOps<SymbolicDim,
              DimOp,
              TieProductEqualOp,
              TieShapeOp,
              FuncOp,
              TensorDimOp>();
}

void ShapeDialect::PrintOperation(Operation *op, IrPrinter &printer) const {
  if (auto func_op = op->dyn_cast<FuncOp>()) {
    func_op.Print(printer);
  } else {
    printer.PrintGeneralOperation(op);
  }
}

}  // namespace dialect
}  // namespace pir

IR_DEFINE_EXPLICIT_TYPE_ID(pir::dialect::ShapeDialect)
