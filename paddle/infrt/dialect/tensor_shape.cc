// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/dialect/tensor_shape.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Support/LogicalResult.h>

namespace infrt {
namespace ts {
using namespace mlir;  // NOLINT

void TensorShapeDialect::initialize() {
  allowUnknownTypes();
  addTypes<ShapeType, PartialShapeType>();
  addOperations<
#define GET_OP_LIST
#include "paddle/infrt/dialect/tensor_shape.cpp.inc"
      >();
}

Type TensorShapeDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword)) return Type();
  if (keyword == "shape") return ShapeType::get(getContext());
  if (keyword == "partial_shape") return PartialShapeType::get(getContext());

  parser.emitError(parser.getNameLoc(), "unknown shape type: ") << keyword;
  return Type();
}

void TensorShapeDialect::printType(mlir::Type type,
                                   mlir::DialectAsmPrinter &os) const {
  if (type.isa<ShapeType>()) {
    os << "shape";
    return;
  }

  if (type.isa<PartialShapeType>()) {
    os << "partial_shape";
    return;
  }
  llvm_unreachable("unexpected 'shape' type kind");
}
}  // namespace ts
}  // namespace infrt

#define GET_OP_CLASSES
#include "paddle/infrt/dialect/tensor_shape.cpp.inc"  // NOLINT

#include "paddle/infrt/dialect/tensor_shape_dialect.cpp.inc"
