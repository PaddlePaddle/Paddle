// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/dialect/infrt/infrt_dialect.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectImplementation.h>
#include "paddle/infrt/dialect/dense_tensor.h"
#include "paddle/infrt/dialect/infrt/infrt_opsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "paddle/infrt/dialect/infrt/infrt_opsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "paddle/infrt/dialect/infrt/infrt_ops.cpp.inc"

namespace infrt {

void InfrtDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "paddle/infrt/dialect/infrt/infrt_opsTypes.cpp.inc"  // NOLINT
      >();
  addOperations<
#define GET_OP_LIST
#include "paddle/infrt/dialect/infrt/infrt_ops.cpp.inc"  // NOLINT
      >();
}

/// Parse a type registered to this dialect.
mlir::Type InfrtDialect::parseType(::mlir::DialectAsmParser &parser) const {
  llvm::StringRef keyword;
  if (parser.parseKeyword(&keyword)) return nullptr;
  // parse TensorType, for example: !infrt.lod_tensor<3x64x3x3xf32,5>
  // 5 is the lod_level
  if (keyword == "lod_tensor") {
    // Parse the size and elementType.
    llvm::SmallVector<int64_t, 4> shape;
    mlir::Type elementType;
    int32_t lod_level = 0;
    // parse "<"
    if (parser.parseLess()) return nullptr;

    if (parser.parseDimensionList(shape)) return nullptr;

    // Parse the element type.
    if (parser.parseType(elementType)) return nullptr;
    // parse ","
    if (parser.parseComma()) return nullptr;

    // llvm::APInt lod_level;
    if (parser.parseInteger(lod_level)) return nullptr;

    // parse ">"
    if (parser.parseGreater()) return nullptr;

    return LoDTensorType::get(
        parser.getContext(), shape, elementType, lod_level);
  }
  // Todo: parse other type
  return mlir::Type();
}

void InfrtDialect::printType(::mlir::Type type,
                             ::mlir::DialectAsmPrinter &os) const {
  // print TensorType, for example: !infrt.tensor<X86, CUDA, F32>
  if (type.isa<infrt::LoDTensorType>()) {
    auto lodTensorType = type.cast<infrt::LoDTensorType>();
    os << "lod_tensor<";
    auto shape = lodTensorType.getShape();
    for (auto dim = shape.begin(), e = shape.end() - 1; dim != e; ++dim)
      os << *dim << 'x';
    os << shape.back() << 'x' << lodTensorType.getElementType() << ", "
       << lodTensorType.getLod_level() << ">";
    return;
  }
  llvm_unreachable("unknown infrt type.");
}

}  // namespace infrt
