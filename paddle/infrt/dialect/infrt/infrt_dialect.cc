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
  // maybe remove in future
  allowUnknownTypes();
  allowUnknownOperations();
}

/// Parse a type registered to this dialect.
mlir::Type InfrtDialect::parseType(::mlir::DialectAsmParser &parser) const {
  llvm::StringRef keyword;
  if (parser.parseKeyword(&keyword)) return nullptr;
  // parse TensorType, for example: !infrt.tensor<X86, CUDA, F32>
  if (keyword == "lod_tensor") {
    // Parse the size and elementType.
    llvm::SmallVector<int64_t> shape;
    mlir::Type elementType;
    // int32_t lod_level;
    // parse "<"
    if (!parser.parseLess()) return nullptr;

    if (parser.parseDimensionList(shape, /*allowDynamic=*/false) ||
        parser.parseType(elementType))
      return nullptr;
    // parse ","
    if (parser.parseComma()) return mlir::Type();

    llvm::APInt lod_level;
    if (parser.parseInteger(lod_level)) {
      return mlir::Type();
    }
    return mlir::Type();
  }
  // parse TensorType, for example: !infrt.tensor<X86, CUDA, F32>
  if (keyword == "tensor") {
    llvm::StringRef target;
    llvm::StringRef layout;
    llvm::StringRef precision;

    // parse "<"
    if (parser.parseLess()) return mlir::Type();
    // parse target
    if (parser.parseKeyword(&target)) return mlir::Type();
    auto targetType = infrt::dt::GetTargetType(target);
    if (!targetType) {
      parser.emitError(parser.getCurrentLocation(), "unknown target type: ")
          << target;
      return mlir::Type();
    }

    // parse ","
    if (parser.parseComma()) return mlir::Type();
    // parse layout
    if (parser.parseKeyword(&layout)) return mlir::Type();
    auto layoutType = infrt::dt::GetLayoutType(layout);
    if (!layoutType) {
      parser.emitError(parser.getCurrentLocation(), "unknown layout type: ")
          << layout;
      return mlir::Type();
    }

    // parse ","
    if (parser.parseComma()) return mlir::Type();
    // parse precision
    if (parser.parseKeyword(&precision)) return mlir::Type();
    auto precisionType = infrt::dt::GetPrecisionType(precision);
    if (!precisionType) {
      parser.emitError(parser.getCurrentLocation(), "unknown precision type: ")
          << precision;
      return mlir::Type();
    }

    // parse ">"
    if (parser.parseGreater()) return mlir::Type();

    return infrt::dt::TensorType::get(*targetType, *layoutType, *precisionType);
  }

  if (keyword == "string") {
    return infrt::dt::StringType::get();
  }

  // parse TensorMapType, for example: !infrt.tensor_map
  if (keyword == "tensor_map") {
    return infrt::dt::TensorMapType::get();
  }

  // Todo: parse other type
  return mlir::Type();
}

void InfrtDialect::printType(::mlir::Type type,
                             ::mlir::DialectAsmPrinter &os) const {
  // print TensorType, for example: !infrt.tensor<X86, CUDA, F32>
  if (type.isa<infrt::dt::TensorType>()) {
    auto tensorType = type.cast<infrt::dt::TensorType>();
    os << "tensor<" << tensorType.target() << ", " << tensorType.layout()
       << ", " << tensorType.precision() << ">";
    return;
  }
  // print TensorMapType, for example: !infrt.tensor_map
  if (type.isa<infrt::dt::TensorMapType>()) {
    os << "tensor_map";
    return;
  }
  // print StringType, for example: !infrt.string
  if (type.isa<infrt::dt::StringType>()) {
    os << "string";
    return;
  }
  llvm_unreachable("unknown infrt type.");
}

}  // namespace infrt
