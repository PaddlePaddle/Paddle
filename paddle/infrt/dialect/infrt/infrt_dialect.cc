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

#define GET_ATTRDEF_CLASSES
#include "paddle/infrt/dialect/infrt/infrt_opsAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "paddle/infrt/dialect/infrt/infrt_ops.cpp.inc"

namespace infrt {

void InfrtDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "paddle/infrt/dialect/infrt/infrt_opsTypes.cpp.inc"  // NOLINT
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "paddle/infrt/dialect/infrt/infrt_opsAttributes.cpp.inc"  // NOLINT
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
    // parse optional lod_level
    if (parser.parseOptionalComma().succeeded()) {
      // llvm::APInt lod_level;
      if (parser.parseInteger(lod_level)) return nullptr;
    }
    // parse ">"
    if (parser.parseGreater()) return nullptr;

    return LoDTensorType::get(
        parser.getContext(), shape, elementType, lod_level);
  }
  if (keyword == "dense_tensor") {
    // parse DenseTensor, for example: !i=Infrt.tensor<X86, CUDA, F32>
    llvm::StringRef target;
    llvm::StringRef layout;
    llvm::StringRef precision;

    // parse "<"
    if (parser.parseLess()) return mlir::Type();
    // parse target
    if (parser.parseKeyword(&target)) return mlir::Type();
    auto targetType = GetTargetType(target);
    if (!targetType) {
      parser.emitError(parser.getCurrentLocation(), "unknown target type: ")
          << target;
      return mlir::Type();
    }

    // parse ","
    if (parser.parseComma()) return mlir::Type();
    // parse precision
    if (parser.parseKeyword(&precision)) return mlir::Type();
    auto precisionType = GetPrecisionType(precision);
    if (!precisionType) {
      parser.emitError(parser.getCurrentLocation(), "unknown precision type: ")
          << precision;
      return mlir::Type();
    }

    // parse ","
    if (parser.parseComma()) return mlir::Type();

    // parse layout
    if (parser.parseKeyword(&layout)) return mlir::Type();
    auto layoutType = GetLayoutType(layout);
    if (!layoutType) {
      parser.emitError(parser.getCurrentLocation(), "unknown layout type: ")
          << layout;
      return mlir::Type();
    }
    // parse ">"
    if (parser.parseGreater()) return mlir::Type();
    return DenseTensorType::get(
        parser.getContext(), *targetType, *precisionType, *layoutType);
  }
  // Todo: parse other type
  return mlir::Type();
}

void InfrtDialect::printType(::mlir::Type type,
                             ::mlir::DialectAsmPrinter &os) const {
  // print LoDTensorType, for example: !Infrt.lod_tensor<3x64x3x3xf32,5>
  if (type.isa<infrt::LoDTensorType>()) {
    auto lod_tensor_type = type.cast<infrt::LoDTensorType>();
    os << "lod_tensor<";
    auto shape = lod_tensor_type.getShape();
    for (auto dim = shape.begin(), e = shape.end() - 1; dim != e; ++dim) {
      *dim < 0 ? os << '?' : os << *dim;
      os << 'x';
    }
    shape.back() < 0 ? os << '?' : os << shape.back();
    os << 'x' << lod_tensor_type.getElementType() << ", "
       << lod_tensor_type.getLod_level() << ">";
    return;
  }

  // print DenseTensorType, for example: !infrt.dense_tensor<CPU, FP32, NCHW>
  if (type.isa<infrt::DenseTensorType>()) {
    auto dense_tensor_type = type.cast<infrt::DenseTensorType>();
    os << "dense_tensor<" << dense_tensor_type.getTarget() << ", "
       << dense_tensor_type.getPrecision() << ", "
       << dense_tensor_type.getLayout() << ">";
    return;
  }

  llvm_unreachable("unknown infrt type.");
}

// /// Parse an attribute registered to this dialect.
// ::mlir::Attribute InfrtDialect::parseAttribute(::mlir::DialectAsmParser
// &parser,
//                                    ::mlir::Type type) const {
//   return mlir::Attribute();
//                                    }
// /// Print an attribute registered to this dialect.
// void InfrtDialect::printAttribute(::mlir::Attribute attr,
//                       ::mlir::DialectAsmPrinter &os) const {

//                       }

}  // namespace infrt
