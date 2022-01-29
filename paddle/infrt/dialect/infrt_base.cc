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

#include "paddle/infrt/dialect/infrt_base.h"

#include "paddle/infrt/dialect/basic_kernels.h"
#include "paddle/infrt/dialect/dense_tensor.h"
#include "paddle/infrt/dialect/test_kernels.h"

namespace infrt {
namespace dialect {

// ----INFRTDialect definition begin----
void INFRTDialect::initialize() {
  allowUnknownTypes();
  allowUnknownOperations();

  addTypes<infrt::dt::StringType>();
  addTypes<infrt::dt::TensorType>();
  addTypes<infrt::dt::TensorMapType>();

  addOperations<
#define GET_OP_LIST
#include "paddle/infrt/dialect/basic_kernels.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "paddle/infrt/dialect/test_kernels.cpp.inc"
      >();
}

mlir::Type INFRTDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef keyword;
  if (parser.parseKeyword(&keyword)) return mlir::Type();
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
  // parse TensorMapType, for example: !infrt.tensor_map
  if (keyword == "tensor_map") {
    return infrt::dt::TensorMapType::get();
  }
  // parse StringType, for example: !infrt.string
  if (keyword == "string") {
    return infrt::dt::StringType::get();
  }

  parser.emitError(parser.getCurrentLocation(), "unknown infrt type: ")
      << keyword;
  return mlir::Type();
}

void INFRTDialect::printType(mlir::Type type,
                             mlir::DialectAsmPrinter &printer) const {
  // print TensorType, for example: !infrt.tensor<X86, CUDA, F32>
  if (type.isa<infrt::dt::TensorType>()) {
    auto tensorType = type.cast<infrt::dt::TensorType>();
    printer << "tensor<" << tensorType.target() << ", " << tensorType.layout()
            << ", " << tensorType.precision() << ">";
    return;
  }
  // print TensorMapType, for example: !infrt.tensor_map
  if (type.isa<infrt::dt::TensorMapType>()) {
    printer << "tensor_map";
    return;
  }
  // print StringType, for example: !infrt.string
  if (type.isa<infrt::dt::StringType>()) {
    printer << "string";
    return;
  }
  llvm_unreachable("unknown infrt type.");
}

// ----INFRTDialect definition end----

}  // namespace dialect
}  // namespace infrt
