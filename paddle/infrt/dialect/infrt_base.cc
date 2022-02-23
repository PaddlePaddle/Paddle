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
