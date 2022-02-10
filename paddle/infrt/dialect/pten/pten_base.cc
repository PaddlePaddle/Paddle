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

#include "paddle/infrt/dialect/pten/pten_base.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Types.h>
#include "paddle/infrt/common/global.h"
#include "paddle/infrt/dialect/pten/infrt_pten_base.cpp.inc"
#include "paddle/infrt/dialect/pten/infrt_pten_baseDialect.cpp.inc"

namespace infrt {
namespace pten {

void PTENDialect::printType(::mlir::Type type,
                            mlir::DialectAsmPrinter& os) const {
  Dialect::printType(type, os);
}

void PTENDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "paddle/infrt/dialect/pten/infrt_pten_base.cpp.inc"  // NOLINT
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "paddle/infrt/dialect/pten/infrt_pten_baseTypes.cpp.inc"  // NOLINT
      >();
}

mlir::Type PTENDialect::parseType(mlir::DialectAsmParser& parser) const {
  llvm::StringRef keyword;
  if (parser.parseKeyword(&keyword)) return mlir::Type();
  if (keyword == "allocator_CPU") {
    return CPUAllocatorType::get(parser.getContext());
  } else if (keyword == "allocator_GPU") {
    return GPUAllocatorType::get(parser.getContext());
  } else if (keyword == "context_CPU") {
    return CPUContextType::get(parser.getContext());
  } else if (keyword == "context_GPU") {
    return GPUContextType::get(parser.getContext());
  }

  return mlir::Type();
}

}  // namespace pten
}  // namespace infrt

#define GET_TYPEDEF_CLASSES
#include "paddle/infrt/dialect/pten/infrt_pten_baseTypes.cpp.inc"  // NOLINT
