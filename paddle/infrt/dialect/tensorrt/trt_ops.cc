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
#include "paddle/infrt/dialect/tensorrt/trt_ops.h"
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include "paddle/infrt/common/global.h"
#include "paddle/infrt/dialect/tensorrt/trt_dialect_types.h"

#include "paddle/infrt/dialect/dense_tensor.h"
#include "paddle/infrt/dialect/infrt/ir/infrt_dialect.h"
#include "paddle/infrt/dialect/phi/ir/phi_base.h"

namespace infrt {
namespace trt {

EngineType EngineType::get() {
  return Base::get(::infrt::Global::getMLIRContext());
}

EngineType EngineType::get(mlir::MLIRContext *context) {
  return Base::get(context);
}

TensorRTDialect::TensorRTDialect(mlir::MLIRContext *context)
    : mlir::Dialect("trt", context, mlir::TypeID::get<TensorRTDialect>()) {
  addTypes<EngineType>();
  addOperations<
#define GET_OP_LIST
#include "paddle/infrt/dialect/tensorrt/trt_ops.cpp.inc"  // NOLINT
      >();
}

mlir::Type TensorRTDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef keyword;
  if (parser.parseKeyword(&keyword)) return mlir::Type();
  // parse trt dilaect types, for example: !trt.engine
  if (keyword == "engine") {
    return infrt::trt::EngineType::get(getContext());
  }
  parser.emitError(parser.getCurrentLocation(), "unknown infrt::trt type: ")
      << keyword;
  return mlir::Type();
}

void TensorRTDialect::printType(mlir::Type type,
                                mlir::DialectAsmPrinter &printer) const {
  // print trt dilaect types, for example: !trt.engien
  if (type.isa<infrt::trt::EngineType>()) {
    printer << "engine";
    return;
  }
  llvm_unreachable("unknown infrt::trt type.");
}

}  // namespace trt
}  // namespace infrt

#define GET_OP_CLASSES
#include "paddle/infrt/dialect/tensorrt/trt_ops.cpp.inc"  // NOLINT
