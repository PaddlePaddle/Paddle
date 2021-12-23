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
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace infrt {
namespace trt {

TensorRTDialect::TensorRTDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect("trt", context, ::mlir::TypeID::get<TensorRTDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "paddle/infrt/dialect/tensorrt/trt_ops.cpp.inc"  // NOLINT
      >();
#undef GET_OP_LIST
}

#define GET_OP_CLASSES
#include "paddle/infrt/dialect/tensorrt/trt_ops.cpp.inc"  // NOLINT
#undef GET_OP_CLASSES

}  // namespace trt
}  // namespace infrt
