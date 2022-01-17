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

#pragma once

#include <mlir/Dialect/Traits.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/DerivedAttributeOpInterface.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace infrt {
namespace trt {

class TensorRTDialect : public mlir::Dialect {
 public:
  explicit TensorRTDialect(mlir::MLIRContext* context);
  static llvm::StringRef getDialectNamespace() { return "trt"; }
};

}  // namespace trt
}  // namespace infrt

#define GET_OP_CLASSES
#include "paddle/infrt/dialect/tensorrt/trt_ops.hpp.inc"
