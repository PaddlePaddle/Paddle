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
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <string>

using namespace mlir;  // NOLINT
namespace infrt::dt {

namespace detail {
struct TensorTypeStorage;
}  // namespace detail

enum class TargetType : uint8_t { X86, CUDA };
enum class LayoutType : uint8_t { NCHW, NHWC };
enum class PrecisionType : uint8_t { I32, F32 };

llvm::Optional<TargetType> GetTargetType(mlir::StringRef key);
llvm::Optional<LayoutType> GetLayoutType(mlir::StringRef key);
llvm::Optional<PrecisionType> GetPrecisionType(mlir::StringRef key);

raw_ostream &operator<<(raw_ostream &os, TargetType type);
raw_ostream &operator<<(raw_ostream &os, LayoutType type);
raw_ostream &operator<<(raw_ostream &os, PrecisionType type);

class TensorType : public mlir::Type::TypeBase<TensorType,
                                               mlir::Type,
                                               detail::TensorTypeStorage> {
 public:
  using Base::Base;
  static TensorType get(TargetType target,
                        LayoutType layout,
                        PrecisionType precision);

  TargetType target();
  LayoutType layout();
  PrecisionType precision();
};

raw_ostream &operator<<(raw_ostream &os, TensorType tensorType);

class TensorMapType : public mlir::Type::TypeBase<TensorMapType,
                                                  mlir::Type,
                                                  mlir::TypeStorage> {
 public:
  using Base::Base;
  static TensorMapType get();
  static TensorMapType get(mlir::MLIRContext *context);
};

class StringType
    : public mlir::Type::TypeBase<StringType, mlir::Type, mlir::TypeStorage> {
 public:
  using Base::Base;
  static StringType get();
  static StringType get(mlir::MLIRContext *context);
};

#include "paddle/infrt/dialect/dense_tensor_dialect.hpp.inc"

#define GET_OP_CLASSES
#include "paddle/infrt/dialect/dense_tensor.hpp.inc"

}  // namespace infrt::dt
