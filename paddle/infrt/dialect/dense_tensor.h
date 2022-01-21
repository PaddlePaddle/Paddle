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

namespace infrt {
namespace dt {
enum class TargetType : uint8_t { X86, CUDA };
enum class LayoutType : uint8_t { NCHW, NHWC };
enum class PrecisionType : uint8_t { I32, F32 };

llvm::Optional<TargetType> GetTargetType(mlir::StringRef key);
llvm::Optional<LayoutType> GetLayoutType(mlir::StringRef key);
llvm::Optional<PrecisionType> GetPrecisionType(mlir::StringRef key);

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, TargetType type);
mlir::raw_ostream &operator<<(mlir::raw_ostream &os, LayoutType type);
mlir::raw_ostream &operator<<(mlir::raw_ostream &os, PrecisionType type);

namespace detail {
struct TensorTypeStorage : public mlir::TypeStorage {
  TensorTypeStorage(TargetType target,
                    LayoutType layout,
                    PrecisionType precision)
      : target_(target), layout_(layout), precision_(precision) {}

  using KeyTy = std::tuple<TargetType, LayoutType, PrecisionType>;

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(target_, layout_, precision_);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static TensorTypeStorage *construct(
      mlir::TypeStorageAllocator &allocator,  // NOLINT
      const KeyTy &key) {
    return new (allocator.allocate<TensorTypeStorage>())
        TensorTypeStorage(std::get<0>(key), std::get<1>(key), std::get<2>(key));
  }

  TargetType target_;
  LayoutType layout_;
  PrecisionType precision_;
};
}  // namespace detail

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

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, TensorType tensorType);

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
}  // namespace dt
}  // namespace infrt

#include "paddle/infrt/dialect/dense_tensor_dialect.hpp.inc"

#define GET_OP_CLASSES
#include "paddle/infrt/dialect/dense_tensor.hpp.inc"
