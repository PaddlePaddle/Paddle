// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ir/type.h"

namespace ir {
///
/// \brief Interfaces for user-created built-in types. For example:
/// Type fp32 = Float32Type::get(ctx);
///
class Float32Type : public ir::Type {
 public:
  // 可以通过宏定义自动生成的（Float32Type, TypeStorage）
  using Type::Type;  // 必须提供

  using StorageType = ir::TypeStorage;  // 必须指定StorageType

  StorageType *storage() const {
    return static_cast<StorageType *>(this->storage_);
  }  // 必须提供

  static TypeId type_id() { return TypeId::get<Float32Type>(); }  // 必须提供

  template <typename T>  // 必须提供
  static bool classof(T val) {
    return val.type_id() == type_id();
  }

  template <typename... Args>  // 必须提供
  static Float32Type create(IrContext *ctx, Args... args) {
    return ir::TypeUniquer::template get<Float32Type>(ctx, args...);
  }

  // 手动提供的接口
  static Float32Type get(ir::IrContext *context);
};

struct IntegerTypeStorage : public TypeStorage {
  IntegerTypeStorage(unsigned width, unsigned signedness)
      : width_(width), signedness_(signedness) {}
  using ParamKey = std::pair<unsigned, unsigned>;

  static std::size_t HashValue(const ParamKey &key) {
    return hash_combine(std::hash<unsigned>()(std::get<0>(key)),
                        std::hash<unsigned>()(std::get<1>(key)));
  }

  bool operator==(const ParamKey &key) const {
    return ParamKey(width_, signedness_) == key;
  }

  static IntegerTypeStorage *Construct(ParamKey key) {
    return new IntegerTypeStorage(key.first, key.second);
  }

  ParamKey GetAsKey() const { return ParamKey(width_, signedness_); }

  unsigned width_ : 30;
  unsigned signedness_ : 2;

 private:
  static std::size_t hash_combine(std::size_t lhs, std::size_t rhs) {
    return lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
  }
};

class IntegerType : public ir::Type {
 public:
  // 可以通过宏定义自动生成的（IntegerType, TypeStorage）
  using Type::Type;  // 必须提供

  using StorageType = ir::IntegerTypeStorage;  // 必须指定StorageType

  StorageType *storage() const {
    return static_cast<StorageType *>(this->storage_);
  }  // 必须提供

  static TypeId type_id() { return TypeId::get<IntegerType>(); }  // 必须提供

  template <typename T>  // 必须提供
  static bool classof(T val) {
    return val.type_id() == type_id();
  }

  template <typename... Args>  // 必须提供
  static IntegerType create(IrContext *ctx, Args... args) {
    return ir::TypeUniquer::template get<IntegerType>(ctx, args...);
  }

  /// Integer representation maximal bitwidth.
  /// Note: This is aligned with the maximum width of llvm::IntegerType.
  static constexpr unsigned kMaxWidth = (1 << 24) - 1;

  static IntegerType get(ir::IrContext *context,
                         unsigned width,
                         unsigned signedness = 0);
};

}  // namespace ir
