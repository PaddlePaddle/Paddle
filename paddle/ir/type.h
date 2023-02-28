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

#include "paddle/ir/type_base.h"

namespace ir {
///
/// \brief Unified interface of the Type class. Derivation of all Type classes
/// only derives interfaces, not members. For example, DenseTensorType,
/// Float32Type, etc. are all derived classes of Type, but no new member
/// variables will be added.
///
class Type {
 public:
  using StorageType = TypeStorage;

  constexpr Type() = default;

  Type(const StorageType *storage)  // NOLINT
      : storage_(const_cast<StorageType *>(storage)) {}

  Type(const Type &other) = default;

  Type &operator=(const Type &other) = default;

  ///
  /// \brief Comparison operations.
  ///
  bool operator==(Type other) const { return storage_ == other.storage_; }
  bool operator!=(Type other) const { return storage_ != other.storage_; }

  explicit operator bool() const { return storage_; }

  bool operator!() const { return storage_ == nullptr; }

  TypeId type_id() { return storage_->abstract_type().type_id(); }

  const AbstractType &abstract_type() { return storage_->abstract_type(); }

  StorageType *storage() const { return storage_; }

  const Dialect &dialect() const { return storage_->abstract_type().dialect(); }

  IrContext *ir_context() const;

  ///
  /// \brief Enable hashing Type.
  ///
  friend struct std::hash<Type>;

 protected:
  StorageType *storage_{nullptr};
};

}  // namespace ir

namespace std {
///
/// \brief Enable hashing Type.
///
template <>
struct hash<ir::Type> {
  std::size_t operator()(const ir::Type &obj) const {
    return std::hash<ir::Type::StorageType *>()(obj.storage_);
  }
};
}  // namespace std
