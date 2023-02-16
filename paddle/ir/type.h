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

#include "paddle/ir/type_support.h"

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

  Type(const StorageType *impl)  // NOLINT
      : impl_(const_cast<StorageType *>(impl)) {}

  Type(const Type &other) = default;

  Type &operator=(const Type &other) = default;

  ///
  /// \brief Comparison operations.
  ///
  bool operator==(Type other) const { return impl_ == other.impl_; }
  bool operator!=(Type other) const { return impl_ != other.impl_; }

  explicit operator bool() const { return impl_; }

  bool operator!() const { return impl_ == nullptr; }

  TypeId type_id() { return impl_->abstract_type().type_id(); }

  const AbstractType &abstract_type() { return impl_->abstract_type(); }

  StorageType *impl() const { return impl_; }

  ///
  /// \brief Enable hashing Type.
  ///
  friend struct std::hash<Type>;

 protected:
  StorageType *impl_{nullptr};
};

}  // namespace ir

namespace std {
///
/// \brief Enable hashing Type.
///
template <>
struct hash<ir::Type> {
  std::size_t operator()(const ir::Type &obj) const {
    return std::hash<ir::Type::StorageType *>()(obj.impl_);
  }
};
}  // namespace std
