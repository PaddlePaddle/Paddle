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

#include "paddle/ir/attribute_base.h"
#include "paddle/ir/cast_utils.h"

namespace ir {
///
/// \brief Unified interface of the Attribute class. Derivation of all Attribute
/// classes only derives interfaces, not members.
///
class Attribute {
 public:
  using Storage = AttributeStorage;

  constexpr Attribute() = default;

  Attribute(const Storage *storage)  // NOLINT
      : storage_(storage) {}

  Attribute(const Attribute &other) = default;

  Attribute &operator=(const Attribute &other) = default;

  bool operator==(Attribute other) const { return storage_ == other.storage_; }

  bool operator!=(Attribute other) const { return storage_ != other.storage_; }

  explicit operator bool() const { return storage_; }

  bool operator!() const { return storage_ == nullptr; }

  ///
  /// \brief Some Attribute attribute acquisition interfaces.
  ///
  TypeId type_id() { return storage_->abstract_attribute().type_id(); }

  const AbstractAttribute &abstract_attribute() {
    return storage_->abstract_attribute();
  }

  const Storage *storage() const { return storage_; }

  const Dialect &dialect() const {
    return storage_->abstract_attribute().dialect();
  }

  IrContext *ir_context() const;

  ///
  /// \brief Methods for type judgment and cast.
  ///
  static bool classof(Attribute) { return true; }

  template <typename T>
  bool isa() const {
    return ir::isa<T>(*this);
  }

  template <typename U>
  U dyn_cast() const {
    return ir::dyn_cast<U>(*this);
  }

  friend struct std::hash<Attribute>;

 protected:
  const Storage *storage_{nullptr};
};
}  // namespace ir

namespace std {
template <>
struct hash<ir::Attribute> {
  std::size_t operator()(const ir::Attribute &obj) const {
    return std::hash<const ir::Attribute::Storage *>()(obj.storage_);
  }
};
}  // namespace std
