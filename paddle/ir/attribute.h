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
      : storage_(const_cast<Storage *>(storage)) {}

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

  Storage *storage() const { return storage_; }

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
  Storage *storage_{nullptr};
};

///
/// \brief A combination of a Attribute name(StrAttribute) and an Attribute
/// value.
///
class StrAttribute;

class NamedAttribute {
 public:
  NamedAttribute(StrAttribute name, Attribute value);

  StrAttribute name() const;

  Attribute value() const;

  void SetName(StrAttribute name);

  void SetValue(Attribute value);

  bool operator<(const NamedAttribute &right) const;

  bool operator==(const NamedAttribute &right) const;

  bool operator!=(const NamedAttribute &right) const;

  friend struct std::hash<NamedAttribute>;

 private:
  NamedAttribute(Attribute name, Attribute value)
      : name_(name), value_(value) {}

  Attribute name_;

  Attribute value_;
};
}  // namespace ir

namespace std {
static std::size_t hash_combine(std::size_t lhs, std::size_t rhs) {
  return lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
}
///
/// \brief Enable hashing Attribute .
///
template <>
struct hash<ir::Attribute> {
  std::size_t operator()(const ir::Attribute &obj) const {
    return std::hash<ir::Attribute::Storage *>()(obj.storage_);
  }
};

template <>
struct hash<ir::NamedAttribute> {
  std::size_t operator()(const ir::NamedAttribute &obj) const {
    std::size_t hash_value = 0;
    hash_value =
        hash_combine(hash_value, std::hash<ir::Attribute>()(obj.name_));
    hash_value =
        hash_combine(hash_value, std::hash<ir::Attribute>()(obj.value_));
    return hash_value;
  }
};
}  // namespace std
