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

#include "paddle/pir/include/core/cast_utils.h"
#include "paddle/pir/include/core/storage_manager_support.h"
#include "paddle/pir/include/core/type_id.h"

constexpr char kAttrStopGradients[] = "stop_gradient";
constexpr char kAttrIsPersistable[] = "persistable";
constexpr char kAttrOpDistAttr[] = "op_dist_attr";

namespace pir {
class AttributeStorage;
class AbstractAttribute;
class IrContext;
class Dialect;

///
/// \brief Unified interface of the Attribute class. Derivation of all Attribute
/// classes only derives interfaces, not members.
///
class IR_API Attribute {
 public:
  using Storage = AttributeStorage;

  Attribute() = default;

  Attribute(const Storage *storage)  // NOLINT
      : storage_(storage) {}

  Attribute(const Attribute &other) = default;

  Attribute &operator=(const Attribute &other) = default;

  bool operator==(Attribute other) const { return storage_ == other.storage_; }

  bool operator!=(Attribute other) const { return storage_ != other.storage_; }

  explicit operator bool() const { return storage_; }

  bool operator!() const { return storage_ == nullptr; }

  operator const void *() const { return storage_; }

  ///
  /// \brief Some Attribute attribute acquisition interfaces.
  ///
  TypeId type_id();

  const AbstractAttribute &abstract_attribute();

  const Storage *storage() const { return storage_; }

  const Dialect &dialect() const;

  IrContext *ir_context() const;

  /// @brief print attribute
  /// @param os
  void Print(std::ostream &os) const;

  static Attribute Parse(std::istream &is, IrContext *ctx);

  ///
  /// \brief Methods for type judgment and cast.
  ///
  static bool classof(Attribute) { return true; }

  template <typename T>
  bool isa() const {
    return pir::isa<T>(*this);
  }

  template <typename U>
  U dyn_cast() const {
    return pir::dyn_cast<U>(*this);
  }

  std::size_t hash() const { return std::hash<const void *>()(storage_); }

 protected:
  const Storage *storage_{nullptr};
};

IR_API std::ostream &operator<<(std::ostream &os, Attribute attr);
}  // namespace pir

namespace std {
template <>
struct hash<pir::Attribute> {
  std::size_t operator()(const pir::Attribute &obj) const { return obj.hash(); }
};
}  // namespace std
