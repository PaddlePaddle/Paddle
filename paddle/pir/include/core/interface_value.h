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

#include <memory>
#include <set>
#include <type_traits>

#include "paddle/pir/include/core/type_id.h"
#include "paddle/pir/include/core/utils.h"

namespace pir {

class IR_API InterfaceValue {
 public:
  template <typename Interface, typename Model>
  static InterfaceValue Get();
  TypeId type_id() const { return type_id_; }
  void *model() const { return model_.get(); }

  InterfaceValue() = default;
  InterfaceValue(TypeId type_id) : type_id_(type_id) {}  // NOLINT
  InterfaceValue(const InterfaceValue &) = delete;
  InterfaceValue(InterfaceValue &&) noexcept;
  InterfaceValue &operator=(const InterfaceValue &) = delete;
  InterfaceValue &operator=(InterfaceValue &&) noexcept;
  ~InterfaceValue() = default;
  void swap(InterfaceValue &&val) {
    using std::swap;
    swap(type_id_, val.type_id_);
    swap(model_, val.model_);
  }

  ///
  /// \brief Comparison operations.
  ///
  inline bool operator<(const InterfaceValue &other) const {
    return type_id_ < other.type_id_;
  }

 private:
  TypeId type_id_;
  std::unique_ptr<void, decltype(static_cast<void (*)(void *)>(free))> model_{
      nullptr, static_cast<void (*)(void *)>(free)};
};

template <typename Interface, typename Model>
InterfaceValue InterfaceValue::Get() {
  InterfaceValue val;
  val.type_id_ = TypeId::get<Interface>();
  static_assert(std::is_base_of<typename Interface::Concept, Model>::value,
                "Model must derived from corresponding Interface Concept.");
  static_assert(
      sizeof(typename Interface::Concept) == sizeof(Model),
      "Compared with Concept, Model class shouldn't define new data members");

  void *model_raw = malloc(sizeof(Model));
  if (model_raw == nullptr) {
    throw("Alloc memory for interface failed.");
  }
  static_assert(std::is_trivially_destructible<Model>::value,
                "interface models must be trivially destructible");
  new (model_raw) Model();
  val.model_.reset(model_raw);
  return val;
}

using InterfaceSet = std::set<InterfaceValue>;
}  // namespace pir
