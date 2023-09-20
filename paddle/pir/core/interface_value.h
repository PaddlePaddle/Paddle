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
#include "paddle/pir/core/type_id.h"
#include "paddle/pir/core/utils.h"

namespace pir {

class IR_API InterfaceValue {
 public:
  template <typename ConcreteT, typename T>
  static InterfaceValue get() {
    InterfaceValue val;
    val.type_id_ = TypeId::get<T>();
    val.model_ = malloc(sizeof(typename T::template Model<ConcreteT>));
    if (val.model_ == nullptr) {
      throw("Alloc memory for interface failed.");
    }
    static_assert(std::is_trivially_destructible<
                      typename T::template Model<ConcreteT>>::value,
                  "interface models must be trivially destructible");
    new (val.model_) typename T::template Model<ConcreteT>();
    return val;
  }
  TypeId type_id() const { return type_id_; }
  void *model() const { return model_; }

  InterfaceValue() = default;
  explicit InterfaceValue(TypeId type_id) : type_id_(type_id) {}
  InterfaceValue(const InterfaceValue &) = delete;
  InterfaceValue(InterfaceValue &&) noexcept;
  InterfaceValue &operator=(const InterfaceValue &) = delete;
  InterfaceValue &operator=(InterfaceValue &&) noexcept;
  ~InterfaceValue();
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
  void *model_{nullptr};
};

}  // namespace pir
