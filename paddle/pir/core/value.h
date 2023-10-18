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

#include "paddle/pir/core/op_operand.h"
#include "paddle/pir/core/type.h"
#include "paddle/pir/core/use_iterator.h"

namespace pir {
class Operation;

namespace detail {
class ValueImpl;
}  // namespace detail

///
/// \brief Value class represents the SSA value in the IR system. This class
/// only provides interfaces, for specific implementation, see Impl class.
///
class IR_API Value {
 public:
  Value() = default;

  Value(detail::ValueImpl *impl) : impl_(impl) {}  // NOLINT

  Value(const Value &other) = default;

  bool operator==(const Value &other) const;

  bool operator!=(const Value &other) const;

  bool operator!() const;

  bool operator<(const Value &other) const;

  explicit operator bool() const;

  template <typename U>
  bool isa() const {
    return U::classof(*this);
  }

  template <typename U>
  U dyn_cast() const {
    return U::dyn_cast_from(*this);
  }

  Type type() const;

  void set_type(Type type);

  std::string PrintUdChain();

  ///
  /// \brief Provide iterator interface to access Value use chain.
  ///
  using UseIterator = ValueUseIterator<OpOperand>;

  UseIterator use_begin() const;

  UseIterator use_end() const;

  OpOperand first_use() const;

  void Print(std::ostream &os) const;

  bool use_empty() const;

  bool HasOneUse() const;

  size_t use_count() const;

  friend struct std::hash<Value>;

  void ReplaceUsesWithIf(
      Value new_value,
      const std::function<bool(OpOperand)> &should_replace) const;
  void ReplaceAllUsesWith(Value new_value) const;
  detail::ValueImpl *impl() const { return impl_; }

 protected:
  detail::ValueImpl *impl_{nullptr};
};

}  // namespace pir

namespace std {
template <>
struct hash<pir::Value> {
  std::size_t operator()(const pir::Value &obj) const {
    return std::hash<const pir::detail::ValueImpl *>()(obj.impl_);
  }
};

}  // namespace std
