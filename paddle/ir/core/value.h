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

#include "paddle/ir/core/cast_utils.h"
#include "paddle/ir/core/type.h"
#include "paddle/ir/core/use_iterator.h"

namespace ir {
class Operation;
class Value;

namespace detail {
class OpOperandImpl;
class ValueImpl;
class OpResultImpl;
}  // namespace detail

///
/// \brief OpOperand class represents the op_operand of operation. This class
/// only provides interfaces, for specific implementation, see Impl class.
///
class IR_API OpOperand {
 public:
  OpOperand() = default;

  OpOperand(const OpOperand &other) = default;

  OpOperand(const detail::OpOperandImpl *impl);  // NOLINT

  OpOperand &operator=(const OpOperand &rhs);

  OpOperand &operator=(const detail::OpOperandImpl *impl);

  bool operator==(const OpOperand &other) const { return impl_ == other.impl_; }

  bool operator!=(const OpOperand &other) const { return !operator==(other); }

  bool operator!() const { return impl_ == nullptr; }

  operator bool() const;

  OpOperand next_use() const;

  Value source() const;

  Type type() const;

  void set_source(Value value);

  Operation *owner() const;

  void RemoveFromUdChain();

  friend Operation;

 private:
  detail::OpOperandImpl *impl_{nullptr};
};

///
/// \brief Value class represents the SSA value in the IR system. This class
/// only provides interfaces, for specific implementation, see Impl class.
///
class IR_API Value {
 public:
  Value() = default;

  Value(const detail::ValueImpl *impl);  // NOLINT

  Value(const Value &other) = default;

  bool operator==(const Value &other) const;

  bool operator!=(const Value &other) const;

  bool operator!() const;

  explicit operator bool() const;

  template <typename T>
  bool isa() const {
    return ir::isa<T>(*this);
  }

  template <typename U>
  U dyn_cast() const {
    return ir::dyn_cast<U>(*this);
  }

  Type type() const;

  void set_type(Type type);

  Operation *GetDefiningOp() const;

  std::string PrintUdChain();

  ///
  /// \brief Provide iterator interface to access Value use chain.
  ///
  using UseIterator = ValueUseIterator<OpOperand>;

  UseIterator use_begin() const;

  UseIterator use_end() const;

  OpOperand first_use() const;

  bool use_empty() const;

  bool HasOneUse() const;

  friend struct std::hash<Value>;

  void ReplaceUsesWithIf(
      Value new_value,
      const std::function<bool(OpOperand)> &should_replace) const;
  void ReplaceAllUsesWith(Value new_value) const;

  detail::ValueImpl *impl() { return impl_; }
  const detail::ValueImpl *impl() const { return impl_; }

 protected:
  detail::ValueImpl *impl_{nullptr};
};

///
/// \brief OpResult class represents the value defined by a result of operation.
/// This class only provides interfaces, for specific implementation, see Impl
/// class.
///
class IR_API OpResult : public Value {
 public:
  using Value::Value;

  static bool classof(Value value);

  Operation *owner() const;

  uint32_t GetResultIndex() const;

  bool operator==(const OpResult &other) const;

  friend Operation;

  detail::ValueImpl *value_impl() const;
  detail::OpResultImpl *impl() const;

 private:
  static uint32_t GetValidInlineIndex(uint32_t index);
};

}  // namespace ir

namespace std {
template <>
struct hash<ir::Value> {
  std::size_t operator()(const ir::Value &obj) const {
    return std::hash<const ir::detail::ValueImpl *>()(obj.impl_);
  }
};

}  // namespace std
