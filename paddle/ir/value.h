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

#include "paddle/ir/cast_utils.h"
#include "paddle/ir/type.h"

namespace ir {
class Operation;

namespace detail {
class OpOperandImpl;
class ValueImpl;
class OpResultImpl;
}  // namespace detail

///
/// \brief OpOperand class represents the operand of operation. This class only
/// provides interfaces, for specific implementation, see Impl class.
///
class OpOperand {
 public:
  OpOperand() = default;

  OpOperand(const OpOperand &other) = default;

  OpOperand(const detail::OpOperandImpl *impl);  // NOLINT

  OpOperand &operator=(const OpOperand &rhs);

  OpOperand &operator=(const detail::OpOperandImpl *impl);

  bool operator==(OpOperand other) const;

  bool operator!=(OpOperand other) const;

  bool operator!() const;

  explicit operator bool() const;

  detail::OpOperandImpl *impl() const;

 private:
  detail::OpOperandImpl *impl_{nullptr};
};

///
/// \brief Value Iterator
///
template <typename OperandType>
class ValueUseIterator {
 public:
  ValueUseIterator(OperandType use = nullptr) : current_(use) {}  // NOLINT

  bool operator==(const ValueUseIterator<OperandType> &rhs) const {
    return current_ == rhs.current_;
  }
  ir::Operation *owner() const { return current_.impl()->owner(); }

  OperandType get() const { return current_; }

  OperandType operator*() const { return get(); }

  ValueUseIterator<OperandType> &operator++() {
    current_ = current_.impl()->next_use();
    return *this;
  }

  ValueUseIterator<OperandType> operator++(int) {
    ValueUseIterator<OperandType> tmp = *this;
    ++*(this);
    return tmp;
  }

 protected:
  OperandType current_;
};

///
/// \brief Value class represents the SSA value in the IR system. This class
/// only provides interfaces, for specific implementation, see Impl class.
///
class Value {
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

  detail::ValueImpl *impl() const;

  ir::Type type() const;

  void SetType(ir::Type type);

  Operation *GetDefiningOp() const;

  std::string print_ud_chain();

  ///
  /// \brief Provide iterator interface to access Value use chain.
  ///
  using use_iterator = ValueUseIterator<OpOperand>;

  use_iterator begin() const;

  use_iterator end() const;

  friend struct std::hash<Value>;

 protected:
  detail::ValueImpl *impl_{nullptr};
};

///
/// \brief OpResult class represents the value defined by a result of operation.
/// This class only provides interfaces, for specific implementation, see Impl
/// class.
///
class OpResult : public Value {
 public:
  using Value::Value;

  static bool classof(Value value);

  Operation *owner() const;

  uint32_t GetResultIndex() const;

  friend Operation;

 private:
  static uint32_t GetValidInlineIndex(uint32_t index);

  detail::OpResultImpl *impl() const;
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
