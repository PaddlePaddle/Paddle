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

#include "paddle/ir/type.h"
#include "paddle/ir/value_impl.h"

namespace ir {
class Operation;

///
/// \brief Value class represents the SSA value in the IR system. This class
/// only provides interfaces, for specific implementation, see Impl class.
///
class Value {
 public:
  Value() = default;

  Value(const detail::ValueImpl *impl)  // NOLINT
      : impl_(const_cast<detail::ValueImpl *>(impl)) {}

  Value(const Value &other) = default;

  bool operator==(const Value &other) const { return impl_ == other.impl_; }

  bool operator!=(const Value &other) const { return impl_ != other.impl_; }

  bool operator!() const { return impl_ == nullptr; }

  explicit operator bool() const { return impl_; }

  template <typename T>
  bool isa() const {
    return ir::isa<T>(*this);
  }

  template <typename U>
  U dyn_cast() const {
    return ir::dyn_cast<U>(*this);
  }

  detail::ValueImpl *impl() const { return impl_; }

  ir::Type type() const { return impl_->type(); }

  void SetType(ir::Type type) { impl_->SetType(type); }

  Operation *GetDefiningOp() const;

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

  static bool classof(Value value) {
    return ir::isa<detail::OpResultImpl>(value.impl());
  }

  Operation *owner() const { return impl()->owner(); }

  uint32_t GetResultIndex() const { return impl()->GetResultIndex(); }

  friend Operation;

 private:
  detail::OpResultImpl *impl() const {
    return reinterpret_cast<detail::OpResultImpl *>(impl_);
  }

  static uint32_t GetValidInlineIndex(uint32_t index);
};

///
/// \brief OpOperand class represents the operand of operation. This class only
/// provides interfaces, for specific implementation, see Impl class.
///
class OpOperand {
 public:
  OpOperand() = default;

  OpOperand(const OpOperand &other) = default;

  explicit OpOperand(const detail::OpOperandImpl *impl)
      : impl_(const_cast<detail::OpOperandImpl *>(impl)) {}

  OpOperand &operator=(const OpOperand &rhs) {
    if (this == &rhs) return *this;
    impl_ = rhs.impl_;
    return *this;
  }

  OpOperand &operator=(const detail::OpOperandImpl *impl) {
    if (this->impl_ == impl) return *this;
    impl_ = const_cast<detail::OpOperandImpl *>(impl);
    return *this;
  }

  bool operator==(OpOperand other) const { return impl_ == other.impl_; }

  bool operator!=(OpOperand other) const { return impl_ != other.impl_; }

  bool operator!() const { return impl_ == nullptr; }

  explicit operator bool() const { return impl_; }

  detail::OpOperandImpl *impl() const { return impl_; }

 private:
  detail::OpOperandImpl *impl_{nullptr};
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
