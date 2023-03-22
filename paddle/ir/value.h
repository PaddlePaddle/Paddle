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

namespace ir {
namespace detail {
class ValueImpl;
class OpResultImpl;
class OpOperandImpl;
}  // namespace detail

class Operation;
///
/// \brief OpOperand
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

  ir::OpOperand *&back_user_addr();

 private:
  detail::OpOperandImpl *impl_{nullptr};
};

///
/// \brief Value
///
class Value {
 public:
  explicit Value(detail::ValueImpl *impl) : impl_(impl) {}

  bool operator==(const Value &other) const { return impl_ == other.impl_; }

  bool operator!=(const Value &other) const { return impl_ != other.impl_; }

  bool operator!() const { return impl_ == nullptr; }

  explicit operator bool() const { return impl_; }

  ir::Type type() const;

  void SetType(ir::Type type);

 private:
  detail::ValueImpl *impl_{nullptr};
};

///
/// \brief OpResult
///
class OpResult : public Value {
 public:
  using Value::Value;

  OpOperand &first_user();
};

///
/// \brief details
///
namespace detail {
///
/// \brief ValueImpl
///
class ValueImpl {
 public:
  ///
  /// \brief Value is defined by operator, return the position of this value in
  /// the operator output list.
  ///
  uint32_t index();

  ///
  /// \brief Return the type of this value.
  ///
  ir::Type type() const { return type_; }

  ///
  /// \brief Set the type of this value.
  ///
  void SetType(ir::Type type) { type_ = type; }

  ir::OpOperand &first_user() { return first_user_; }

  friend ir::Value;

 protected:
  ///
  /// \brief Only allowed to be constructed by derived classes such as
  /// OpResultImpl.
  ///
  explicit ValueImpl(ir::Type type) : type_(type) {}

  ir::Type type_;

  ir::OpOperand first_user_;
};

///
/// \brief OpResultImpl
///
class OpResultImpl : public ValueImpl {};

///
/// \brief OpResult, idx < 5
///
class OpInlineResultImpl : public detail::OpResultImpl {};

///
/// \brief OpResult, idx >= 5
///
class OpOutlineResultImpl : public detail::OpResultImpl {
 private:
  uint32_t index_;
};

///
/// \brief OpOperandImpl
///
class OpOperandImpl {
 private:
  OpOperandImpl::OpOperandImpl(ir::OpResult source, ir::Operation *owner)
      : source_(source), owner_(owner) {
    back_user_addr_ = &source.first_user();
    next_user_ = source.first_user();
    if (next_user_) {
      next_user_.back_user_addr() = &next_user_;
    }
    source.first_user() = this;
  }

  ir::OpResult source_;

  ir::OpOperand next_user_;

  ir::OpOperand *back_user_addr_{nullptr};

  ir::Operation *owner_{nullptr};

  friend ir::OpOperand;
};
}  // namespace detail
}  // namespace ir
