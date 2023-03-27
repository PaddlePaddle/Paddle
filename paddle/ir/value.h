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

  ir::Type type() const { return impl_->type(); }

  void SetType(ir::Type type) { impl_->SetType(type); }

  detail::ValueImpl *impl() const { return impl_; }

  template <typename T>
  bool isa() const {
    return ir::isa<T>(*this);
  }

  template <typename U>
  U dyn_cast() const {
    return ir::dyn_cast<U>(*this);
  }

 protected:
  detail::ValueImpl *impl_{nullptr};
};

///
/// \brief OpResult
///
class OpResult : public Value {
 public:
  using Value::Value;

  static bool classof(Value value) {
    return ir::isa<detail::OpResultImpl>(value.impl());
  }

  Operation *owner() const { return impl()->owner(); }

  uint32_t GetResultIndex() const { return impl()->GetResultIndex(); }

 private:
  detail::OpResultImpl *impl() const {
    return reinterpret_cast<detail::OpResultImpl *>(impl_);
  }
};

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

  detail::OpOperandImpl *impl() const { return impl_; }

 private:
  detail::OpOperandImpl *impl_{nullptr};
};

}  // namespace ir
