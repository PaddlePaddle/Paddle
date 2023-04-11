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
static const uint32_t OUTLINE_OP_RESULT_INDEX = 6;

class Operation;

namespace detail {
class OpOperandImpl;

///
/// \brief ValueImpl is the base class of all drived Value classes such as
/// OpResultImpl. This class defines all the information and usage interface in
/// the IR Value. Each Value include three attributes:
/// (1) type: ir::Type; (2) UD-chain of value: OpOperandImpl*, first operand
/// address with offset of this value; (3) index: the position where the output
/// list of the parent operator.
///
class alignas(8) ValueImpl {
 public:
  ///
  /// \brief Interface functions of "type_" attribute.
  ///
  ir::Type type() const { return type_; }

  void SetType(ir::Type type) { type_ = type; }

  ///
  /// \brief Interface functions of "first_use_offseted_by_index_" attribute.
  ///
  uint32_t index() const;

  OpOperandImpl *first_use() const {
    return reinterpret_cast<OpOperandImpl *>(
        reinterpret_cast<uintptr_t>(first_use_offseted_by_index_) & (~0x07));
  }

  void SetFirstUse(OpOperandImpl *first_use) {
    uint32_t offset = index();
    first_use_offseted_by_index_ = reinterpret_cast<OpOperandImpl *>(
        reinterpret_cast<uintptr_t>(first_use) + offset);
    VLOG(4) << "The index of this value is " << offset
            << ". Offset and set first use: " << first_use << " -> "
            << first_use_offseted_by_index_ << ".";
  }

  OpOperandImpl **first_use_addr() { return &first_use_offseted_by_index_; }

  bool use_empty() const { return first_use() == nullptr; }

  std::string print_ud_chain();

 protected:
  ///
  /// \brief Only can be constructed by derived classes such as OpResultImpl.
  ///
  explicit ValueImpl(ir::Type type, uint32_t index) {
    if (index > OUTLINE_OP_RESULT_INDEX) {
      throw("The value of index must not exceed 6");
    }
    type_ = type;
    first_use_offseted_by_index_ = reinterpret_cast<OpOperandImpl *>(
        reinterpret_cast<uintptr_t>(nullptr) + index);
    VLOG(4) << "Construct a ValueImpl whose's index is " << index
            << ". The offset first_use address is: "
            << first_use_offseted_by_index_;
  }

  ///
  /// \brief Attribute1: Type of value.
  ///
  ir::Type type_;

  ///
  /// \brief Attribute2/3: Record the UD-chain of value and index.
  /// NOTE: The members of the OpOperandImpl include four pointers, so this
  /// class is 8-byte aligned, and the lower 3 bits of its address are 0, so the
  /// index can be stored in these 3 bits, stipulate:
  /// (1) index = 0~5: represent positions 0 to 5 inline
  /// output(OpInlineResultImpl); (2) index = 6: represent the position >=6
  /// outline output(OpOutlineResultImpl); (3) index = 7 is reserved.
  ///
  OpOperandImpl *first_use_offseted_by_index_ = nullptr;
};

}  // namespace detail

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

  std::string print_ud_chain();

  friend struct std::hash<Value>;

 protected:
  detail::ValueImpl *impl_{nullptr};
};

namespace detail {
///
/// \brief OpResultImpl is the implementation of an operation result.
///
class alignas(8) OpResultImpl : public ValueImpl {
 public:
  using ValueImpl::ValueImpl;

  static bool classof(const ValueImpl &value) { return true; }

  ///
  /// \brief Get the parent operation of this result.(op_ptr = value_ptr +
  /// index)
  ///
  ir::Operation *owner() const;

  ///
  /// \brief Get the result index of the operation result.
  ///
  uint32_t GetResultIndex() const;

  ///
  /// \brief Get the maximum number of results that can be stored inline.
  ///
  static uint32_t GetMaxInlineResultIndex() {
    return OUTLINE_OP_RESULT_INDEX - 1;
  }
};

///
/// \brief OpInlineResultImpl is the implementation of an operation result whose
/// index <= 5.
///
class OpInlineResultImpl : public OpResultImpl {
 public:
  OpInlineResultImpl(ir::Type type, uint32_t result_index)
      : OpResultImpl(type, result_index) {
    if (result_index > GetMaxInlineResultIndex()) {
      throw("Inline result index should not exceed MaxInlineResultIndex(5)");
    }
  }

  static bool classof(const OpResultImpl &value) {
    return value.index() < OUTLINE_OP_RESULT_INDEX;
  }

  uint32_t GetResultIndex() const { return index(); }
};

///
/// \brief OpOutlineResultImpl is the implementation of an operation result
/// whose index > 5.
///
class OpOutlineResultImpl : public OpResultImpl {
 public:
  OpOutlineResultImpl(ir::Type type, uint32_t outline_index)
      : OpResultImpl(type, OUTLINE_OP_RESULT_INDEX),
        outline_index_(outline_index) {}

  static bool classof(const OpResultImpl &value) {
    return value.index() >= OUTLINE_OP_RESULT_INDEX;
  }

  uint32_t GetResultIndex() const { return outline_index_; }

  uint32_t outline_index_;
};

}  // namespace detail

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

namespace detail {
///
/// \brief OpOperandImpl
///
class OpOperandImpl {
 public:
  ir::Operation *owner() const { return owner_; }

  ir::detail::OpOperandImpl *next_use() { return next_use_; }

  /// Remove this operand from the current use list.
  void remove_from_ud_chain() {
    if (!prev_use_addr_) return;
    if (prev_use_addr_ == source_.impl()->first_use_addr()) {
      /// NOTE: In ValueImpl, first_use_offseted_by_index_ use lower three bits
      /// storage index information, so need to be updated using the SetFirstUse
      /// method here.
      source_.impl()->SetFirstUse(next_use_);
    } else {
      *prev_use_addr_ = next_use_;
    }
    if (next_use_) {
      next_use_->prev_use_addr_ = prev_use_addr_;
    }
  }

  ~OpOperandImpl() { remove_from_ud_chain(); }

  friend ir::Operation;

 private:
  OpOperandImpl(ir::Value source, ir::Operation *owner)
      : source_(source), owner_(owner) {
    prev_use_addr_ = source.impl()->first_use_addr();
    next_use_ = source.impl()->first_use();
    if (next_use_) {
      next_use_->prev_use_addr_ = &next_use_;
    }
    source.impl()->SetFirstUse(this);
  }

  ir::detail::OpOperandImpl *next_use_ = nullptr;

  ir::detail::OpOperandImpl **prev_use_addr_ = nullptr;

  ir::Value source_;

  ir::Operation *owner_ = nullptr;
};

}  // namespace detail

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
