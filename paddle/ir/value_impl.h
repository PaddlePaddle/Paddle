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
  /// \brief Interface functions of "offset_first_user_" attribute.
  ///
  uint32_t index() const;

  OpOperandImpl *first_user() const {
    return reinterpret_cast<OpOperandImpl *>(
        reinterpret_cast<uintptr_t>(offset_first_user_) & (~0x07));
  }

  void SetFirstUser(OpOperandImpl *first_user) {
    uint32_t offset = index();
    offset_first_user_ = reinterpret_cast<OpOperandImpl *>(
        reinterpret_cast<uintptr_t>(first_user) + offset);
    VLOG(4) << "The index of this value is " << offset
            << ". Offset and set first user: " << first_user << " -> "
            << offset_first_user_ << ".";
  }

  OpOperandImpl **first_user_addr() { return &offset_first_user_; }

  bool use_empty() const { return first_user() == nullptr; }

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
    offset_first_user_ = reinterpret_cast<OpOperandImpl *>(
        reinterpret_cast<uintptr_t>(nullptr) + index);
    VLOG(4) << "Construct a ValueImpl whose's index is " << index
            << ". The offset first_user address is: " << offset_first_user_;
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
  OpOperandImpl *offset_first_user_ = nullptr;
};

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

///
/// \brief OpOperandImpl
///
class OpOperandImpl {
 public:
  ir::Operation *owner() const { return owner_; }

  ir::detail::OpOperandImpl *next_user() { return next_user_; }

  /// Remove this operand from the current use list.
  void remove_from_ud_chain() {
    if (!back_user_addr_) return;
    if (back_user_addr_ == source_->first_user_addr()) {
      /// NOTE: In ValueImpl, offset_first_user_ use lower three bits storage
      /// index information, so need to be updated using the SetFirstUser method
      /// here.
      source_->SetFirstUser(next_user_);
    } else {
      *back_user_addr_ = next_user_;
    }
    if (next_user_) {
      next_user_->back_user_addr_ = back_user_addr_;
    }
  }

  ~OpOperandImpl() { remove_from_ud_chain(); }

  friend ir::Operation;

 private:
  OpOperandImpl(ir::detail::ValueImpl *source, ir::Operation *owner)
      : source_(source), owner_(owner) {
    back_user_addr_ = source->first_user_addr();
    next_user_ = source->first_user();
    if (next_user_) {
      next_user_->back_user_addr_ = &next_user_;
    }
    source->SetFirstUser(this);
  }

  ir::detail::OpOperandImpl *next_user_ = nullptr;

  /// *back_user_addr_ is prev_user
  ir::detail::OpOperandImpl **back_user_addr_ = nullptr;

  ir::detail::ValueImpl *source_ = nullptr;

  ir::Operation *owner_ = nullptr;
};

}  // namespace detail
}  // namespace ir
