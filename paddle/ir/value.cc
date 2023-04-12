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

#include "paddle/ir/value.h"
#include "paddle/ir/value_impl.h"

namespace ir {
// Operand
OpOperand::OpOperand(const detail::OpOperandImpl *impl)
    : impl_(const_cast<detail::OpOperandImpl *>(impl)) {}

OpOperand &OpOperand::operator=(const OpOperand &rhs) {
  if (this == &rhs) return *this;
  impl_ = rhs.impl_;
  return *this;
}

OpOperand &OpOperand::operator=(const detail::OpOperandImpl *impl) {
  if (this->impl_ == impl) return *this;
  impl_ = const_cast<detail::OpOperandImpl *>(impl);
  return *this;
}

bool OpOperand::operator==(OpOperand other) const {
  return impl_ == other.impl_;
}

bool OpOperand::operator!=(OpOperand other) const {
  return impl_ != other.impl_;
}

bool OpOperand::operator!() const { return impl_ == nullptr; }

OpOperand::operator bool() const { return impl_; }

detail::OpOperandImpl *OpOperand::impl() const { return impl_; }

// Value
Value::Value(const detail::ValueImpl *impl)
    : impl_(const_cast<detail::ValueImpl *>(impl)) {}

bool Value::operator==(const Value &other) const {
  return impl_ == other.impl_;
}

bool Value::operator!=(const Value &other) const {
  return impl_ != other.impl_;
}

bool Value::operator!() const { return impl_ == nullptr; }

Value::operator bool() const { return impl_; }

detail::ValueImpl *Value::impl() const { return impl_; }

ir::Type Value::type() const { return impl_->type(); }

void Value::SetType(ir::Type type) { impl_->SetType(type); }

Operation *Value::GetDefiningOp() const {
  if (auto result = dyn_cast<OpResult>()) return result.owner();
  return nullptr;
}

std::string Value::print_ud_chain() { return impl_->print_ud_chain(); }

Value::use_iterator Value::begin() const {
  return ir::OpOperand(impl_->first_use());
}

Value::use_iterator Value::end() const { return Value::use_iterator(); }

// OpResult
bool OpResult::classof(Value value) {
  return ir::isa<detail::OpResultImpl>(value.impl());
}

Operation *OpResult::owner() const { return impl()->owner(); }

uint32_t OpResult::GetResultIndex() const { return impl()->GetResultIndex(); }

detail::OpResultImpl *OpResult::impl() const {
  return reinterpret_cast<detail::OpResultImpl *>(impl_);
}

uint32_t OpResult::GetValidInlineIndex(uint32_t index) {
  uint32_t max_inline_index =
      ir::detail::OpResultImpl::GetMaxInlineResultIndex();
  return index <= max_inline_index ? index : max_inline_index;
}

// details
namespace detail {
ir::Operation *OpOperandImpl::owner() const { return owner_; }

ir::detail::OpOperandImpl *OpOperandImpl::next_use() { return next_use_; }

OpOperandImpl::OpOperandImpl(ir::Value source, ir::Operation *owner)
    : source_(source), owner_(owner) {
  prev_use_addr_ = source.impl()->first_use_addr();
  next_use_ = source.impl()->first_use();
  if (next_use_) {
    next_use_->prev_use_addr_ = &next_use_;
  }
  source.impl()->SetFirstUse(this);
}

void OpOperandImpl::remove_from_ud_chain() {
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

OpOperandImpl::~OpOperandImpl() { remove_from_ud_chain(); }

uint32_t ValueImpl::index() const {
  uint32_t index =
      reinterpret_cast<uintptr_t>(first_use_offseted_by_index_) & 0x07;
  if (index < 6) return index;
  return reinterpret_cast<OpOutlineResultImpl *>(const_cast<ValueImpl *>(this))
      ->GetResultIndex();
}

std::string ValueImpl::print_ud_chain() {
  std::stringstream result;
  result << "Value[" << this << "] -> ";
  OpOperandImpl *tmp = first_use();
  if (tmp) {
    result << "OpOperand[" << reinterpret_cast<void *>(tmp) << "] -> ";
    while (tmp->next_use() != nullptr) {
      result << "OpOperand[" << reinterpret_cast<void *>(tmp->next_use())
             << "] -> ";
      tmp = tmp->next_use();
    }
  }
  result << "nullptr";
  return result.str();
}

uint32_t OpResultImpl::GetResultIndex() const {
  if (const auto *outline_result = ir::dyn_cast<OpOutlineResultImpl>(this)) {
    return outline_result->GetResultIndex();
  }
  return ir::dyn_cast<OpInlineResultImpl>(this)->GetResultIndex();
}

ir::Operation *OpResultImpl::owner() const {
  // For inline result, pointer offset index to obtain the address of op.
  if (const auto *result = ir::dyn_cast<OpInlineResultImpl>(this)) {
    result += result->GetResultIndex() + 1;
    return reinterpret_cast<Operation *>(
        const_cast<OpInlineResultImpl *>(result));
  }
  // For outline result, pointer offset outline_index to obtain the address of
  // maximum inline result.
  const OpOutlineResultImpl *outline_result =
      (const OpOutlineResultImpl *)(this);
  outline_result +=
      (outline_result->outline_index_ - GetMaxInlineResultIndex());
  // The offset of the maximum inline result distance op is
  // GetMaxInlineResultIndex.
  const auto *inline_result =
      reinterpret_cast<const OpInlineResultImpl *>(outline_result);
  inline_result += (GetMaxInlineResultIndex() + 1);
  return reinterpret_cast<Operation *>(
      const_cast<OpInlineResultImpl *>(inline_result));
}
}  // namespace detail
}  // namespace ir
