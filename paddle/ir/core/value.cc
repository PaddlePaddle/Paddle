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

#include "paddle/ir/core/value.h"
#include "paddle/ir/core/enforce.h"
#include "paddle/ir/core/value_impl.h"

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
OpOperand::operator bool() const { return impl_ && impl_->source(); }

OpOperand OpOperand::next_use() const { return impl()->next_use(); }

Value OpOperand::source() const { return impl()->source(); }

Type OpOperand::type() const { return source().type(); }

void OpOperand::set_source(Value value) { impl()->set_source(value); }

Operation *OpOperand::owner() const { return impl()->owner(); }

void OpOperand::RemoveFromUdChain() { return impl()->RemoveFromUdChain(); }

detail::OpOperandImpl *OpOperand::impl() const {
  IR_ENFORCE(impl_, "Can't use impl() interface while op_operand is null.");
  return impl_;
}
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

ir::Type Value::type() const { return impl()->type(); }

void Value::set_type(ir::Type type) { impl()->set_type(type); }

Operation *Value::GetDefiningOp() const {
  if (auto result = dyn_cast<OpResult>()) return result.owner();
  return nullptr;
}

std::string Value::PrintUdChain() { return impl()->PrintUdChain(); }

Value::use_iterator Value::begin() const { return ir::OpOperand(first_use()); }

Value::use_iterator Value::end() const { return Value::use_iterator(); }

OpOperand Value::first_use() const { return impl()->first_use(); }

bool Value::use_empty() const { return !first_use(); }

bool Value::HasOneUse() const { return impl()->HasOneUse(); }

void Value::ReplaceUsesWithIf(
    Value new_value,
    const std::function<bool(OpOperand)> &should_replace) const {
  for (auto it = begin(); it != end();) {
    if (should_replace(*it)) {
      (it++)->set_source(new_value);
    }
  }
}

void Value::ReplaceAllUsesWith(Value new_value) const {
  for (auto it = begin(); it != end();) {
    (it++)->set_source(new_value);
  }
}

detail::ValueImpl *Value::impl() const {
  IR_ENFORCE(impl_, "Can't use impl() interface while value is null.");
  return impl_;
}

// OpResult
bool OpResult::classof(Value value) {
  return value && ir::isa<detail::OpResultImpl>(value.impl());
}

Operation *OpResult::owner() const { return impl()->owner(); }

uint32_t OpResult::GetResultIndex() const { return impl()->GetResultIndex(); }

detail::OpResultImpl *OpResult::impl() const {
  IR_ENFORCE(impl_, "Can't use impl() interface while value is null.");
  return reinterpret_cast<detail::OpResultImpl *>(impl_);
}

bool OpResult::operator==(const OpResult &other) const {
  return impl_ == other.impl_;
}

detail::ValueImpl *OpResult::value_impl() const {
  IR_ENFORCE(impl_, "Can't use value_impl() interface while value is null.");
  return impl_;
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

ir::Value OpOperandImpl::source() const { return source_; }

void OpOperandImpl::set_source(Value source) {
  RemoveFromUdChain();
  if (!source) {
    return;
  }
  source_ = source;
  InsertToUdChain();
}

OpOperandImpl::OpOperandImpl(ir::Value source, ir::Operation *owner)
    : source_(source), owner_(owner) {
  if (!source) {
    return;
  }
  InsertToUdChain();
}

void OpOperandImpl::InsertToUdChain() {
  prev_use_addr_ = source_.impl()->first_use_addr();
  next_use_ = source_.impl()->first_use();
  if (next_use_) {
    next_use_->prev_use_addr_ = &next_use_;
  }
  source_.impl()->SetFirstUse(this);
}

void OpOperandImpl::RemoveFromUdChain() {
  if (!source_) return;
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
  next_use_ = nullptr;
  prev_use_addr_ = nullptr;
  source_ = nullptr;
}

OpOperandImpl::~OpOperandImpl() { RemoveFromUdChain(); }

uint32_t ValueImpl::index() const {
  uint32_t index =
      reinterpret_cast<uintptr_t>(first_use_offseted_by_index_) & 0x07;
  if (index < 6) return index;
  return reinterpret_cast<OpOutlineResultImpl *>(const_cast<ValueImpl *>(this))
      ->GetResultIndex();
}

std::string ValueImpl::PrintUdChain() {
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
