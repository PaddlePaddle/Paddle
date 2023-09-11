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

#include <glog/logging.h>
#include <cstddef>

#include "paddle/ir/core/enforce.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/value_impl.h"

#define CHECK_NULL_IMPL(class_name, func_name)                  \
  IR_ENFORCE(impl_,                                             \
             "impl_ pointer is null when call func:" #func_name \
             " , in class: " #class_name ".")

#define CHECK_OPOPEREND_NULL_IMPL(func_name) \
  CHECK_NULL_IMPL(OpOpernad, func_name)

#define CHECK_VALUE_NULL_IMPL(func_name) CHECK_NULL_IMPL(Value, func_name)

#define CHECK_OPRESULT_NULL_IMPL(func_name) CHECK_NULL_IMPL(OpResult, func_name)
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

OpOperand OpOperand::next_use() const {
  CHECK_OPOPEREND_NULL_IMPL(next_use);
  return impl_->next_use();
}

Value OpOperand::source() const {
  CHECK_OPOPEREND_NULL_IMPL(source);
  return impl_->source();
}

Type OpOperand::type() const { return source().type(); }

void OpOperand::set_source(Value value) {
  CHECK_OPOPEREND_NULL_IMPL(set_source);
  impl_->set_source(value);
}

Operation *OpOperand::owner() const {
  CHECK_OPOPEREND_NULL_IMPL(owner);
  return impl_->owner();
}

void OpOperand::RemoveFromUdChain() {
  CHECK_OPOPEREND_NULL_IMPL(RemoveFromUdChain);
  return impl_->RemoveFromUdChain();
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

ir::Type Value::type() const {
  CHECK_VALUE_NULL_IMPL(type);
  return impl_->type();
}

void Value::set_type(ir::Type type) {
  CHECK_VALUE_NULL_IMPL(set_type);
  impl_->set_type(type);
}

Operation *Value::GetDefiningOp() const {
  if (auto result = dyn_cast<OpResult>()) return result.owner();
  return nullptr;
}

std::string Value::PrintUdChain() {
  CHECK_VALUE_NULL_IMPL(PrintUdChain);
  return impl()->PrintUdChain();
}

Value::UseIterator Value::use_begin() const {
  return ir::OpOperand(first_use());
}

Value::UseIterator Value::use_end() const { return Value::UseIterator(); }

OpOperand Value::first_use() const {
  CHECK_VALUE_NULL_IMPL(first_use);
  return impl_->first_use();
}

bool Value::use_empty() const { return !first_use(); }

bool Value::HasOneUse() const {
  CHECK_VALUE_NULL_IMPL(HasOneUse);
  return impl_->HasOneUse();
}

size_t Value::use_count() const {
  size_t count = 0;
  for (auto it = use_begin(); it != use_end(); ++it) count++;
  return count;
}

void Value::ReplaceUsesWithIf(
    Value new_value,
    const std::function<bool(OpOperand)> &should_replace) const {
  for (auto it = use_begin(); it != use_end();) {
    if (should_replace(*it)) {
      (it++)->set_source(new_value);
    }
  }
}

void Value::ReplaceAllUsesWith(Value new_value) const {
  for (auto it = use_begin(); it != use_end();) {
    (it++)->set_source(new_value);
  }
}

// OpResult
bool OpResult::classof(Value value) {
  return value && ir::isa<detail::OpResultImpl>(value.impl());
}

Operation *OpResult::owner() const {
  CHECK_OPRESULT_NULL_IMPL(owner);
  return impl()->owner();
}

uint32_t OpResult::GetResultIndex() const {
  CHECK_OPRESULT_NULL_IMPL(GetResultIndex);
  return impl()->GetResultIndex();
}

detail::OpResultImpl *OpResult::impl() const {
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
  source_.impl()->set_first_use(this);
}

void OpOperandImpl::RemoveFromUdChain() {
  if (!source_) return;
  if (!prev_use_addr_) return;
  if (prev_use_addr_ == source_.impl()->first_use_addr()) {
    /// NOTE: In ValueImpl, first_use_offseted_by_index_ use lower three bits
    /// storage index information, so need to be updated using the set_first_use
    /// method here.
    source_.impl()->set_first_use(next_use_);
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

OpResultImpl::~OpResultImpl() {
  if (!use_empty()) {
    LOG(ERROR) << owner()->name() << " operation destroyed but still has uses.";
  }
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
