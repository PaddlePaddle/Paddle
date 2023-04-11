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

namespace ir {
Operation *Value::GetDefiningOp() const {
  if (auto result = dyn_cast<OpResult>()) return result.owner();
  return nullptr;
}

uint32_t OpResult::GetValidInlineIndex(uint32_t index) {
  uint32_t max_inline_index =
      ir::detail::OpResultImpl::GetMaxInlineResultIndex();
  return index <= max_inline_index ? index : max_inline_index;
}

std::string Value::print_ud_chain() { return impl_->print_ud_chain(); }

namespace detail {
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
