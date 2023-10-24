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
#include "paddle/pir/core/value_impl.h"

namespace pir {

namespace detail {
void ValueImpl::set_first_use(OpOperandImpl *first_use) {
  uint32_t offset = kind();
  first_use_offseted_by_kind_ = reinterpret_cast<OpOperandImpl *>(
      reinterpret_cast<uintptr_t>(first_use) + offset);
  VLOG(4) << "The index of this value is " << offset
          << ". Offset and set first use: " << first_use << " -> "
          << first_use_offseted_by_kind_ << ".";
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
ValueImpl::ValueImpl(Type type, uint32_t kind) {
  if (kind > BLOCK_ARG_IDX) {
    LOG(FATAL) << "The kind of value_impl(" << kind
               << "), is bigger than BLOCK_ARG_IDX(7)";
  }
  type_ = type;
  first_use_offseted_by_kind_ = reinterpret_cast<OpOperandImpl *>(
      reinterpret_cast<uintptr_t>(nullptr) + kind);
  VLOG(4) << "Construct a ValueImpl whose's kind is " << kind
          << ". The offset first_use address is: "
          << first_use_offseted_by_kind_;
}

}  // namespace detail
}  // namespace pir
