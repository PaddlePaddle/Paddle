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
#include "paddle/pir/core/op_result_impl.h"

#include <cassert>

namespace pir {
namespace detail {

uint32_t OpResultImpl::GetResultIndex() const {
  if (const auto *outline_result = dyn_cast<OpOutlineResultImpl>(this)) {
    return outline_result->GetResultIndex();
  }
  return dyn_cast<OpInlineResultImpl>(this)->GetResultIndex();
}

OpResultImpl::~OpResultImpl() { assert(use_empty()); }

Operation *OpResultImpl::owner() const {
  // For inline result, pointer offset index to obtain the address of op.
  if (const auto *result = dyn_cast<OpInlineResultImpl>(this)) {
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
}  // namespace pir
