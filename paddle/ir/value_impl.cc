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

#include "paddle/ir/value_impl.h"

namespace ir {
namespace detail {

uint32_t OpResultImpl::GetResultIndex() const {
  if (const auto *outline_result = ir::dyn_cast<OpOutlineResultImpl>(this)) {
    return outline_result->GetResultIndex();
  }
  return ir::dyn_cast<OpInlineResultImpl>(this)->GetResultIndex();
}

ir::Operation *OpResultImpl::owner() const {
  if (const auto *result = ir::dyn_cast<OpInlineResultImpl>(this)) {
    result += result->GetResultIndex() + 1;
    return reinterpret_cast<Operation *>(
        const_cast<OpInlineResultImpl *>(result));
  }
  // Out-of-line results are stored in an array just before the inline results.
  const OpOutlineResultImpl *outOfLineIt = (const OpOutlineResultImpl *)(this);
  outOfLineIt += (outOfLineIt->outline_index_ + 1);

  // Move the owner past the inline results to get to the operation.
  const auto *inlineIt =
      reinterpret_cast<const OpInlineResultImpl *>(outOfLineIt);
  inlineIt += GetMaxInlineResultIndex();
  return reinterpret_cast<Operation *>(
      const_cast<OpInlineResultImpl *>(inlineIt));
}

}  // namespace detail
}  // namespace ir
