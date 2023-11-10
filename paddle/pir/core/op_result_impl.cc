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
#include "paddle/pir/core/operation.h"

namespace pir {
namespace detail {

uint32_t OpResultImpl::index() const {
  if (const auto *outline_result = dyn_cast<OpOutlineResultImpl>(this)) {
    return outline_result->index();
  }
  return static_cast<const OpInlineResultImpl *>(this)->index();
}

OpResultImpl::~OpResultImpl() {
  if (!use_empty()) {
    LOG(FATAL) << "Destoryed a op_result that is still in use. \n"
               << "The owner op type is:" << owner()->name();
  }
}

Operation *OpResultImpl::owner() {
  // For inline result, pointer offset index to obtain the address of op.
  if (auto *result = dyn_cast<OpInlineResultImpl>(this)) {
    result += result->index() + 1;
    return reinterpret_cast<Operation *>(result);
  }
  // For outline result, pointer offset outline_index to obtain the address of
  // maximum inline result.
  auto *outline_result = static_cast<OpOutlineResultImpl *>(this);
  outline_result += (outline_result->index() - MAX_INLINE_RESULT_IDX);
  // The offset of the maximum inline result distance op is
  // GetMaxInlineResultIndex.
  auto *inline_result = reinterpret_cast<OpInlineResultImpl *>(outline_result);
  inline_result += OUTLINE_RESULT_IDX;
  return reinterpret_cast<Operation *>(inline_result);
}

}  // namespace detail
}  // namespace pir
