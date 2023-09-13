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
#include "paddle/pir/core/op_result.h"
#include "paddle/pir/core/enforce.h"
#include "paddle/pir/core/op_result_impl.h"

#define CHECK_NULL_IMPL(class_name, func_name)                  \
  IR_ENFORCE(impl_,                                             \
             "impl_ pointer is null when call func:" #func_name \
             " , in class: " #class_name ".")

#define CHECK_OPRESULT_NULL_IMPL(func_name) CHECK_NULL_IMPL(OpResult, func_name)

namespace pir {

// OpResult
bool OpResult::classof(Value value) {
  return value && pir::isa<detail::OpResultImpl>(value.impl());
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

uint32_t OpResult::GetValidInlineIndex(uint32_t index) {
  uint32_t max_inline_index =
      pir::detail::OpResultImpl::GetMaxInlineResultIndex();
  return index <= max_inline_index ? index : max_inline_index;
}

}  // namespace pir
