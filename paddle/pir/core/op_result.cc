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

#define CHECK_OPRESULT_NULL_IMPL(func_name) \
  IR_ENFORCE(impl_, "impl_ pointer is null when call OpResult::" #func_name)
#define IMPL_ static_cast<detail::OpResultImpl *>(impl_)

namespace pir {
// OpResult
bool OpResult::classof(Value value) {
  return value && detail::OpResultImpl::classof(*value.impl());
}

Operation *OpResult::owner() const {
  CHECK_OPRESULT_NULL_IMPL(owner);
  return IMPL_->owner();
}

uint32_t OpResult::index() const {
  CHECK_OPRESULT_NULL_IMPL(index);
  return IMPL_->index();
}

OpResult OpResult::dyn_cast_from(Value value) {
  if (classof(value)) {
    return static_cast<detail::OpResultImpl *>(value.impl());
  } else {
    return nullptr;
  }
}

bool OpResult::operator==(const OpResult &other) const {
  return impl_ == other.impl_;
}

OpResult::OpResult(detail::OpResultImpl *impl) : Value(impl) {}

}  // namespace pir
