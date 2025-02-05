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

#include <glog/logging.h>

#include "paddle/common/enforce.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/src/core/op_result_impl.h"

namespace pir::detail {

uint32_t OpResultImpl::index() const {
  if (const auto *outline_result = dyn_cast<OpOutlineResultImpl>(this)) {
    return outline_result->index();
  }
  return static_cast<const OpInlineResultImpl *>(this)->index();
}

OpResultImpl::~OpResultImpl() {
  if (!use_empty()) {
    PADDLE_FATAL(
        "Destroyed a op_result that is still in use by %d. The owner op type "
        "is: %s",
        first_use()->owner()->name(),
        owner()->name());
  }
}

int32_t OpResultImpl::ComputeOperationOffset() const {
  auto kind = this->kind();
  // Compute inline op result offset.
  if (kind < OUTLINE_RESULT_IDX) {
    return static_cast<int32_t>((kind + 1u) * sizeof(OpInlineResultImpl));
  }
  // Compute outline op result offset.
  constexpr int32_t outline_size =
      static_cast<int32_t>(sizeof(OpOutlineResultImpl));
  constexpr int32_t inline_size =
      static_cast<int32_t>(sizeof(OpInlineResultImpl));
  constexpr int32_t diff = OUTLINE_RESULT_IDX * (outline_size - inline_size);

  auto index = static_cast<const OpOutlineResultImpl *>(this)->index();

  return static_cast<int32_t>(index + 1) * outline_size - diff;
}

const Operation *OpResultImpl::owner() const {
  int32_t offset = ComputeOperationOffset();
  return reinterpret_cast<const Operation *>(
      reinterpret_cast<const char *>(this) + offset);
}

Operation *OpResultImpl::owner() {
  int32_t offset = ComputeOperationOffset();
  return reinterpret_cast<Operation *>(reinterpret_cast<char *>(this) + offset);
}

Attribute OpResultImpl::attribute(const std::string &key) const {
  auto array = owner()->attribute<ArrayAttribute>(key);
  auto index = this->index();
  return array && array.size() > index ? array[index] : nullptr;
}

void OpResultImpl::set_attribute(const std::string &key, Attribute value) {
  auto owner = this->owner();
  auto attr = owner->attribute(key);
  PADDLE_ENFORCE_EQ(attr && !attr.isa<ArrayAttribute>(),
                    false,
                    common::errors::PreconditionNotMet(
                        "The %s attribute has existed as operation attribute. "
                        "Can't set it as value attribute. ",
                        key));
  auto array_attr = attr.dyn_cast<ArrayAttribute>();
  auto index = this->index();
  std::vector<Attribute> vec;
  if (array_attr) vec = array_attr.AsVector();
  vec.resize(owner->num_results());
  vec[index] = value;
  owner->set_attribute(key, ArrayAttribute::get(owner->ir_context(), vec));
}

void *OpResultImpl::property(const std::string &key) const {
  return owner()->value_property(key, index());
}

void OpResultImpl::set_property(const std::string &key, const Property &value) {
  auto owner = this->owner();
  owner->set_value_property(key, value, index());
}

OpInlineResultImpl::OpInlineResultImpl(Type type, uint32_t result_index)
    : OpResultImpl(type, result_index) {
  PADDLE_ENFORCE_LE(
      result_index,
      MAX_INLINE_RESULT_IDX,
      common::errors::PreconditionNotMet(
          "Inline result index [%u] should not exceed MaxInlineResultIndex(5)",
          result_index));
}

}  // namespace pir::detail
