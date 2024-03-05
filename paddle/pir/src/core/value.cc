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

#include "paddle/pir/include/core/value.h"

#include <cstddef>

#include "paddle/common/enforce.h"
#include "paddle/pir/include/core/op_operand.h"
#include "paddle/pir/include/core/op_result.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/src/core/value_impl.h"

#define CHECK_NULL_IMPL(class_name, func_name)                  \
  IR_ENFORCE(impl_,                                             \
             "impl_ pointer is null when call func:" #func_name \
             " , in class: " #class_name ".")

#define CHECK_VALUE_NULL_IMPL(func_name) CHECK_NULL_IMPL(Value, func_name)

namespace pir {
bool Value::operator==(const Value &other) const {
  return impl_ == other.impl_ &&
         (impl_ == nullptr || impl_->id() == other.impl_->id());
}

bool Value::operator!=(const Value &other) const {
  return !(operator==(other));
}

bool Value::operator!() const { return impl_ == nullptr; }

bool Value::operator<(const Value &other) const { return impl_ < other.impl_; }

Value::operator bool() const { return impl_; }

pir::Type Value::type() const { return impl_ ? impl_->type() : nullptr; }

Operation *Value::defining_op() const { return dyn_cast<OpResult>().owner(); }

void Value::set_type(pir::Type type) {
  CHECK_VALUE_NULL_IMPL(set_type);
  impl_->set_type(type);
}

void Value::set_impl(detail::ValueImpl *impl) { impl_ = impl; }

std::string Value::PrintUdChain() {
  CHECK_VALUE_NULL_IMPL(PrintUdChain);
  return impl()->PrintUdChain();
}

Value::UseIterator Value::use_begin() const { return OpOperand(first_use()); }

Value::UseIterator Value::use_end() const { return Value::UseIterator(); }

OpOperand Value::first_use() const {
  return impl_ ? impl_->first_use() : nullptr;
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
    } else {
      it++;
    }
  }
}

void Value::ReplaceAllUsesWith(Value new_value) const {
  for (auto it = use_begin(); it != use_end();) {
    (it++)->set_source(new_value);
  }
}

Attribute Value::attribute(const std::string &key) const {
  auto op_result = dyn_cast<OpResult>();
  if (op_result) return op_result.attribute(key);
  return dyn_cast<BlockArgument>().attribute(key);
}

void Value::set_attribute(const std::string &key, Attribute value) {
  auto op_result = dyn_cast<OpResult>();
  if (op_result) return op_result.set_attribute(key, value);
  return dyn_cast<BlockArgument>().set_attribute(key, value);
}

}  // namespace pir
