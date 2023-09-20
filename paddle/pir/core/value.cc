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

#include "paddle/pir/core/value.h"

#include <cstddef>

#include "paddle/pir/core/enforce.h"
#include "paddle/pir/core/op_operand.h"
#include "paddle/pir/core/op_result.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/value_impl.h"

#define CHECK_NULL_IMPL(class_name, func_name)                  \
  IR_ENFORCE(impl_,                                             \
             "impl_ pointer is null when call func:" #func_name \
             " , in class: " #class_name ".")

#define CHECK_VALUE_NULL_IMPL(func_name) CHECK_NULL_IMPL(Value, func_name)

namespace pir {
bool Value::operator==(const Value &other) const {
  return impl_ == other.impl_;
}

bool Value::operator!=(const Value &other) const {
  return impl_ != other.impl_;
}

bool Value::operator!() const { return impl_ == nullptr; }

Value::operator bool() const { return impl_; }

pir::Type Value::type() const {
  CHECK_VALUE_NULL_IMPL(type);
  return impl_->type();
}

void Value::set_type(pir::Type type) {
  CHECK_VALUE_NULL_IMPL(set_type);
  impl_->set_type(type);
}

std::string Value::PrintUdChain() {
  CHECK_VALUE_NULL_IMPL(PrintUdChain);
  return impl()->PrintUdChain();
}

Value::UseIterator Value::use_begin() const { return OpOperand(first_use()); }

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

}  // namespace pir
