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
#include "paddle/pir/core/op_operand.h"
#include "paddle/pir/core/enforce.h"
#include "paddle/pir/core/op_operand_impl.h"

#define CHECK_NULL_IMPL(class_name, func_name)                  \
  IR_ENFORCE(impl_,                                             \
             "impl_ pointer is null when call func:" #func_name \
             " , in class: " #class_name ".")

#define CHECK_OPOPEREND_NULL_IMPL(func_name) \
  CHECK_NULL_IMPL(OpOpernad, func_name)

namespace pir {
OpOperand &OpOperand::operator=(const OpOperand &rhs) {
  impl_ = rhs.impl_;
  return *this;
}

OpOperand::operator bool() const { return impl_ && impl_->source(); }

OpOperand OpOperand::next_use() const {
  CHECK_OPOPEREND_NULL_IMPL(next_use);
  return impl_->next_use();
}

Value OpOperand::source() const {
  CHECK_OPOPEREND_NULL_IMPL(source);
  return impl_->source();
}

Type OpOperand::type() const { return source().type(); }

void OpOperand::set_source(Value value) {
  CHECK_OPOPEREND_NULL_IMPL(set_source);
  impl_->set_source(value);
}

Operation *OpOperand::owner() const {
  CHECK_OPOPEREND_NULL_IMPL(owner);
  return impl_->owner();
}

void OpOperand::RemoveFromUdChain() {
  CHECK_OPOPEREND_NULL_IMPL(RemoveFromUdChain);
  return impl_->RemoveFromUdChain();
}

}  // namespace pir
