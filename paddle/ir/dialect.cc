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

#include "paddle/ir/dialect.h"

namespace ir {
Dialect::Dialect(std::string name, ir::IrContext *context, ir::TypeId id)
    : name_(std::move(name)), context_(context), id_(id) {}

void Dialect::RegisterType(ir::AbstractType &&abstract_type) {
  ir::AbstractType *new_abstract_type =
      new ir::AbstractType(std::move(abstract_type));
  this->ir_context()->RegisterAbstractType(new_abstract_type->type_id(),
                                           new_abstract_type);
}

void Dialect::RegisterAttribute(ir::AbstractAttribute &&abstract_attribute) {
  ir::AbstractAttribute *new_abstract_attribute =
      new ir::AbstractAttribute(std::move(abstract_attribute));
  this->ir_context()->RegisterAbstractAttribute(
      new_abstract_attribute->type_id(), new_abstract_attribute);
}

void Dialect::RegisterOp(const std::string &name, OpInfoImpl *op_info) {
  this->ir_context()->RegisterOpInfo(name, op_info);
}
}  // namespace ir
