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

#include "paddle/pir/include/core/type.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/core/type_base.h"

namespace pir {
IrContext *Type::ir_context() const { return dialect().ir_context(); }

TypeId Type::type_id() { return storage_->abstract_type().type_id(); }

const AbstractType &Type::abstract_type() { return storage_->abstract_type(); }

Dialect &Type::dialect() const { return storage_->abstract_type().dialect(); }

bool Type::IsIntOrIndex() const {
  return isa<IndexType>() || isa<Int8Type>() || isa<UInt8Type>() ||
         isa<Int16Type>() || isa<Int32Type>() || isa<Int64Type>();
}

bool Type::IsIndex() const { return isa<IndexType>(); }

}  // namespace pir
