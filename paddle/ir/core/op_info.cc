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

#include "paddle/ir/core/op_info.h"
#include "paddle/ir/core/dialect.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/op_info_impl.h"

namespace ir {
bool OpInfo::HasTrait(TypeId trait_id) const {
  return impl_ && impl_->HasTrait(trait_id);
}

bool OpInfo::HasInterface(TypeId interface_id) const {
  return impl_ && impl_->HasInterface(interface_id);
}

IrContext *OpInfo::ir_context() const {
  return impl_ ? impl_->ir_context() : nullptr;
}
Dialect *OpInfo::dialect() const { return impl_ ? impl_->dialect() : nullptr; }

const char *OpInfo::name() const { return impl_ ? impl_->name() : nullptr; }

TypeId OpInfo::id() const { return impl_ ? impl_->id() : TypeId(); }

void OpInfo::Verify(const std::vector<OpResult> &inputs,
                    const std::vector<Type> &outputs,
                    const AttributeMap &attributes) {
  impl_->verify()(inputs, outputs, attributes);
}

void *OpInfo::GetInterfaceImpl(TypeId interface_id) const {
  return impl_ ? impl_->GetInterfaceImpl(interface_id) : nullptr;
}
}  // namespace ir
