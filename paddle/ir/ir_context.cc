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

#include <unordered_map>

#include "paddle/ir/ir_context.h"
#include "paddle/ir/type/type_support.h"
#include "paddle/phi/api/ext/exception.h"

namespace ir {
/// The implementation class of the IrContext class
class IrContextImpl {
 public:
  IrContextImpl() {}

  ~IrContextImpl() {}

  /// \brief Cached AbstractType instances.
  std::unordered_map<TypeId, AbstractType *> registed_abstract_types_;

  /// \brief TypeStorage uniquer and cache instances.
  StorageUniquer registed_storage_uniquer_;

  // TODO(zhangbo9674): add some built type.
};

IrContext::IrContext() : impl_(new IrContextImpl()) {}

StorageUniquer &IrContext::storage_uniquer() {
  return impl().registed_storage_uniquer_;
}

const AbstractType &AbstractType::lookup(TypeId type_id, IrContext *ctx) {
  auto &impl = ctx->impl();
  auto iter = impl.registed_abstract_types_.find(type_id);
  if (iter == impl.registed_abstract_types_.end()) {
    PD_THROW("The input data pointer is null.");
  } else {
    return *(iter->second);
  }
}

}  // namespace ir
