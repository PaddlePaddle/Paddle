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

#include "paddle/ir/builtin_type.h"
#include "paddle/ir/ir_context.h"
#include "paddle/ir/spin_lock.h"
#include "paddle/ir/type_base.h"

namespace ir {
// The implementation class of the IrContext class
class IrContextImpl {
 public:
  IrContextImpl() {}

  ~IrContextImpl() {
    std::lock_guard<ir::SpinLock> guard(registed_abstract_types_lock_);
    for (auto abstract_type_map : registed_abstract_types_) {
      delete abstract_type_map.second;
    }
    registed_abstract_types_.clear();
  }

  void RegisterAbstractType(ir::TypeId type_id, AbstractType *abstract_type) {
    std::lock_guard<ir::SpinLock> guard(registed_abstract_types_lock_);
    registed_abstract_types_.emplace(type_id, abstract_type);
  }

  AbstractType *lookup(ir::TypeId type_id) {
    std::lock_guard<ir::SpinLock> guard(registed_abstract_types_lock_);
    auto iter = registed_abstract_types_.find(type_id);
    if (iter == registed_abstract_types_.end()) {
      return nullptr;
    } else {
      return iter->second;
    }
  }

  ir::SpinLock registed_abstract_types_lock_;

  // Cached AbstractType instances.
  std::unordered_map<TypeId, AbstractType *> registed_abstract_types_;

  // TypeStorage uniquer and cache instances.
  StorageManager registed_storage_manager_;

  // Some built-in type.
  Float32Type fp32_type;
  Int32Type int32_type;
};

IrContext &IrContext::Instance() {
  static IrContext context;
  return context;
}

IrContext::IrContext() : impl_(new IrContextImpl()) {
  REGISTER_TYPE_2_IRCONTEXT(Float32Type, this);
  impl_->fp32_type = TypeManager::get<Float32Type>(this);

  REGISTER_TYPE_2_IRCONTEXT(Int32Type, this);
  impl_->int32_type = TypeManager::get<Int32Type>(this);
}

void IrContext::RegisterAbstractType(ir::TypeId type_id,
                                     AbstractType *abstract_type) {
  impl().RegisterAbstractType(type_id, abstract_type);
}

StorageManager &IrContext::storage_manager() {
  return impl().registed_storage_manager_;
}

std::unordered_map<TypeId, AbstractType *>
    &IrContext::registed_abstracted_type() {
  return impl().registed_abstract_types_;
}

const AbstractType &AbstractType::lookup(TypeId type_id, IrContext *ctx) {
  VLOG(4) << "==> Get registed abstract type (" << &type_id
          << ") from IrContext (" << ctx << ").";
  auto &impl = ctx->impl();
  AbstractType *abstract_type = impl.lookup(type_id);
  if (abstract_type) {
    return *abstract_type;
  } else {
    throw("Abstract type not found in IrContext.");
  }
}

Float32Type Float32Type::get(IrContext *ctx) { return ctx->impl().fp32_type; }

Int32Type Int32Type::get(IrContext *ctx) { return ctx->impl().int32_type; }

}  // namespace ir
