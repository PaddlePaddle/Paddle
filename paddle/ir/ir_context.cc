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
#include "paddle/ir/rw_lock.h"
#include "paddle/ir/type_base.h"

namespace ir {
// The implementation class of the IrContext class
class IrContextImpl {
 public:
  IrContextImpl() {}

  ~IrContextImpl() {}

  void RegisterAbstractType(ir::TypeId type_id, AbstractType *abstract_type) {
    ir::AutoWRLock lock(&registed_abstract_types_rw_lock_);
    registed_abstract_types_.emplace(type_id, abstract_type);
  }

  AbstractType *lookup(ir::TypeId type_id) {
    ir::AutoRDLock lock(&registed_abstract_types_rw_lock_);
    auto iter = registed_abstract_types_.find(type_id);
    if (iter == registed_abstract_types_.end()) {
      return nullptr;
    } else {
      return iter->second;
    }
  }

  ir::RWLock registed_abstract_types_rw_lock_;

  // Cached AbstractType instances.
  std::unordered_map<TypeId, AbstractType *> registed_abstract_types_;

  // TypeStorage uniquer and cache instances.
  StorageManager registed_storage_manager_;

  // Some built-in type.
  Float32Type fp32_type;
  IntegerType int1Ty;
};

IrContext *IrContext::ir_context_ = nullptr;

IrContext::IrContext() : impl_(new IrContextImpl()) {
  VLOG(4) << "==> Register Float32Type.";
  AbstractType *fp32_abstract_type = new AbstractType(
      std::move(AbstractType::get(TypeId::get<Float32Type>())));
  impl_->RegisterAbstractType(TypeId::get<Float32Type>(), fp32_abstract_type);
  TypeManager::RegisterType<Float32Type>(this);
  impl_->fp32_type = TypeManager::get<Float32Type>(this);

  VLOG(4) << "==> Register IntegerType.";
  AbstractType *int_abstract_type = new AbstractType(
      std::move(AbstractType::get(TypeId::get<IntegerType>())));
  impl_->RegisterAbstractType(TypeId::get<IntegerType>(), int_abstract_type);
  TypeManager::RegisterType<IntegerType>(this);
  impl_->int1Ty = TypeManager::get<IntegerType>(this, 1, 0);
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

static IntegerType GetCachedIntegerType(unsigned width,
                                        unsigned signedness,
                                        IrContext *context) {
  if (signedness != 0) return IntegerType();

  switch (width) {
    case 1:
      return context->impl().int1Ty;
    default:
      return IntegerType();
  }
}

IntegerType IntegerType::get(ir::IrContext *context,
                             unsigned width,
                             unsigned signedness) {
  if (auto cached = GetCachedIntegerType(width, signedness, context))
    return cached;
  VLOG(4) << "no cache, create a new";
  return IntegerType::create(context, width, signedness);
}

}  // namespace ir
