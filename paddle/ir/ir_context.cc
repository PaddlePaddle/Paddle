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

#include "paddle/ir/ir_context.h"

#include <unordered_map>

#include "paddle/ir/attribute_base.h"
#include "paddle/ir/builtin_dialect.h"
#include "paddle/ir/builtin_type.h"
#include "paddle/ir/dialect.h"
#include "paddle/ir/op_info_impl.h"
#include "paddle/ir/spin_lock.h"
#include "paddle/ir/type_base.h"

namespace ir {
// The implementation class of the IrContext class, cache registered
// AbstractType, TypeStorage, AbstractAttribute, AttributeStorage, Dialect.
class IrContextImpl {
 public:
  IrContextImpl() {}

  ~IrContextImpl() {
    std::lock_guard<ir::SpinLock> guard(destructor_lock_);
    for (auto &abstract_type_map : registed_abstract_types_) {
      delete abstract_type_map.second;
    }
    registed_abstract_types_.clear();

    for (auto &abstract_attribute_map : registed_abstract_attributes_) {
      delete abstract_attribute_map.second;
    }
    registed_abstract_attributes_.clear();

    for (auto &dialect_map : registed_dialect_) {
      delete dialect_map.second;
    }
    registed_dialect_.clear();

    for (auto &op_map : registed_op_infos_) {
      op_map.second->destroy();
    }
    registed_op_infos_.clear();
  }

  void RegisterAbstractType(ir::TypeId type_id, AbstractType *abstract_type) {
    std::lock_guard<ir::SpinLock> guard(registed_abstract_types_lock_);
    VLOG(4) << "Register an abstract_type of: [TypeId_hash="
            << std::hash<ir::TypeId>()(type_id)
            << ", AbstractType_ptr=" << abstract_type << "].";
    registed_abstract_types_.emplace(type_id, abstract_type);
  }

  AbstractType *GetAbstractType(ir::TypeId type_id) {
    std::lock_guard<ir::SpinLock> guard(registed_abstract_types_lock_);
    auto iter = registed_abstract_types_.find(type_id);
    if (iter != registed_abstract_types_.end()) {
      VLOG(4) << "Fonund a cached abstract_type of: [TypeId_hash="
              << std::hash<ir::TypeId>()(type_id)
              << ", AbstractType_ptr=" << iter->second << "].";
      return iter->second;
    }
    LOG(WARNING) << "No cache found abstract_type of: [TypeId_hash="
                 << std::hash<ir::TypeId>()(type_id) << "].";
    return nullptr;
  }

  void RegisterAbstractAttribute(ir::TypeId type_id,
                                 AbstractAttribute *abstract_attribute) {
    std::lock_guard<ir::SpinLock> guard(registed_abstract_attributes_lock_);
    VLOG(4) << "Register an abstract_attribute of: [TypeId_hash="
            << std::hash<ir::TypeId>()(type_id)
            << ", AbstractAttribute_ptr=" << abstract_attribute << "].";
    registed_abstract_attributes_.emplace(type_id, abstract_attribute);
  }

  AbstractAttribute *GetAbstractAttribute(ir::TypeId type_id) {
    std::lock_guard<ir::SpinLock> guard(registed_abstract_attributes_lock_);
    auto iter = registed_abstract_attributes_.find(type_id);
    if (iter != registed_abstract_attributes_.end()) {
      VLOG(4) << "Fonund a cached abstract_attribute of: [TypeId_hash="
              << std::hash<ir::TypeId>()(type_id)
              << ", AbstractAttribute_ptr=" << iter->second << "].";
      return iter->second;
    }
    LOG(WARNING) << "No cache found abstract_attribute of: [TypeId_hash="
                 << std::hash<ir::TypeId>()(type_id) << "].";
    return nullptr;
  }

  void RegisterOpInfo(const std::string &name, OpInfoImpl *opinfo) {
    std::lock_guard<ir::SpinLock> guard(registed_op_infos_lock_);
    VLOG(4) << "Register an operation of: [Name=" << name
            << ", OpInfoImpl ptr=" << opinfo << "].";
    registed_op_infos_.emplace(name, opinfo);
  }

  OpInfoImpl *GetOpInfo(const std::string &name) {
    std::lock_guard<ir::SpinLock> guard(registed_op_infos_lock_);
    auto iter = registed_op_infos_.find(name);
    if (iter != registed_op_infos_.end()) {
      VLOG(4) << "Fonund a cached operation of: [name=" << name
              << ", OpInfoImpl ptr=" << iter->second << "].";
      return iter->second;
    }
    LOG(WARNING) << "No cache found operation of: [Name=" << name << "].";
    return nullptr;
  }

  void RegisterDialect(std::string name, Dialect *dialect) {
    std::lock_guard<ir::SpinLock> guard(registed_dialect_lock_);
    VLOG(4) << "Register a dialect of: [name=" << name
            << ", dialect_ptr=" << dialect << "].";
    registed_dialect_.emplace(name, dialect);
  }

  Dialect *GetDialect(std::string name) {
    std::lock_guard<ir::SpinLock> guard(registed_dialect_lock_);
    auto iter = registed_dialect_.find(name);
    if (iter != registed_dialect_.end()) {
      VLOG(4) << "Fonund a cached dialect of: [name=" << name
              << ", dialect_ptr=" << iter->second << "].";
      return iter->second;
    }
    LOG(WARNING) << "No cache fonund dialect of: [name=" << name << "].";
    return nullptr;
  }

  // Cached AbstractType instances.
  std::unordered_map<TypeId, AbstractType *> registed_abstract_types_;
  ir::SpinLock registed_abstract_types_lock_;
  // TypeStorage uniquer and cache instances.
  StorageManager registed_type_storage_manager_;
  // Cache some built-in type objects.
  Float16Type fp16_type;
  Float32Type fp32_type;
  Float64Type fp64_type;
  Int16Type int16_type;
  Int32Type int32_type;
  Int64Type int64_type;

  // Cached AbstractAttribute instances.
  std::unordered_map<TypeId, AbstractAttribute *> registed_abstract_attributes_;
  ir::SpinLock registed_abstract_attributes_lock_;
  // AttributeStorage uniquer and cache instances.
  StorageManager registed_attribute_storage_manager_;

  // The dialcet registered in the context.
  std::unordered_map<std::string, Dialect *> registed_dialect_;
  ir::SpinLock registed_dialect_lock_;

  // The Op registered in the context.
  std::unordered_map<std::string, OpInfoImpl *> registed_op_infos_;
  ir::SpinLock registed_op_infos_lock_;

  ir::SpinLock destructor_lock_;
};

IrContext *IrContext::Instance() {
  static IrContext context;
  return &context;
}

IrContext::IrContext() : impl_(new IrContextImpl()) {
  VLOG(4) << "BuiltinDialect registered into IrContext. ===>";
  GetOrRegisterDialect<BuiltinDialect>();
  VLOG(4) << "==============================================";

  impl_->fp16_type = TypeManager::get<Float16Type>(this);
  impl_->fp32_type = TypeManager::get<Float32Type>(this);
  impl_->fp64_type = TypeManager::get<Float64Type>(this);
  impl_->int16_type = TypeManager::get<Int16Type>(this);
  impl_->int32_type = TypeManager::get<Int32Type>(this);
  impl_->int64_type = TypeManager::get<Int64Type>(this);
}

void IrContext::RegisterAbstractType(ir::TypeId type_id,
                                     AbstractType *abstract_type) {
  impl().RegisterAbstractType(type_id, abstract_type);
}

StorageManager &IrContext::type_storage_manager() {
  return impl().registed_type_storage_manager_;
}

AbstractType *IrContext::GetRegisteredAbstractType(TypeId id) {
  auto search = impl().registed_abstract_types_.find(id);
  if (search != impl().registed_abstract_types_.end()) {
    return search->second;
  }
  return nullptr;
}

void IrContext::RegisterAbstractAttribute(
    ir::TypeId type_id, AbstractAttribute *abstract_attribute) {
  impl().RegisterAbstractAttribute(type_id, abstract_attribute);
}

StorageManager &IrContext::attribute_storage_manager() {
  return impl().registed_attribute_storage_manager_;
}

AbstractAttribute *IrContext::GetRegisteredAbstractAttribute(TypeId id) {
  auto search = impl().registed_abstract_attributes_.find(id);
  if (search != impl().registed_abstract_attributes_.end()) {
    return search->second;
  }
  return nullptr;
}

Dialect *IrContext::GetOrRegisterDialect(
    std::string dialect_name, std::function<Dialect *()> constructor) {
  VLOG(4) << "Try to get or register a Dialect of: [name=" << dialect_name
          << "].";
  Dialect *dialect = impl().GetDialect(dialect_name);
  if (dialect == nullptr) {
    VLOG(4) << "Create and register a new Dialect of: [name=" << dialect_name
            << "].";
    dialect = constructor();
    impl().RegisterDialect(dialect_name, dialect);
  }
  return dialect;
}

std::vector<Dialect *> IrContext::GetRegisteredDialects() {
  std::vector<Dialect *> result;
  for (auto dialect_map : impl().registed_dialect_) {
    result.push_back(dialect_map.second);
  }
  return result;
}

Dialect *IrContext::GetRegisteredDialect(const std::string &dialect_name) {
  for (auto dialect_map : impl().registed_dialect_) {
    if (dialect_map.first == dialect_name) {
      return dialect_map.second;
    }
  }
  LOG(WARNING) << "No dialect registered for " << dialect_name;
  return nullptr;
}

OpInfoImpl *IrContext::GetRegisteredOpInfo(const std::string &name) {
  OpInfoImpl *rtn = impl().GetOpInfo(name);
  return rtn ? rtn : nullptr;
}

void IrContext::RegisterOpInfo(const std::string &name, OpInfoImpl *opinfo) {
  if (impl().GetOpInfo(name) == nullptr) {
    impl().RegisterOpInfo(name, opinfo);
  }
}

const AbstractType &AbstractType::lookup(TypeId type_id, IrContext *ctx) {
  auto &impl = ctx->impl();
  AbstractType *abstract_type = impl.GetAbstractType(type_id);
  if (abstract_type) {
    return *abstract_type;
  } else {
    throw("Abstract type not found in IrContext.");
  }
}

const AbstractAttribute &AbstractAttribute::lookup(TypeId type_id,
                                                   IrContext *ctx) {
  auto &impl = ctx->impl();
  AbstractAttribute *abstract_attribute = impl.GetAbstractAttribute(type_id);
  if (abstract_attribute) {
    return *abstract_attribute;
  } else {
    throw("Abstract attribute not found in IrContext.");
  }
}

Float16Type Float16Type::get(IrContext *ctx) { return ctx->impl().fp16_type; }

Float32Type Float32Type::get(IrContext *ctx) { return ctx->impl().fp32_type; }

Float64Type Float64Type::get(IrContext *ctx) { return ctx->impl().fp64_type; }

Int16Type Int16Type::get(IrContext *ctx) { return ctx->impl().int16_type; }

Int32Type Int32Type::get(IrContext *ctx) { return ctx->impl().int32_type; }

Int64Type Int64Type::get(IrContext *ctx) { return ctx->impl().int64_type; }

}  // namespace ir
