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

#pragma once

#include <functional>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "paddle/pir/include/core/dll_decl.h"

namespace pir {
class IrContextImpl;
class StorageManager;
class AbstractType;
class AbstractAttribute;
class TypeId;
class Dialect;
class OpInfo;
class Type;
class Attribute;
class Operation;
class InterfaceValue;

using OpInfoMap = std::unordered_map<std::string, OpInfo>;

///
/// \brief IrContext is a global parameterless class used to store and manage
/// Type, Attribute and other related data structures.
///
class IR_API IrContext {
 public:
  ///
  /// \brief Initializes a new instance of IrContext.
  ///
  static IrContext *Instance();

  IrContext();
  ~IrContext();

  ///
  /// \brief Get an instance of IrContextImpl, a private member of IrContext.
  /// For the specific definition of IrContextImpl, see ir_context.cc.
  ///
  /// \return The instance of IrContextImpl.
  ///
  IrContextImpl &impl() { return *impl_; }

  ///
  /// \brief Register an AbstractType to IrContext.
  ///
  /// \param type_id The type id of the AbstractType.
  /// \param abstract_type AbstractType* provided by user.
  ///
  void RegisterAbstractType(TypeId type_id, AbstractType &&abstract_type);

  ///
  /// \brief Returns the storage uniquer used for constructing TypeStorage
  /// instances.
  ///
  /// \return The storage uniquer used for constructing TypeStorage
  /// instances.
  ///
  StorageManager &type_storage_manager();

  ///
  /// \brief Get registered AbstractType from IrContext.
  ///
  AbstractType *GetRegisteredAbstractType(TypeId id);

  ///
  /// \brief Register an AbstractAttribute to IrContext
  ///
  /// \param type_id The type id of the AbstractAttribute.
  /// \param abstract_attribute AbstractAttribute provided by user.
  ///
  void RegisterAbstractAttribute(pir::TypeId type_id,
                                 AbstractAttribute &&abstract_attribute);

  ///
  /// \brief Returns the storage uniquer used for constructing AttributeStorage
  /// instances.
  ///
  /// \return The storage uniquer used for constructing AttributeStorage
  /// instances.
  ///
  StorageManager &attribute_storage_manager();

  ///
  /// \brief Get registered AbstractAttribute from IrContext.
  ///
  AbstractAttribute *GetRegisteredAbstractAttribute(TypeId id);

  ///
  /// \brief Register an op infomation to IrContext
  ///
  void RegisterOpInfo(Dialect *dialect,
                      TypeId op_id,
                      const char *name,
                      std::set<InterfaceValue> &&interface_set,
                      const std::vector<TypeId> &trait_set,
                      size_t attributes_num,
                      const char **attributes_name,
                      void (*verify_sig)(Operation *),
                      void (*verify_region)(Operation *));

  ///
  /// \brief Get registered operation infomation.
  ///
  OpInfo GetRegisteredOpInfo(const std::string &name);

  ///
  /// \brief Get registered operation infomation map.
  ///
  const OpInfoMap &registered_op_info_map();

  ///
  /// \brief Get the dialect of the DialectT class in the context, ff not found,
  /// create and register to context.
  ///
  /// \param DialectT The Dialect class that needs to be found or register.
  ///
  /// \return The dialect of the DialectT class in the context.
  ///
  template <typename DialectT>
  DialectT *GetOrRegisterDialect() {
    return static_cast<DialectT *>(
        GetOrRegisterDialect(DialectT::name(), [this]() {
          DialectT *dialect = new DialectT(this);
          return dialect;
        }));
  }

  ///
  /// \brief Get the dialect of the DialectT class in the context, ff not found,
  /// create and register to context.
  ///
  /// \param dialect_name The dialect name.
  /// \param dialect_id The TypeId of the dialect.
  /// \param constructor The dialect constructor.
  ///
  /// \return The dialect named "dialect_name" in the context.
  ///
  Dialect *GetOrRegisterDialect(const std::string &dialect_name,
                                std::function<Dialect *()> constructor);

  ///
  /// \brief Get the dialect list registered to the context.
  ///
  /// \return The dialect list registered to the context.
  ///
  std::vector<Dialect *> GetRegisteredDialects();

  ///
  /// \brief Get the dialect named "name" from the context.
  ///
  /// \param name The name of the dialect to be obtained.
  ///
  /// \return The dialect named "name" from the context.
  ///
  Dialect *GetRegisteredDialect(const std::string &dialect_name);

  ///
  /// \brief Get a registered dialect for the given dialect type T. The
  /// Dialect must provide a static 'name' method.
  ///
  /// \return The registered dialect for the given dialect type T.
  ///
  template <typename T>
  T *GetRegisteredDialect() {
    return static_cast<T *>(GetRegisteredDialect(T::name()));
  }

  IrContext(const IrContext &) = delete;

  void operator=(const IrContext &) = delete;

 private:
  IrContextImpl *impl_;
};

}  // namespace pir
