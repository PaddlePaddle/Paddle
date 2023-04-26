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

#include <glog/logging.h>
#include <functional>
#include <memory>
#include <unordered_map>

namespace ir {
class IrContextImpl;
class StorageManager;
class AbstractType;
class AbstractAttribute;
class TypeId;
class Dialect;
class OpInfoImpl;

///
/// \brief IrContext is a global parameterless class used to store and manage
/// Type, Attribute and other related data structures.
///
class IrContext {
 public:
  ///
  /// \brief Initializes a new instance of IrContext.
  ///
  static IrContext *Instance();

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
  void RegisterAbstractType(ir::TypeId type_id, AbstractType *abstract_type);

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
  /// \param abstract_attribute AbstractAttribute* provided by user.
  ///
  void RegisterAbstractAttribute(ir::TypeId type_id,
                                 AbstractAttribute *abstract_attribute);

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
  /// \brief Get or register operaiton.
  ///
  void RegisterOpInfo(const std::string &name, OpInfoImpl *opinfo);

  OpInfoImpl *GetRegisteredOpInfo(const std::string &name);

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
  Dialect *GetOrRegisterDialect(std::string dialect_name,
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
  IrContext();

  const std::unique_ptr<IrContextImpl> impl_;
};

}  // namespace ir
