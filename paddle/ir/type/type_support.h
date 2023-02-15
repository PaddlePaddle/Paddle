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

#include "paddle/ir/ir_context.h"
#include "paddle/ir/type/storage_uniquer.h"
#include "paddle/ir/type/type_base.h"

namespace ir {
///
/// \brief Abstract the properties and behaviors common to all Type classes into
/// an AbstractType class. There are two types in Type system:
/// on-parameter/singleton type and parameter-type. The common attributes of all
/// types is TypeId (and possibly others). Therefore, construct a class with
/// TypeId as its member.
///
class AbstractType {
 public:
  ///
  /// \brief Construct an AbstractType by TypeId directly.
  ///
  /// \param type_id The type id of the AbstractType.
  ///
  static AbstractType get(TypeId type_id) { return AbstractType(type_id); }

  ///
  /// \brief Returns the type id of the AbstractType.
  ///
  /// \return The type id of the AbstractType.
  ///
  TypeId type_id() const { return type_id_; }

  ///
  /// \brief Find the AbstractType instance whose TypeId is type_id from
  /// IrContext.
  ///
  /// \param type_id The type id of the AbstractType.
  /// \param ctx The IrContext.
  /// \return The AbstractType instance whose TypeId is type_id.
  ///
  static const AbstractType &lookup(TypeId type_id, IrContext *ctx);

 private:
  ///
  /// \brief The constructor is set to private and provides the user with the
  /// get method to obtain and manage the AstractType.
  ///
  /// \param type_id The type id of the AbstractType.
  ///
  explicit AbstractType(TypeId type_id) : type_id_(type_id) {}

  TypeId type_id_;
};

struct TypeUniquer;

///
/// \brief TypeStorage is used to store all information of a Type. A Type object
/// contains a TypeStorage. For non-parameter type, the information includes:
/// TypeId, so TypeStorage only needs to include AbstractType; For parameter
/// type, in addition to AbstractType/TypeId, parameter information needs to be
/// included. So that, non-parameter type can be constructed by TypeStorage
/// directly but parameter type should be constructed by Derived TypeStorage.
///
class TypeStorage : public StorageUniquer::BaseStorage {
  friend StorageUniquer;
  friend TypeUniquer;

 public:
  ///
  /// \brief Construct a TypeStorage and initialize abstract_type.
  ///
  /// \param abstract_type The abstract_type of this TypeStorage.
  ///
  explicit TypeStorage(AbstractType *abstract_type)
      : abstract_type_(abstract_type) {}

  TypeStorage() {}

  ///
  /// \brief Returns the AbstractType of the TypeStorage.
  ///
  /// \return The AbstractType of the TypeStorage.
  ///
  const AbstractType &abstract_type() { return *abstract_type_; }

 private:
  ///
  /// \brief Initialize TypeStorage based on the AbstractType* provided by the
  /// user
  ///
  /// \param abstract_type AbstractType* provided by the user, the
  /// construction method of AbstractType refers to AbstractType::get.
  ///
  void initialize(const AbstractType &abstract_type) {
    abstract_type_ = const_cast<AbstractType *>(&abstract_type);
  }

  AbstractType *abstract_type_{nullptr};
};

///
/// \brief TypeUniquer is a utility class that provides interfaces for get or
/// unique Type instances in IrContext.
///
struct TypeUniquer {
  ///
  /// \brief Get a unique instance of Type T from IrContext. Note: For a
  /// parameteric_type, if not found in IrContext, it will try to create a new
  /// instance and register it to IrContext; for a singleton_type, only search.
  ///
  /// \param ctx The IrContext instance.
  /// \param args Parameters of the wrapped function.
  /// \return The unique instance of Type T from IrContext.
  ///
  template <typename T, typename... Args>
  static T get(IrContext *ctx, Args &&...args) {
    return GetWithTypeId<T, Args...>(
        ctx, T::type_id(), std::forward<Args>(args)...);
  }

  ///
  /// \brief Get a unique instance of parametric Type T from IrContext. If not
  /// found in IrContext, it will try to create a new instance and register it
  /// to IrContext;
  ///
  /// \param ctx The IrContext instance.
  /// \param type_id The type id of the AbstractType.
  /// \param args Parameters of the wrapped function.
  /// \return The unique instance of Type T from IrContext.
  ///
  template <typename T, typename... Args>
  static std::
      enable_if_t<!std::is_same<typename T::ImplType, TypeStorage>::value, T>
      GetWithTypeId(IrContext *ctx, TypeId type_id, Args &&...args) {
    return ctx->storage_uniquer().get<typename T::ImplType>(
        [&, type_id](TypeStorage *storage) {
          storage->initialize(AbstractType::lookup(type_id, ctx));
        },
        type_id,
        std::forward<Args>(args)...);
  }

  ///
  /// \brief Get a unique instance of singleton Type T from IrContext, only
  /// search.
  ///
  /// \param ctx The IrContext instance.
  /// \param type_id The type id of the AbstractType.
  /// \return The unique instance of Type T from IrContext.
  ///
  template <typename T>
  static std::
      enable_if_t<std::is_same<typename T::ImplType, TypeStorage>::value, T>
      GetWithTypeId(IrContext *ctx, TypeId type_id) {
    return ctx->storage_uniquer().get<typename T::ImplType>(type_id);
  }

  ///
  /// \brief Register a unique instance of Type T to IrContext.
  ///
  /// \param ctx The IrContext instance.
  ///
  template <typename T>
  static void RegisterType(IrContext *ctx) {
    RegisterType<T>(ctx, T::type_id());  // class Type需要提供type_id接口
  }

  ///
  /// \brief Register a unique instance of parametric Type T to IrContext.
  ///
  /// \param ctx The IrContext instance.
  /// \param type_id The type id of the Type T.
  ///
  template <typename T>
  static std::enable_if_t<
      !std::is_same<typename T::ImplType, TypeStorage>::value>
  RegisterType(IrContext *ctx, TypeId type_id) {
    ctx->storage_uniquer().RegisterParametricStorageType<typename T::ImplType>(
        type_id);
  }

  ///
  /// \brief Register a unique instance of singleton Type T to IrContext.
  ///
  /// \param ctx The IrContext instance.
  /// \param type_id The type id of the Type T.
  ///
  template <typename T>
  static std::enable_if_t<
      std::is_same<typename T::ImplType, TypeStorage>::value>
  RegisterType(IrContext *ctx, TypeId type_id) {
    ctx->storage_uniquer().RegisterSingletonStorageType<TypeStorage>(
        type_id, [&ctx, type_id](TypeStorage *storage) {
          storage->initialize(AbstractType::lookup(type_id, ctx));
        });
  }
};

}  // namespace ir
