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

#include "paddle/pir/include/core/interface_value.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/storage_manager.h"
#include "paddle/pir/include/core/type_id.h"

namespace pir {
class Dialect;

///
/// \brief Abstract the properties and behaviors common to all Type classes into
/// an AbstractType class. There are two types in Type system:
/// non-parameter/parameterless type and parametric-type. The common attributes
/// of all types is TypeId (and possibly others). Therefore, construct a class
/// with TypeId as its member.
///
class IR_API AbstractType {
 public:
  ///
  /// \brief Construct an AbstractType by TypeId directly.
  ///
  /// \param type_id The type id of the AbstractType.
  /// \param dialect The Dialect which the type registered to.
  ///
  static AbstractType get(TypeId type_id,
                          const Dialect &dialect,
                          std::set<InterfaceValue> &&interface_set) {
    return AbstractType(type_id, dialect, std::move(interface_set));
  }

  ///
  /// \brief Construct an AbstractType by TypeId directly.
  ///
  /// \param dialect The Dialect which the type registered to.
  ///
  template <typename T>
  static AbstractType get(const Dialect &dialect) {
    return AbstractType(TypeId::get<T>(), dialect, T::interface_set());
  }

  ///
  /// \brief Returns the type id of the AbstractType.
  ///
  /// \return The type id of the AbstractType.
  ///
  TypeId type_id() const { return type_id_; }

  ///
  /// \brief Get the dialect this type was registered to.
  ///
  /// \return The dialect this type was registered to.
  ///
  Dialect &dialect() const { return const_cast<Dialect &>(dialect_); }

  ///
  /// \brief Find the AbstractType instance whose TypeId is type_id from
  /// IrContext.
  ///
  /// \param type_id The type id of the AbstractType.
  /// \param ctx The IrContext.
  /// \return The AbstractType instance whose TypeId is type_id.
  ///
  static const AbstractType &lookup(TypeId type_id, IrContext *ctx);

  ///
  /// \brief Returns an instance of the concept object for the given interface
  /// if it was registered to this type, null otherwise. This should not be used
  /// directly.
  ///
  template <typename InterfaceT>
  typename InterfaceT::Concept *GetInterfaceImpl() const;

  ///
  /// \brief Returns true if the type has the interface with the given ID.
  /// \param interface_id The interface ID of the type.
  ///
  bool HasInterface(TypeId interface_id) const {
    return GetInterfaceImpl(interface_id);
  }

  AbstractType(AbstractType &&) = default;

 private:
  ///
  /// \brief The constructor is set to private and provides the user with the
  /// get method to obtain and manage the AbstractType.
  ///
  /// \param type_id The type id of the AbstractType.
  /// \param dialect The Dialect which the type registered to.
  ///
  explicit AbstractType(TypeId type_id,
                        const Dialect &dialect,
                        std::set<InterfaceValue> &&interface_set)
      : type_id_(type_id),
        dialect_(dialect),
        interface_set_(std::move(interface_set)) {}

  AbstractType(const AbstractType &) = delete;

  void *GetInterfaceImpl(TypeId interface_id) const;

  /// A unique identifier of the derived Type class.
  const TypeId type_id_;

  /// Dialect to which this type was registered
  const Dialect &dialect_;

  /// A collection of the interfaces registered to this type.
  std::set<InterfaceValue> interface_set_;

  /// Trait will be recorded by TypeId.
  uint32_t num_traits_ = 0;
};

template <typename InterfaceT>
typename InterfaceT::Concept *AbstractType::GetInterfaceImpl() const {
  void *model = GetInterfaceImpl(TypeId::get<InterfaceT>());
  return reinterpret_cast<typename InterfaceT::Concept *>(model);
}

struct TypeManager;

///
/// \brief TypeStorage is used to store all information of a Type. A Type object
/// contains a TypeStorage. For non-parameter type, the information includes:
/// TypeId, so TypeStorage only needs to include AbstractType; For parametric
/// type, in addition to AbstractType/TypeId, parametric information needs to
/// be included. So that, non-parametric type can be constructed by TypeStorage
/// directly but parametric type should be constructed by Derived TypeStorage.
///
class IR_API TypeStorage : public StorageManager::StorageBase {
  friend StorageManager;
  friend TypeManager;

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
  const AbstractType &abstract_type() const { return *abstract_type_; }

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

  AbstractType *abstract_type_{nullptr};  // not owned
};

///
/// \brief TypeManager is a utility class that provides interfaces for get or
/// unique Type instances in IrContext.
///
struct IR_API TypeManager {
  ///
  /// \brief Get a unique instance of Type T from IrContext. Note: For a
  /// parametric_type, if not found in IrContext, it will try to create a new
  /// instance and register it to IrContext; for a parameterless type, only
  /// search.
  ///
  /// \param ctx The IrContext instance.
  /// \param args Parameters of the wrapped function.
  /// \return The unique instance of Type T from IrContext.
  ///
  template <typename T, typename... Args>
  static T get(IrContext *ctx, Args &&...args) {
    return get<T, Args...>(
        ctx, pir::TypeId::get<T>(), std::forward<Args>(args)...);
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
      enable_if_t<!std::is_same<typename T::Storage, TypeStorage>::value, T>
      get(IrContext *ctx, TypeId type_id, Args &&...args) {
    return ctx->type_storage_manager()
        .GetParametricStorage<typename T::Storage>(
            [&, type_id](TypeStorage *storage) {
              storage->initialize(AbstractType::lookup(type_id, ctx));
            },
            type_id,
            std::forward<Args>(args)...);
  }

  ///
  /// \brief Get a unique instance of parameterless Type T from IrContext, only
  /// search.
  ///
  /// \param ctx The IrContext instance.
  /// \param type_id The type id of the AbstractType.
  /// \return The unique instance of Type T from IrContext.
  ///
  template <typename T>
  static std::enable_if_t<std::is_same<typename T::Storage, TypeStorage>::value,
                          T>
  get(IrContext *ctx, TypeId type_id) {
    return ctx->type_storage_manager()
        .GetParameterlessStorage<typename T::Storage>(type_id);
  }

  ///
  /// \brief Register a unique instance of Type T to IrContext.
  ///
  /// \param ctx The IrContext instance.
  ///
  template <typename T>
  static void RegisterType(IrContext *ctx) {
    RegisterType<T>(ctx, pir::TypeId::get<T>());
  }

  ///
  /// \brief Register a unique instance of parametric Type T to IrContext.
  ///
  /// \param ctx The IrContext instance.
  /// \param type_id The type id of the Type T.
  ///
  template <typename T>
  static std::enable_if_t<
      !std::is_same<typename T::Storage, TypeStorage>::value>
  RegisterType(IrContext *ctx, TypeId type_id) {
    ctx->type_storage_manager().RegisterParametricStorage<typename T::Storage>(
        type_id);
  }

  ///
  /// \brief Register a unique instance of parameterless Type T to IrContext.
  ///
  /// \param ctx The IrContext instance.
  /// \param type_id The type id of the Type T.
  ///
  template <typename T>
  static std::enable_if_t<std::is_same<typename T::Storage, TypeStorage>::value>
  RegisterType(IrContext *ctx, TypeId type_id) {
    ctx->type_storage_manager().RegisterParameterlessStorage<TypeStorage>(
        type_id, [&ctx, type_id](TypeStorage *storage) {
          storage->initialize(AbstractType::lookup(type_id, ctx));
        });
  }
};

}  // namespace pir
