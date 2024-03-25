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

#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/storage_manager.h"
#include "paddle/pir/include/core/storage_manager_support.h"
#include "paddle/pir/include/core/type_id.h"
namespace pir {
class Dialect;

///
/// \brief Abstract the properties and behaviors common to all Attribute classes
/// into an AbstractAttribute class.
///
class IR_API AbstractAttribute {
 public:
  ///
  /// \brief Construct an AbstractAttribute by TypeId directly.
  ///
  /// \param type_id The id of the AbstractAttribute.
  /// \param dialect The Dialect which the attribute registered to.
  ///
  static AbstractAttribute get(TypeId type_id, const Dialect &dialect) {
    return AbstractAttribute(type_id, dialect);
  }

  ///
  /// \brief Construct an AbstractAttribute by TypeId directly.
  ///
  /// \param dialect The Dialect which the attribute registered to.
  ///
  template <typename T>
  static AbstractAttribute get(const Dialect &dialect) {
    return AbstractAttribute(TypeId::get<T>(), dialect);
  }

  ///
  /// \brief Returns the type id of the AbstractAttribute.
  ///
  /// \return The id of the AbstractAttribute.
  ///
  TypeId type_id() const { return type_id_; }

  ///
  /// \brief Get the dialect this attribute was registered to.
  ///
  /// \return The dialect this attribute was registered to.
  ///
  const Dialect &dialect() const { return dialect_; }

  ///
  /// \brief Find the AbstractAttribute instance whose TypeId is type_id from
  /// IrContext.
  ///
  /// \param type_id The type id of the AbstractAttribute.
  /// \param ctx The IrContext.
  /// \return The AbstractAttribute instance whose TypeId is type_id.
  ///
  static const AbstractAttribute &lookup(TypeId type_id, IrContext *ctx);

 private:
  ///
  /// \brief The constructor is set to private and provides the user with the
  /// get method to obtain and manage the AbstractAttribute.
  ///
  /// \param type_id The type id of the AbstractAttribute.
  /// \param dialect The Dialect which the attribute registered to.
  ///
  explicit AbstractAttribute(TypeId type_id, const Dialect &dialect)
      : type_id_(type_id), dialect_(dialect) {}

  TypeId type_id_;
  const Dialect &dialect_;
};

struct AttributeManager;

///
/// \brief AttributeStorage is used to store all information of a Attribute. A
/// Attribute object contains a AttributeStorage. For non-parameter attribute,
/// the information includes: TypeId, so AttributeStorage only needs to include
/// AbstractAttribute; For parametric attribute, in addition to
/// AbstractAttribute/TypeId, parametric information needs to be included. So
/// that, non-parametric attribute can be constructed by AttributeStorage
/// directly but parametric attribute should be constructed by Derived
/// AttributeStorage.
///
class IR_API AttributeStorage : public StorageManager::StorageBase {
  friend StorageManager;
  friend AttributeManager;

 public:
  ///
  /// \brief Construct a AttributeStorage and initialize abstract_attribute.
  ///
  /// \param abstract_attribute The abstract_attribute of this AttributeStorage.
  ///
  explicit AttributeStorage(AbstractAttribute *abstract_attribute)
      : abstract_attribute_(abstract_attribute) {}

  AttributeStorage() {}

  ///
  /// \brief Returns the AbstractAttribute of the AttributeStorage.
  ///
  /// \return The AbstractAttribute of the AttributeStorage.
  ///
  const AbstractAttribute &abstract_attribute() const {
    return *abstract_attribute_;
  }

 private:
  ///
  /// \brief Initialize AttributeStorage based on the AbstractAttribute*
  /// provided by the user
  ///
  /// \param abstract_attribute AbstractAttribute* provided by the user, the
  /// construction method of AbstractAttribute refers to AbstractAttribute::get.
  ///
  void initialize(const AbstractAttribute &abstract_attribute) {
    abstract_attribute_ = const_cast<AbstractAttribute *>(&abstract_attribute);
  }

  AbstractAttribute *abstract_attribute_{nullptr};  // not owned
};

///
/// \brief AttributeManager is a utility class that provides interfaces for get
/// or unique Attribute instances in IrContext.
///
struct IR_API AttributeManager {
  ///
  /// \brief Get a unique instance of Attribute T from IrContext. Note: For a
  /// parametric attribute, if not found in IrContext, it will try to create a
  /// new instance and register it to IrContext; for a parameterless attribute,
  /// only search.
  ///
  /// \param ctx The IrContext instance.
  /// \param args Parameters of the wrapped function.
  /// \return The unique instance of Attribute T from IrContext.
  ///
  template <typename T, typename... Args>
  static T get(IrContext *ctx, Args &&...args) {
    return get<T, Args...>(
        ctx, pir::TypeId::get<T>(), std::forward<Args>(args)...);
  }

  ///
  /// \brief Get a unique instance of parametric Attribute T from IrContext. If
  /// not found in IrContext, it will try to create a new instance and register
  /// it to IrContext;
  ///
  /// \param ctx The IrContext instance.
  /// \param type_id The type id of the AbstractAttribute.
  /// \param args Parameters of the wrapped function.
  /// \return The unique instance of Attribute T from IrContext.
  ///
  template <typename T, typename... Args>
  static std::enable_if_t<
      !std::is_same<typename T::Storage, AttributeStorage>::value,
      T>
  get(IrContext *ctx, TypeId type_id, Args &&...args) {
    return ctx->attribute_storage_manager()
        .GetParametricStorage<typename T::Storage>(
            [&, type_id](AttributeStorage *storage) {
              storage->initialize(AbstractAttribute::lookup(type_id, ctx));
            },
            type_id,
            std::forward<Args>(args)...);
  }

  ///
  /// \brief Get a unique instance of parameterless Attribute T from IrContext.
  ///
  /// \param ctx The IrContext instance.
  /// \param type_id The type id of the AbstractAttribute.
  /// \return The unique instance of Attribute T from IrContext.
  ///
  template <typename T>
  static std::
      enable_if_t<std::is_same<typename T::Storage, AttributeStorage>::value, T>
      get(IrContext *ctx, TypeId type_id) {
    return ctx->attribute_storage_manager()
        .GetParameterlessStorage<typename T::Storage>(type_id);
  }

  ///
  /// \brief Register a unique instance of Attribute T to IrContext.
  ///
  /// \param ctx The IrContext instance.
  ///
  template <typename T>
  static void RegisterAttribute(IrContext *ctx) {
    RegisterAttribute<T>(ctx, pir::TypeId::get<T>());
  }

  ///
  /// \brief Register a unique parametric Attribute T to IrContext.
  ///
  /// \param ctx The IrContext instance.
  /// \param type_id The type id of the Attribute T.
  ///
  template <typename T>
  static std::enable_if_t<
      !std::is_same<typename T::Storage, AttributeStorage>::value>
  RegisterAttribute(IrContext *ctx, TypeId type_id) {
    ctx->attribute_storage_manager()
        .RegisterParametricStorage<typename T::Storage>(type_id);
  }

  ///
  /// \brief Register a unique parameterless Attribute T to IrContext.
  ///
  /// \param ctx The IrContext instance.
  /// \param type_id The type id of the Attribute T.
  ///
  template <typename T>
  static std::enable_if_t<
      std::is_same<typename T::Storage, AttributeStorage>::value>
  RegisterAttribute(IrContext *ctx, TypeId type_id) {
    ctx->attribute_storage_manager()
        .RegisterParameterlessStorage<AttributeStorage>(
            type_id, [&ctx, type_id](AttributeStorage *storage) {
              storage->initialize(AbstractAttribute::lookup(type_id, ctx));
            });
  }
};

template <typename ConcreteType,
          typename BaseType,
          typename StorageType,
          class... TraitOrInterface>
using AttrBase = detail::StorageHelperBase<ConcreteType,
                                           BaseType,
                                           StorageType,
                                           AttributeManager,
                                           TraitOrInterface...>;

///
/// \brief Add some necessary functions to the custom Attribute class.
///
#define DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(concrete_attribute, storage_type)  \
  using Storage = storage_type;                                              \
                                                                             \
  const Storage *storage() const {                                           \
    return static_cast<const Storage *>(this->storage_);                     \
  }                                                                          \
                                                                             \
  static pir::TypeId type_id() {                                             \
    return pir::TypeId::get<concrete_attribute>();                           \
  }                                                                          \
                                                                             \
  template <typename T>                                                      \
  static bool classof(T val) {                                               \
    return val.type_id() == type_id();                                       \
  }                                                                          \
                                                                             \
  template <typename... Args>                                                \
  static concrete_attribute get(pir::IrContext *ctx, Args... args) {         \
    return pir::AttributeManager::template get<concrete_attribute>(ctx,      \
                                                                   args...); \
  }
}  // namespace pir
