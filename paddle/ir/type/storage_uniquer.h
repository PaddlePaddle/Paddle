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

#include <memory>
#include <type_traits>
#include <unordered_map>

#include "paddle/ir/type/type_id.h"

namespace ir {
///
/// \brief The implementation of the class StorageUniquer.
///
struct StorageUniquerImpl;

///
/// \brief A utility class for getting or creating Storage class instances.
/// Storage class must be a derived class of StorageUniquer::BaseStorage.
/// There are two types of Storage class:
/// One is a singleton type, which can directly obtain an instance through the
/// get method; The other is a parameteric type, which needs to comply with the
/// following conditions: (1) Need to define a type alias called ParamKey, it
/// serves as the unique identifier for the Storage class; (2) Need to provide a
/// hash method on the ParamKey for storage and access; (3) Need to provide
/// method 'bool operator==(const ParamKey &) const', used to compare Storage
/// instance and ParamKey instance.
///
class StorageUniquer {
 public:
  ///
  /// \brief This class is the base class of all storage classes,
  /// and any type of storage needs to inherit from this class.
  ///
  class BaseStorage {
   protected:
    BaseStorage() = default;
  };

  StorageUniquer();

  ~StorageUniquer();

  ///
  /// \brief Get a unique storage instance of parametric Type.
  ///
  /// \param init_func Used to initialize a newly inserted storage instance.
  /// \param type_id The type id of the AbstractType.
  /// \param args Parameters of the wrapped function.
  /// \return A uniqued instance of Storage.
  ///
  template <typename Storage, typename... Args>
  Storage *get(std::function<void(Storage *)> init_func,
               TypeId type_id,
               Args &&...args) {
    auto derived_param_key = Storage::ParamKey(std::forward<Args>(args)...);
    std::size_t hash_value = Storage::HashValue(derived_param_key);
    bool is_equal_func = [&derived_param_key](const BaseStorage *existing) {
      return static_cast<const Storage &>(*existing) == derived_param_key;
    };
    auto ctor_func = [&]() {
      auto *storage = Storage::Construct(derived_param_key);
      if (init_func) init_func(storage);
      return storage;
    };
    return static_cast<Storage *>(GetParametricStorageTypeImpl(
        type_id, hash_value, is_equal_func, ctor_func));
  }

  ///
  /// \brief Get a unique storage instance of singleton Type.
  ///
  /// \param type_id The type id of the AbstractType.
  /// \return A uniqued instance of Storage.
  ///
  template <typename Storage>
  Storage *get(TypeId type_id) {
    return static_cast<Storage *>(GetSingletonStorageTypeImpl(type_id));
  }

  ///
  /// \brief Register a new parametric storage class.
  ///
  /// \param type_id The type id of the AbstractType.
  ///
  template <typename Storage>
  void RegisterParametricStorageType(TypeId type_id) {
    if (std::is_trivially_destructible<Storage>::value) {
      return RegisterParametricStorageTypeImpl(type_id, nullptr);
    } else {
      return RegisterParametricStorageTypeImpl(
          type_id, [](BaseStorage *storage) {
            static_cast<Storage *>(storage)->~Storage();
          });
    }
  }

  ///
  /// \brief Register a new singleton storage class.
  ///
  /// \param type_id The type id of the AbstractType.
  /// \param init_func Used to initialize a newly inserted storage instance.
  ///
  template <typename Storage>
  void RegisterSingletonStorageType(TypeId type_id,
                                    std::function<void(Storage *)> init_func) {
    VLOG(4) << "==> StorageUniquer::RegisterSingletonStorageType()";
    auto ctor_func = [&]() {
      auto *storage = new Storage(nullptr);
      if (init_func) init_func(storage);
      return storage;
    };
    RegisterSingletonStorageTypeImpl(type_id, ctor_func);
  }

 private:
  BaseStorage *GetParametricStorageTypeImpl(
      TypeId type_id,
      std::size_t hash_value,
      std::function<bool(const BaseStorage *)> is_equal_func,
      std::function<BaseStorage *()> ctor_func);

  BaseStorage *GetSingletonStorageTypeImpl(TypeId type_id);

  void RegisterParametricStorageTypeImpl(
      TypeId type_id, std::function<void(BaseStorage *)> del_func);

  void RegisterSingletonStorageTypeImpl(
      TypeId type_id, std::function<BaseStorage *()> ctor_func);

  /// \brief StorageUniquerImpl is the implementation class of the
  /// StorageUniquer.
  std::unique_ptr<StorageUniquerImpl> impl_;
};

}  // namespace ir
