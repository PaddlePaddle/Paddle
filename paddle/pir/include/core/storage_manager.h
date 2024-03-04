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

#include "paddle/pir/include/core/spin_lock.h"
#include "paddle/pir/include/core/type_id.h"

namespace pir {
///
/// \brief The implementation of the class StorageManager.
///
// struct StorageManagerImpl;
struct ParametricStorageManager;

///
/// \brief A utility class for getting or creating Storage class instances.
/// Storage class must be a derived class of StorageManager::StorageBase.
/// There are two types of Storage class:
/// One is a parameterless type, which can directly obtain an instance through
/// the get method; The other is a parametric type, which needs to comply with
/// the following conditions: (1) Need to define a type alias called ParamKey,
/// it serves as the unique identifier for the Storage class; (2) Need to
/// provide a hash method on the ParamKey for storage and access; (3) Need to
/// provide method 'bool operator==(const ParamKey &) const', used to compare
/// Storage instance and ParamKey instance.
///
class IR_API StorageManager {
 public:
  ///
  /// \brief This class is the base class of all storage classes,
  /// and any type of storage needs to inherit from this class.
  ///
  class StorageBase {
   protected:
    StorageBase() = default;
  };

  StorageManager();

  ~StorageManager();

  ///
  /// \brief Get a unique storage instance of parametric Type.
  ///
  /// \param init_func Used to initialize a newly inserted storage instance.
  /// \param type_id The type id of the AbstractType.
  /// \param args Parameters of the wrapped function.
  /// \return A uniqued instance of Storage.
  ///
  template <typename Storage, typename... Args>
  Storage *GetParametricStorage(std::function<void(Storage *)> init_func,
                                TypeId type_id,
                                Args &&...args) {
    typename Storage::ParamKey param =
        typename Storage::ParamKey(std::forward<Args>(args)...);
    std::size_t hash_value = Storage::HashValue(param);
    auto equal_func = [&param](const StorageBase *existing) {
      return static_cast<const Storage &>(*existing) == param;
    };
    auto constructor = [&]() {
      auto *storage = Storage::Construct(std::move(param));
      if (init_func) init_func(storage);
      return storage;
    };
    return static_cast<Storage *>(
        GetParametricStorageImpl(type_id, hash_value, equal_func, constructor));
  }

  ///
  /// \brief Get a unique storage instance of parameterless Type.
  ///
  /// \param type_id The type id of the AbstractType.
  /// \return A uniqued instance of Storage.
  ///
  template <typename Storage>
  Storage *GetParameterlessStorage(TypeId type_id) {
    return static_cast<Storage *>(GetParameterlessStorageImpl(type_id));
  }

  ///
  /// \brief Register a new parametric storage class.
  ///
  /// \param type_id The type id of the AbstractType.
  ///
  template <typename Storage>
  void RegisterParametricStorage(TypeId type_id) {
    return RegisterParametricStorageImpl(type_id, [](StorageBase *storage) {
      delete static_cast<Storage *>(storage);
    });
  }

  ///
  /// \brief Register a new parameterless storage class.
  ///
  /// \param type_id The type id of the AbstractType.
  /// \param init_func Used to initialize a newly inserted storage instance.
  ///
  template <typename Storage>
  void RegisterParameterlessStorage(TypeId type_id,
                                    std::function<void(Storage *)> init_func) {
    auto constructor = [&]() {
      auto *storage = new Storage();
      if (init_func) init_func(storage);
      return storage;
    };
    RegisterParameterlessStorageImpl(type_id, constructor);
  }

 private:
  StorageBase *GetParametricStorageImpl(
      TypeId type_id,
      std::size_t hash_value,
      std::function<bool(const StorageBase *)> equal_func,
      std::function<StorageBase *()> constructor);

  StorageBase *GetParameterlessStorageImpl(TypeId type_id);

  void RegisterParametricStorageImpl(
      TypeId type_id, std::function<void(StorageBase *)> destroy);

  void RegisterParameterlessStorageImpl(
      TypeId type_id, std::function<StorageBase *()> constructor);

  // This map is a mapping between type id and parametric type storage.
  std::unordered_map<TypeId, std::unique_ptr<ParametricStorageManager>>
      parametric_instance_;

  pir::SpinLock parametric_instance_lock_;

  // This map is a mapping between type id and parameterless type storage.
  std::unordered_map<TypeId, StorageBase *> parameterless_instance_;

  pir::SpinLock parameterless_instance_lock_;
};

}  // namespace pir
