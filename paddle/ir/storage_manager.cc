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

#include "paddle/ir/storage_manager.h"

#include <memory>
#include <unordered_map>

namespace ir {
// This is a structure for creating, caching, and looking up Storage of
// parameteric types.
struct ParametricStorageManager {
  using StorageBase = StorageManager::StorageBase;

  ParametricStorageManager() {}

  ~ParametricStorageManager() {
    for (const auto &instance : parametric_instances_) {
      delete instance.second;
    }
    parametric_instances_.clear();
  }

  // Get the storage of parametric type, if not in the cache, create and
  // insert the cache.
  StorageBase *GetOrCreate(std::size_t hash_value,
                           std::function<bool(const StorageBase *)> equal_func,
                           std::function<StorageBase *()> constructor) {
    if (parametric_instances_.count(hash_value) != 0) {
      auto pr = parametric_instances_.equal_range(hash_value);
      while (pr.first != pr.second) {
        if (equal_func(pr.first->second)) {
          VLOG(4) << "Found a cached parameteric storage of: [param_hash="
                  << hash_value << ", storage_ptr=" << pr.first->second << "].";
          return pr.first->second;
        }
        ++pr.first;
      }
    }
    StorageBase *storage = constructor();
    parametric_instances_.emplace(hash_value, storage);
    VLOG(4) << "No cache found, construct and cache a new parameteric storage "
               "of: [param_hash="
            << hash_value << ", storage_ptr=" << storage << "].";
    return storage;
  }

 private:
  // In order to prevent hash conflicts, the unordered_multimap data structure
  // is used for storage.
  std::unordered_multimap<size_t, StorageBase *> parametric_instances_;
};

StorageManager::StorageManager() {}

StorageManager::~StorageManager() = default;

StorageManager::StorageBase *StorageManager::GetParametricStorageImpl(
    TypeId type_id,
    std::size_t hash_value,
    std::function<bool(const StorageBase *)> equal_func,
    std::function<StorageBase *()> constructor) {
  std::lock_guard<ir::SpinLock> guard(parametric_instance_lock_);
  VLOG(4) << "Try to get a parameteretric storage of: [TypeId_hash="
          << std::hash<ir::TypeId>()(type_id) << ", param_hash=" << hash_value
          << "].";
  if (parametric_instance_.find(type_id) == parametric_instance_.end())
    throw("The input data pointer is null.");
  ParametricStorageManager &parametric_storage = *parametric_instance_[type_id];
  return parametric_storage.GetOrCreate(hash_value, equal_func, constructor);
}

StorageManager::StorageBase *StorageManager::GetParameterlessStorageImpl(
    TypeId type_id) {
  std::lock_guard<ir::SpinLock> guard(parameterless_instance_lock_);
  VLOG(4) << "Try to get a parameterless storage of: [TypeId_hash="
          << std::hash<ir::TypeId>()(type_id) << "].";
  if (parameterless_instance_.find(type_id) == parameterless_instance_.end())
    throw("TypeId not found in IrContext.");
  StorageBase *parameterless_instance = parameterless_instance_[type_id];
  return parameterless_instance;
}

void StorageManager::RegisterParametricStorageImpl(TypeId type_id) {
  std::lock_guard<ir::SpinLock> guard(parametric_instance_lock_);
  VLOG(4) << "Register a parameteric storage of: [TypeId_hash="
          << std::hash<ir::TypeId>()(type_id) << "].";
  parametric_instance_.emplace(type_id,
                               std::make_unique<ParametricStorageManager>());
}

void StorageManager::RegisterParameterlessStorageImpl(
    TypeId type_id, std::function<StorageBase *()> constructor) {
  std::lock_guard<ir::SpinLock> guard(parameterless_instance_lock_);
  VLOG(4) << "Register a parameterless storage of: [TypeId_hash="
          << std::hash<ir::TypeId>()(type_id) << "].";
  if (parameterless_instance_.find(type_id) != parameterless_instance_.end())
    throw("storage class already registered");
  parameterless_instance_.emplace(type_id, constructor());
}

}  // namespace ir
