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

#include "paddle/ir/type/storage_uniquer.h"
#include <memory>
#include <unordered_map>
#include "paddle/phi/api/ext/exception.h"

namespace ir {

struct ParametricStorageUniquer {
  using BaseStorage = StorageUniquer::BaseStorage;

  explicit ParametricStorageUniquer(std::function<void(BaseStorage *)> del_func)
      : del_func_(del_func) {}

  ~ParametricStorageUniquer() {
    for (const auto &instance : parametric_instances_) {
      del_func_(instance.second);
    }
  }

  /// \brief Get the storage of parametric type, if not in the cache, create and
  /// insert the cache.
  BaseStorage *GetOrCreateParametricStorage(
      std::size_t hash_value,
      std::function<bool(const BaseStorage *)> is_equal_func,
      std::function<BaseStorage *()> ctor_func) {
    BaseStorage *storage = nullptr;
    if (parametric_instances_.count(hash_value) == 0) {
      storage = ctor_func();
      parametric_instances_.emplace(hash_value, storage);
    } else {
      auto pr = parametric_instances_.equal_range(hash_value);
      while (pr.first != pr.second) {
        if (is_equal_func(pr.first->second)) {
          return pr.first->second;
        }
        ++pr.first;
      }
      storage = ctor_func();
      parametric_instances_.emplace(hash_value, storage);
    }
    return storage;
  }

 private:
  std::unordered_multimap<size_t, BaseStorage *> parametric_instances_;

  std::function<void(BaseStorage *)> del_func_;
};

struct StorageUniquerImpl {
  using BaseStorage = StorageUniquer::BaseStorage;

  /// \brief Get the storage of parametric type, if not in the cache, create and
  /// insert the cache.
  BaseStorage *GetOrCreateParametricStorage(
      TypeId type_id,
      std::size_t hash_value,
      std::function<bool(const BaseStorage *)> is_equal_func,
      std::function<BaseStorage *()> ctor_func) {
    if (parametric_uniquers_.find(type_id) == parametric_uniquers_.end())
      PD_THROW("The input data pointer is null.");
    ParametricStorageUniquer &parametric_storage =
        *parametric_uniquers_[type_id];
    return parametric_storage.GetOrCreateParametricStorage(
        hash_value, is_equal_func, ctor_func);
  }

  /// \brief Get the storage of singleton type.
  /// \param type_id The type id of the AbstractType.
  BaseStorage *GetSingletonStorage(TypeId type_id) {
    if (singleton_instances_.find(type_id) == singleton_instances_.end())
      PD_THROW("The input data pointer is null.");
    BaseStorage *singleton_instance = singleton_instances_[type_id];
    return singleton_instance;
  }

  void RegisterParametricStorageType(
      TypeId type_id, std::function<void(BaseStorage *)> del_func);

  void RegisterSingletonStorageType(TypeId type_id,
                                    std::function<void()> init_func);

  std::unordered_map<TypeId, std::unique_ptr<ParametricStorageUniquer>>
      parametric_uniquers_;

  std::unordered_map<TypeId, BaseStorage *> singleton_instances_;
};

StorageUniquer::StorageUniquer() : impl_(new StorageUniquerImpl()) {}

StorageUniquer::BaseStorage *StorageUniquer::GetParametricStorageTypeImpl(
    TypeId type_id,
    std::size_t hash_value,
    std::function<bool(const BaseStorage *)> is_equal_func,
    std::function<BaseStorage *()> ctor_func) {
  return impl_->GetOrCreateParametricStorage(
      type_id, hash_value, is_equal_func, ctor_func);
}

StorageUniquer::BaseStorage *StorageUniquer::GetSingletonStorageTypeImpl(
    TypeId type_id) {
  return impl_->GetSingletonStorage(type_id);
}

void StorageUniquer::RegisterParametricStorageTypeImpl(
    TypeId type_id, std::function<void(BaseStorage *)> del_func) {
  impl_->parametric_uniquers_.emplace(
      type_id, std::make_unique<ParametricStorageUniquer>(del_func));
}

void StorageUniquer::RegisterSingletonStorageTypeImpl(
    TypeId type_id, std::function<BaseStorage *()> ctor_func) {
  if (impl_->singleton_instances_.find(type_id) !=
      impl_->singleton_instances_.end())
    PD_THROW("storage class already registered");
  impl_->singleton_instances_.emplace(type_id, ctor_func());
}

}  // namespace ir
