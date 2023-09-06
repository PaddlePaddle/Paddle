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

#include "paddle/ir/core/type.h"
#include "paddle/ir/core/type_base.h"
#include "paddle/ir/core/utils.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/ddim.h"

namespace std {
///
/// \brief Enable hashing std::vector<T> instances.
///
template <typename T>
struct hash<std::vector<T>> {
  std::size_t operator()(const std::vector<T>& dim) const {
    std::size_t seed = 0;
    for (size_t i = 0; i < dim.size(); ++i) {
      seed ^= std::hash<T>()(dim[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

}  // namespace std

namespace ir {
///
/// \brief Define Parametric TypeStorage for DenseTensorType.
///
/// NOTE(zhangbo9674): The derived TypeStorage class needs to implement the
/// following methods: (1)declare ParamKey, (2)define Construction method,
/// (3)define HashValue method, (4)overload operator==.
///

struct DenseTensorTypeStorage : public ir::TypeStorage {
  ///
  /// \brief Declare ParamKey according to parameter type.
  ///
  using DataLayout = phi::DataLayout;
  using Dim = phi::DDim;
  using LoD = std::vector<std::vector<size_t>>;
  using ParamKey = std::tuple<ir::Type, Dim, DataLayout, LoD, size_t>;

  DenseTensorTypeStorage(const ir::Type& dtype,
                         const Dim& dims,
                         const DataLayout& layout,
                         const LoD& lod,
                         size_t offset)
      : dtype_(dtype),
        dims_(dims),
        layout_(layout),
        lod_(lod),
        offset_(offset) {}

  ///
  /// \brief Each derived TypeStorage must define a Construct method, which
  /// StorageManager uses to construct a derived TypeStorage.
  ///
  static DenseTensorTypeStorage* Construct(const ParamKey& key) {
    return new DenseTensorTypeStorage(std::get<0>(key),
                                      std::get<1>(key),
                                      std::get<2>(key),
                                      std::get<3>(key),
                                      std::get<4>(key));
  }

  ///
  /// \brief Each derived TypeStorage must provide a HashValue method.
  ///
  static std::size_t HashValue(const ParamKey& key) {
    std::size_t hash_value = 0;
    // hash dtype
    hash_value =
        ir::hash_combine(hash_value, std::hash<ir::Type>()(std::get<0>(key)));
    // hash dims
    hash_value =
        ir::hash_combine(hash_value, std::hash<Dim>()(std::get<1>(key)));
    // hash layout
    hash_value = ir::hash_combine(
        hash_value,
        std::hash<std::underlying_type<DataLayout>::type>()(
            static_cast<std::underlying_type<DataLayout>::type>(
                std::get<2>(key))));
    // hash lod
    hash_value =
        ir::hash_combine(hash_value, std::hash<LoD>()(std::get<3>(key)));
    // hash offset
    hash_value =
        ir::hash_combine(hash_value, std::hash<size_t>()(std::get<4>(key)));
    return hash_value;
  }

  ///
  /// \brief Each derived TypeStorage needs to overload operator==.
  ///
  bool operator==(const ParamKey& key) const {
    return ParamKey(dtype_, dims_, layout_, lod_, offset_) == key;
  }

  ParamKey GetAsKey() const {
    return ParamKey(dtype_, dims_, layout_, lod_, offset_);
  }

  ///
  /// \brief DenseTensorTypeStorage include five parameters: dims, dtype,
  /// layout, lod, offset.
  ///
  ir::Type dtype_;
  Dim dims_;
  DataLayout layout_;
  LoD lod_;
  size_t offset_;
};

struct VectorTypeStorage : public TypeStorage {
  using ParamKey = std::vector<Type>;

  explicit VectorTypeStorage(const ParamKey& key) {
    data_ = reinterpret_cast<Type*>(malloc(key.size() * sizeof(Type)));
    memcpy(reinterpret_cast<void*>(data_),
           reinterpret_cast<const void*>(key.data()),
           key.size() * sizeof(Type));
    size_ = key.size();
  }

  ~VectorTypeStorage() { free(data_); }

  ///
  /// \brief Each derived TypeStorage must define a Construc method, which
  /// StorageManager uses to construct a derived TypeStorage.
  ///
  static VectorTypeStorage* Construct(const ParamKey& key) {
    return new VectorTypeStorage(key);
  }

  ///
  /// \brief Each derived TypeStorage must provide a HashValue method.
  ///
  static std::size_t HashValue(const ParamKey& key) {
    std::size_t hash_value = 0;
    for (size_t i = 0; i < key.size(); ++i) {
      hash_value = hash_combine(hash_value, std::hash<Type>()(key[i]));
    }
    return hash_value;
  }

  ///
  /// \brief Each derived TypeStorage needs to overload operator==.
  ///
  bool operator==(const ParamKey& key) const {
    if (key.size() != size_) {
      return false;
    }
    for (size_t i = 0; i < size_; ++i) {
      if (data_[i] != key[i]) {
        return false;
      }
    }
    return true;
  }

  ParamKey GetAsKey() const { return ParamKey(data_, data_ + size_); }

  ///
  /// \brief DenseTensorTypeStorage include five parameters: dims, dtype,
  /// layout, lod, offset.
  ///
  Type* data_;
  size_t size_;
};

}  // namespace ir
