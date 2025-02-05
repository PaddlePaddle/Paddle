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

#include <cstring>

#include "paddle/common/ddim.h"
#include "paddle/common/dim.h"
#include "paddle/common/hash_funcs.h"
#include "paddle/common/layout.h"
#include "paddle/pir/include/core/type.h"
#include "paddle/pir/include/core/type_base.h"
#include "paddle/pir/include/core/utils.h"

namespace pir {
///
/// \brief Define Parametric TypeStorage for DenseTensorType.
///
/// NOTE(zhangbo9674): The derived TypeStorage class needs to implement the
/// following methods: (1)declare ParamKey, (2)define Construction method,
/// (3)define HashValue method, (4)overload operator==.
///

struct DenseTensorTypeStorage : public pir::TypeStorage {
  ///
  /// \brief Declare ParamKey according to parameter type.
  ///
  using Dim = pir::DDim;
  using DataLayout = pir::DataLayout;
  using LegacyLoD = std::vector<std::vector<size_t>>;
  using ParamKey = std::tuple<Type, pir::DDim, DataLayout, LegacyLoD, size_t>;

  DenseTensorTypeStorage(Type dtype,
                         const pir::DDim& dims,
                         DataLayout layout,
                         const LegacyLoD& lod,
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
    hash_value = detail::hash_combine(hash_value,
                                      std::hash<pir::Type>()(std::get<0>(key)));
    // hash dims
    hash_value = detail::hash_combine(hash_value,
                                      std::hash<pir::DDim>()(std::get<1>(key)));
    // hash layout
    hash_value = detail::hash_combine(
        hash_value,
        std::hash<std::underlying_type<DataLayout>::type>()(
            static_cast<std::underlying_type<DataLayout>::type>(
                std::get<2>(key))));
    // hash lod
    hash_value = detail::hash_combine(hash_value,
                                      std::hash<LegacyLoD>()(std::get<3>(key)));
    // hash offset
    hash_value =
        detail::hash_combine(hash_value, std::hash<size_t>()(std::get<4>(key)));
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
  pir::Type dtype_;
  pir::DDim dims_;
  DataLayout layout_;
  LegacyLoD lod_;
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
  /// \brief Each derived TypeStorage must define a Construct method, which
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
      hash_value = detail::hash_combine(hash_value, std::hash<Type>()(key[i]));
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

}  // namespace pir
