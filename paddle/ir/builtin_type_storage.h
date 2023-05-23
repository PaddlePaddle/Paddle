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

#include "paddle/ir/type.h"
#include "paddle/ir/utils.h"

namespace ir {
struct VectorTypeStorage : public TypeStorage {
  using ParamKey = std::vector<Type>;

  explicit VectorTypeStorage(const ParamKey &key) {
    data_ = reinterpret_cast<Type *>(malloc(key.size() * sizeof(Type)));
    memcpy(reinterpret_cast<void *>(data_),
           reinterpret_cast<const void *>(key.data()),
           key.size() * sizeof(Type));
    size_ = key.size();
  }

  ~VectorTypeStorage() { free(data_); }

  ///
  /// \brief Each derived TypeStorage must define a Construc method, which
  /// StorageManager uses to construct a derived TypeStorage.
  ///
  static VectorTypeStorage *Construct(ParamKey key) {
    return new VectorTypeStorage(key);
  }

  ///
  /// \brief Each derived TypeStorage must provide a HashValue method.
  ///
  static std::size_t HashValue(const ParamKey &key) {
    std::size_t hash_value = 0;
    for (size_t i = 0; i < key.size(); ++i) {
      hash_value = hash_combine(hash_value, std::hash<Type>()(key[i]));
    }
    return hash_value;
  }

  ///
  /// \brief Each derived TypeStorage needs to overload operator==.
  ///
  bool operator==(const ParamKey &key) const {
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
  Type *data_;
  size_t size_;
};

}  // namespace ir
