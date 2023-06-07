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

#include <type_traits>

#include "paddle/fluid/ir/dialect/pd_type_storage.h"
#include "paddle/ir/core/type.h"
#include "paddle/ir/core/utils.h"
#include "paddle/phi/core/tensor_meta.h"

namespace paddle {
namespace dialect {
///
/// \brief Define Parametric TypeStorage for AllocatedDenseTensorType.
///
/// NOTE(zhangbo9674): The derived TypeStorage class needs to implement the
/// following methods: (1)declare ParamKey, (2)define Construction method,
/// (3)define HashValue method, (4)overload operator==.
///
struct AllocatedDenseTensorTypeStorage : public ir::TypeStorage {
  using Place = phi::Place;
  ///
  /// \brief Declare ParamKey according to parameter type.
  ///
  using ParamKey = std::tuple<phi::Place, dialect::DenseTensorTypeStorage>;

  AllocatedDenseTensorTypeStorage(phi::Place place,
                                  dialect::DenseTensorTypeStorage storage)
      : place_(place), dense_tensor_storage_(storage) {}

  ///
  /// \brief Each derived TypeStorage must define a Construct method, which
  /// StorageManager uses to construct a derived TypeStorage.
  ///
  static AllocatedDenseTensorTypeStorage *Construct(ParamKey key) {
    return new AllocatedDenseTensorTypeStorage(std::get<0>(key),
                                               std::get<1>(key));
  }

  ///
  /// \brief Each derived TypeStorage must provide a HashValue method.
  ///
  static std::size_t HashValue(const ParamKey &key) {
    std::size_t hash_value = 0;
    // hash place
    hash_value = ir::hash_combine(hash_value, std::get<0>(key).HashValue());

    // hash dtype
    auto type_storage = std::get<1>(key);
    hash_value = ir::hash_combine(
        hash_value,
        dialect::DenseTensorTypeStorage::HashValue(
            dialect::DenseTensorTypeStorage::ParamKey(type_storage.dtype_,
                                                      type_storage.dims_,
                                                      type_storage.layout_,
                                                      type_storage.lod_,
                                                      type_storage.offset_)));

    return hash_value;
  }

  ///
  /// \brief Each derived TypeStorage needs to overload operator==.
  ///
  bool operator==(const ParamKey &key) const {
    return ParamKey(place_, dense_tensor_storage_) == key;
  }

  ParamKey GetAsKey() const { return ParamKey(place_, dense_tensor_storage_); }

  ///
  /// \brief AllocatedDenseTensorTypeStorage include five parameters: place,
  /// DenseTensorTypeStorage
  ///
  phi::Place place_;
  dialect::DenseTensorTypeStorage dense_tensor_storage_;
};

}  // namespace dialect
}  // namespace paddle
