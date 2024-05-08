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

#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/pir/include/core/type.h"
#include "paddle/pir/include/core/utils.h"

namespace paddle {
namespace dialect {
///
/// \brief Define Parametric TypeStorage for AllocatedDenseTensorType.
///
/// NOTE(zhangbo9674): The derived TypeStorage class needs to implement the
/// following methods: (1)declare ParamKey, (2)define Construction method,
/// (3)define HashValue method, (4)overload operator==.
///
struct AllocatedDenseTensorTypeStorage : public pir::TypeStorage {
  using Place = phi::Place;
  ///
  /// \brief Declare ParamKey according to parameter type.
  ///
  using ParamKey = std::tuple<phi::Place, dialect::DenseTensorType>;

  AllocatedDenseTensorTypeStorage(const phi::Place& place,
                                  const dialect::DenseTensorType& type)
      : place_(place), dense_tensor_type_(type) {}

  ///
  /// \brief Each derived TypeStorage must define a Construct method, which
  /// StorageManager uses to construct a derived TypeStorage.
  ///
  static AllocatedDenseTensorTypeStorage* Construct(const ParamKey& key) {
    return new AllocatedDenseTensorTypeStorage(std::get<0>(key),
                                               std::get<1>(key));
  }

  ///
  /// \brief Each derived TypeStorage must provide a HashValue method.
  ///
  static std::size_t HashValue(const ParamKey& key) {
    std::size_t hash_value = 0;
    // hash place
    hash_value =
        pir::detail::hash_combine(hash_value, std::get<0>(key).HashValue());

    // hash dtype
    auto dense_tensor_type = std::get<1>(key);
    hash_value =
        pir::detail::hash_combine(hash_value,
                                  dialect::DenseTensorTypeStorage::HashValue(
                                      dialect::DenseTensorTypeStorage::ParamKey(
                                          dense_tensor_type.dtype(),
                                          dense_tensor_type.dims(),
                                          dense_tensor_type.data_layout(),
                                          dense_tensor_type.lod(),
                                          dense_tensor_type.offset())));
    return hash_value;
  }

  ///
  /// \brief Each derived TypeStorage needs to overload operator==.
  ///
  bool operator==(const ParamKey& key) const {
    return ParamKey(place_, dense_tensor_type_) == key;
  }

  ParamKey GetAsKey() const { return ParamKey(place_, dense_tensor_type_); }

  ///
  /// \brief AllocatedDenseTensorTypeStorage include five parameters: place,
  /// DenseTensorType
  ///
  phi::Place place_;
  dialect::DenseTensorType dense_tensor_type_;
};

///
/// \brief Define Parametric TypeStorage for AllocatedSelectedRowsTypeStorage.
///
///
struct AllocatedSelectedRowsTypeStorage : public pir::TypeStorage {
  using Place = phi::Place;
  ///
  /// \brief Declare ParamKey according to parameter type.
  ///
  using ParamKey = std::tuple<phi::Place, dialect::SelectedRowsType>;

  AllocatedSelectedRowsTypeStorage(const phi::Place& place,
                                   const dialect::SelectedRowsType& type)
      : place_(place), selected_rows_type_(type) {}

  ///
  /// \brief Each derived TypeStorage must define a Construct method, which
  /// StorageManager uses to construct a derived TypeStorage.
  ///
  static AllocatedSelectedRowsTypeStorage* Construct(const ParamKey& key) {
    return new AllocatedSelectedRowsTypeStorage(std::get<0>(key),
                                                std::get<1>(key));
  }

  ///
  /// \brief Each derived TypeStorage must provide a HashValue method.
  ///
  static std::size_t HashValue(const ParamKey& key) {
    std::size_t hash_value = 791;
    // hash place
    hash_value =
        pir::detail::hash_combine(hash_value, std::get<0>(key).HashValue());

    // hash dtype
    auto selected_rows_type = std::get<1>(key);
    hash_value =
        pir::detail::hash_combine(hash_value,
                                  dialect::DenseTensorTypeStorage::HashValue(
                                      dialect::DenseTensorTypeStorage::ParamKey(
                                          selected_rows_type.dtype(),
                                          selected_rows_type.dims(),
                                          selected_rows_type.data_layout(),
                                          selected_rows_type.lod(),
                                          selected_rows_type.offset())));
    return hash_value;
  }

  ///
  /// \brief Each derived TypeStorage needs to overload operator==.
  ///
  bool operator==(const ParamKey& key) const {
    return ParamKey(place_, selected_rows_type_) == key;
  }

  ParamKey GetAsKey() const { return ParamKey(place_, selected_rows_type_); }

  ///
  /// \brief AllocatedSelectedRowsTypeStorage include five parameters: place,
  /// SelectedRowsType
  ///
  phi::Place place_;
  dialect::SelectedRowsType selected_rows_type_;
};

///
/// \brief Define Parametric TypeStorage for
/// AllocatedSparseCooTensorTypeStorage.
///
///
struct AllocatedSparseCooTensorTypeStorage : public pir::TypeStorage {
  using Place = phi::Place;
  ///
  /// \brief Declare ParamKey according to parameter type.
  ///
  using ParamKey = std::tuple<phi::Place, dialect::SparseCooTensorType>;

  AllocatedSparseCooTensorTypeStorage(const phi::Place& place,
                                      const dialect::SparseCooTensorType& type)
      : place_(place), sparsecoo_tensor_type_(type) {}

  ///
  /// \brief Each derived TypeStorage must define a Construct method, which
  /// StorageManager uses to construct a derived TypeStorage.
  ///
  static AllocatedSparseCooTensorTypeStorage* Construct(const ParamKey& key) {
    return new AllocatedSparseCooTensorTypeStorage(std::get<0>(key),
                                                   std::get<1>(key));
  }

  ///
  /// \brief Each derived TypeStorage must provide a HashValue method.
  ///
  static std::size_t HashValue(const ParamKey& key) {
    std::size_t hash_value = 791;
    // hash place
    hash_value =
        pir::detail::hash_combine(hash_value, std::get<0>(key).HashValue());

    // hash dtype
    auto sparsecoo_tensor_type = std::get<1>(key);
    hash_value = pir::detail::hash_combine(
        hash_value,
        dialect::SparseCooTensorTypeStorage::HashValue(
            dialect::SparseCooTensorTypeStorage::ParamKey(
                sparsecoo_tensor_type.dtype(),
                sparsecoo_tensor_type.dims(),
                sparsecoo_tensor_type.non_zero_dims(),
                sparsecoo_tensor_type.data_layout(),
                sparsecoo_tensor_type.non_zero_indices(),
                sparsecoo_tensor_type.non_zero_elements(),
                sparsecoo_tensor_type.coalesced())));
    return hash_value;
  }

  ///
  /// \brief Each derived TypeStorage needs to overload operator==.
  ///
  bool operator==(const ParamKey& key) const {
    return ParamKey(place_, sparsecoo_tensor_type_) == key;
  }

  ParamKey GetAsKey() const { return ParamKey(place_, sparsecoo_tensor_type_); }

  ///
  /// \brief AllocatedSparseCooTensorTypeStorage include five parameters: place,
  /// SparseCooTensorType
  ///
  phi::Place place_;
  dialect::SparseCooTensorType sparsecoo_tensor_type_;
};

///
/// \brief Define Parametric TypeStorage for
/// AllocatedSparseCsrTensorTypeStorage.
///
///
struct AllocatedSparseCsrTensorTypeStorage : public pir::TypeStorage {
  using Place = phi::Place;
  ///
  /// \brief Declare ParamKey according to parameter type.
  ///
  using ParamKey = std::tuple<phi::Place, dialect::SparseCsrTensorType>;

  AllocatedSparseCsrTensorTypeStorage(const phi::Place& place,
                                      const dialect::SparseCsrTensorType& type)
      : place_(place), sparsecsr_tensor_type_(type) {}

  ///
  /// \brief Each derived TypeStorage must define a Construct method, which
  /// StorageManager uses to construct a derived TypeStorage.
  ///
  static AllocatedSparseCsrTensorTypeStorage* Construct(const ParamKey& key) {
    return new AllocatedSparseCsrTensorTypeStorage(std::get<0>(key),
                                                   std::get<1>(key));
  }

  ///
  /// \brief Each derived TypeStorage must provide a HashValue method.
  ///
  static std::size_t HashValue(const ParamKey& key) {
    std::size_t hash_value = 791;
    // hash place
    hash_value =
        pir::detail::hash_combine(hash_value, std::get<0>(key).HashValue());

    // hash dtype
    auto sparsecsr_tensor_type = std::get<1>(key);
    hash_value = pir::detail::hash_combine(
        hash_value,
        dialect::SparseCsrTensorTypeStorage::HashValue(
            dialect::SparseCsrTensorTypeStorage::ParamKey(
                sparsecsr_tensor_type.dtype(),
                sparsecsr_tensor_type.dims(),
                sparsecsr_tensor_type.data_layout(),
                sparsecsr_tensor_type.non_zero_crows(),
                sparsecsr_tensor_type.non_zero_cols(),
                sparsecsr_tensor_type.non_zero_elements())));
    return hash_value;
  }

  ///
  /// \brief Each derived TypeStorage needs to overload operator==.
  ///
  bool operator==(const ParamKey& key) const {
    return ParamKey(place_, sparsecsr_tensor_type_) == key;
  }

  ParamKey GetAsKey() const { return ParamKey(place_, sparsecsr_tensor_type_); }

  ///
  /// \brief AllocatedSparseCooTensorTypeStorage include five parameters: place,
  /// SparseCooTensorType
  ///
  phi::Place place_;
  dialect::SparseCsrTensorType sparsecsr_tensor_type_;
};

///
/// \brief Define Parametric TypeStorage for AllocatedSelectedRowsTypeStorage.
///
///
struct AllocatedDenseTensorArrayTypeStorage : public pir::TypeStorage {
  using Place = phi::Place;
  ///
  /// \brief Declare ParamKey according to parameter type.
  ///
  using ParamKey = std::tuple<phi::Place, dialect::DenseTensorArrayType>;

  AllocatedDenseTensorArrayTypeStorage(
      const phi::Place& place, const dialect::DenseTensorArrayType& type)
      : place_(place), dense_tensor_array_type_(type) {}

  ///
  /// \brief Each derived TypeStorage must define a Construct method, which
  /// StorageManager uses to construct a derived TypeStorage.
  ///
  static AllocatedDenseTensorArrayTypeStorage* Construct(const ParamKey& key) {
    return new AllocatedDenseTensorArrayTypeStorage(std::get<0>(key),
                                                    std::get<1>(key));
  }

  ///
  /// \brief Each derived TypeStorage must provide a HashValue method.
  ///
  static std::size_t HashValue(const ParamKey& key) {
    std::size_t hash_value = 791;
    // hash place
    hash_value =
        pir::detail::hash_combine(hash_value, std::get<0>(key).HashValue());

    // hash dtype
    auto dense_tensor_array_type = std::get<1>(key);
    hash_value = pir::detail::hash_combine(
        hash_value,
        dialect::DenseTensorArrayTypeStorage::HashValue(
            dialect::DenseTensorArrayTypeStorage::ParamKey(
                dense_tensor_array_type.dtype(),
                dense_tensor_array_type.dims(),
                dense_tensor_array_type.data_layout())));
    return hash_value;
  }

  ///
  /// \brief Each derived TypeStorage needs to overload operator==.
  ///
  bool operator==(const ParamKey& key) const {
    return ParamKey(place_, dense_tensor_array_type_) == key;
  }

  ParamKey GetAsKey() const {
    return ParamKey(place_, dense_tensor_array_type_);
  }

  ///
  /// \brief AllocatedDenseTensorArrayTypeStorage include five parameters:
  /// place, SelectedRowsType
  ///
  phi::Place place_;
  dialect::DenseTensorArrayType dense_tensor_array_type_;
};

}  // namespace dialect
}  // namespace paddle
