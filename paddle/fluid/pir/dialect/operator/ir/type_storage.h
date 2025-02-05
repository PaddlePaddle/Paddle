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

#include "paddle/phi/core/tensor_meta.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/builtin_type_storage.h"
#include "paddle/pir/include/core/type.h"
#include "paddle/pir/include/core/type_base.h"
#include "paddle/pir/include/core/utils.h"

namespace paddle {
namespace dialect {
using DenseTensorTypeStorage = pir::DenseTensorTypeStorage;

struct SelectedRowsTypeStorage : public pir::TypeStorage {
  using DataLayout = phi::DataLayout;
  using Dim = phi::DDim;
  using LegacyLoD = std::vector<std::vector<size_t>>;
  ///
  /// \brief Declare ParamKey according to parameter type.
  ///
  using ParamKey =
      std::tuple<pir::Type, phi::DDim, phi::DataLayout, phi::LegacyLoD, size_t>;

  SelectedRowsTypeStorage(const pir::Type& dtype,
                          const phi::DDim& dims,
                          const phi::DataLayout& layout,
                          const phi::LegacyLoD& lod,
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
  static SelectedRowsTypeStorage* Construct(const ParamKey& key) {
    return new SelectedRowsTypeStorage(std::get<0>(key),
                                       std::get<1>(key),
                                       std::get<2>(key),
                                       std::get<3>(key),
                                       std::get<4>(key));
  }

  ///
  /// \brief Each derived TypeStorage must provide a HashValue method.
  ///
  static std::size_t HashValue(const ParamKey& key) {
    std::size_t hash_value = 317;
    // hash dtype
    hash_value = pir::detail::hash_combine(
        hash_value, std::hash<pir::Type>()(std::get<0>(key)));
    // hash dims
    hash_value = pir::detail::hash_combine(
        hash_value, std::hash<phi::DDim>()(std::get<1>(key)));
    // hash layout
    hash_value = pir::detail::hash_combine(
        hash_value,
        std::hash<std::underlying_type<phi::DataLayout>::type>()(
            static_cast<std::underlying_type<phi::DataLayout>::type>(
                std::get<2>(key))));
    // hash lod
    hash_value = pir::detail::hash_combine(
        hash_value, std::hash<phi::LegacyLoD>()(std::get<3>(key)));
    // hash offset
    hash_value = pir::detail::hash_combine(
        hash_value, std::hash<size_t>()(std::get<4>(key)));
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
  phi::DDim dims_;
  phi::DataLayout layout_;
  phi::LegacyLoD lod_;
  size_t offset_;
};

struct DenseTensorArrayTypeStorage : public pir::TypeStorage {
  using DataLayout = phi::DataLayout;
  using DDim = phi::DDim;
  ///
  /// \brief Declare ParamKey according to parameter type.
  ///
  using ParamKey = std::tuple<pir::Type, phi::DDim, phi::DataLayout>;

  DenseTensorArrayTypeStorage(const pir::Type& dtype,
                              const phi::DDim& dims,
                              const phi::DataLayout& layout)
      : dtype_(dtype), dims_(dims), layout_(layout) {}

  ///
  /// \brief Each derived TypeStorage must define a Construct method, which
  /// StorageManager uses to construct a derived TypeStorage.
  ///
  static DenseTensorArrayTypeStorage* Construct(const ParamKey& key) {
    return new DenseTensorArrayTypeStorage(
        std::get<0>(key), std::get<1>(key), std::get<2>(key));
  }

  ///
  /// \brief Each derived TypeStorage must provide a HashValue method.
  ///
  static std::size_t HashValue(const ParamKey& key) {
    std::size_t hash_value = 317;
    // hash dtype
    hash_value = pir::detail::hash_combine(
        hash_value, std::hash<pir::Type>()(std::get<0>(key)));
    // hash dims
    hash_value = pir::detail::hash_combine(
        hash_value, std::hash<phi::DDim>()(std::get<1>(key)));
    // hash layout
    hash_value = pir::detail::hash_combine(
        hash_value,
        std::hash<std::underlying_type<phi::DataLayout>::type>()(
            static_cast<std::underlying_type<phi::DataLayout>::type>(
                std::get<2>(key))));
    return hash_value;
  }

  ///
  /// \brief Each derived TypeStorage needs to overload operator==.
  ///
  bool operator==(const ParamKey& key) const {
    return ParamKey(dtype_, dims_, layout_) == key;
  }

  ParamKey GetAsKey() const { return ParamKey(dtype_, dims_, layout_); }

  ///
  /// \brief DenseTensorTypeStorage include five parameters: dtype, layout
  ///
  pir::Type dtype_;
  phi::DDim dims_;
  phi::DataLayout layout_;
};

struct SparseCooTensorTypeStorage : public pir::TypeStorage {
  ///
  /// \brief Declare ParamKey according to parameter type.
  ///
  using ParamKey = std::tuple<pir::Type,
                              common::DDim,
                              common::DDim,
                              common::DataLayout,
                              pir::DenseTensorType,
                              pir::DenseTensorType,
                              bool>;
  SparseCooTensorTypeStorage(pir::Type dtype,
                             common::DDim dims,
                             common::DDim non_zero_dims,
                             common::DataLayout layout,
                             pir::DenseTensorType non_zero_indices,
                             pir::DenseTensorType non_zero_elements,
                             bool coalesced = false)
      : dtype_(dtype),
        dims_(dims),
        non_zero_dims_(non_zero_dims),
        layout_(layout),
        non_zero_indices_(non_zero_indices),
        non_zero_elements_(non_zero_elements),
        coalesced_(coalesced) {}

  ///
  /// \brief Each derived TypeStorage must define a Construct method, which
  /// StorageManager uses to construct a derived TypeStorage.
  ///
  static SparseCooTensorTypeStorage* Construct(const ParamKey& key) {
    return new SparseCooTensorTypeStorage(std::get<0>(key),
                                          std::get<1>(key),
                                          std::get<2>(key),
                                          std::get<3>(key),
                                          std::get<4>(key),
                                          std::get<5>(key),
                                          std::get<6>(key));
  }

  ///
  /// \brief Each derived TypeStorage must provide a HashValue method.
  ///
  static std::size_t HashValue(const ParamKey& key) {
    std::size_t hash_value = 0;
    // hash dtype
    hash_value = pir::detail::hash_combine(
        hash_value, std::hash<pir::Type>()(std::get<0>(key)));
    // hash dims
    hash_value = pir::detail::hash_combine(
        hash_value, std::hash<common::DDim>()(std::get<1>(key)));
    // hash non_zero_dims
    hash_value = pir::detail::hash_combine(
        hash_value, std::hash<common::DDim>()(std::get<2>(key)));
    // hash layout
    hash_value = pir::detail::hash_combine(
        hash_value,
        std::hash<std::underlying_type<DataLayout>::type>()(
            static_cast<std::underlying_type<DataLayout>::type>(
                std::get<3>(key))));
    // hash DenseTensorType
    auto tuple1 = std::make_tuple(std::get<4>(key).dtype(),
                                  std::get<4>(key).dims(),
                                  std::get<4>(key).data_layout(),
                                  std::get<4>(key).lod(),
                                  std::get<4>(key).offset());
    hash_value = pir::detail::hash_combine(
        hash_value, DenseTensorTypeStorage::HashValue(tuple1));
    // hash DenseTensorType
    auto tuple2 = std::make_tuple(std::get<5>(key).dtype(),
                                  std::get<5>(key).dims(),
                                  std::get<5>(key).data_layout(),
                                  std::get<5>(key).lod(),
                                  std::get<5>(key).offset());
    hash_value = pir::detail::hash_combine(
        hash_value, DenseTensorTypeStorage::HashValue(tuple2));
    // hash coalesced
    hash_value = pir::detail::hash_combine(hash_value,
                                           std::hash<bool>()(std::get<6>(key)));

    return hash_value;
  }

  ///
  /// \brief Each derived TypeStorage needs to overload operator==.
  ///
  bool operator==(const ParamKey& key) const {
    return ParamKey(dtype_,
                    dims_,
                    non_zero_dims_,
                    layout_,
                    non_zero_indices_,
                    non_zero_elements_,
                    coalesced_) == key;
  }

  ParamKey GetAsKey() const {
    return ParamKey(dtype_,
                    dims_,
                    non_zero_dims_,
                    layout_,
                    non_zero_indices_,
                    non_zero_elements_,
                    coalesced_);
  }

  ///
  /// \brief SparseCooTensorTypeStorage include six parameters: dims, dtype,
  /// layout, non_zero_indices_, non_zero_elements_,coalesced_.
  ///

  pir::Type dtype_;
  common::DDim dims_;
  common::DDim non_zero_dims_;
  common::DataLayout layout_{DataLayout::NCHW};
  pir::DenseTensorType non_zero_indices_;
  pir::DenseTensorType non_zero_elements_;
  bool coalesced_ = false;
};

struct SparseCsrTensorTypeStorage : public pir::TypeStorage {
  ///
  /// \brief Declare ParamKey according to parameter type.
  ///
  using ParamKey = std::tuple<pir::Type,
                              common::DDim,
                              common::DataLayout,
                              pir::DenseTensorType,
                              pir::DenseTensorType,
                              pir::DenseTensorType>;
  SparseCsrTensorTypeStorage(pir::Type dtype,
                             common::DDim dims,
                             common::DataLayout layout,
                             pir::DenseTensorType non_zero_crows,
                             pir::DenseTensorType non_zero_cols,
                             pir::DenseTensorType non_zero_elements)
      : dtype_(dtype),
        dims_(dims),
        layout_(layout),
        non_zero_crows_(non_zero_crows),
        non_zero_cols_(non_zero_cols),
        non_zero_elements_(non_zero_elements) {}

  ///
  /// \brief Each derived TypeStorage must define a Construct method, which
  /// StorageManager uses to construct a derived TypeStorage.
  ///
  static SparseCsrTensorTypeStorage* Construct(const ParamKey& key) {
    return new SparseCsrTensorTypeStorage(std::get<0>(key),
                                          std::get<1>(key),
                                          std::get<2>(key),
                                          std::get<3>(key),
                                          std::get<4>(key),
                                          std::get<5>(key));
  }

  ///
  /// \brief Each derived TypeStorage must provide a HashValue method.
  ///
  static std::size_t HashValue(const ParamKey& key) {
    std::size_t hash_value = 0;
    // hash dtype
    hash_value = pir::detail::hash_combine(
        hash_value, std::hash<pir::Type>()(std::get<0>(key)));
    // hash dims
    hash_value = pir::detail::hash_combine(
        hash_value, std::hash<common::DDim>()(std::get<1>(key)));
    // hash layout
    hash_value = pir::detail::hash_combine(
        hash_value,
        std::hash<std::underlying_type<DataLayout>::type>()(
            static_cast<std::underlying_type<DataLayout>::type>(
                std::get<2>(key))));
    // hash DenseTensorType
    auto tuple1 = std::make_tuple(std::get<3>(key).dtype(),
                                  std::get<3>(key).dims(),
                                  std::get<3>(key).data_layout(),
                                  std::get<3>(key).lod(),
                                  std::get<3>(key).offset());
    hash_value = pir::detail::hash_combine(
        hash_value, DenseTensorTypeStorage::HashValue(tuple1));
    // hash DenseTensorType
    auto tuple2 = std::make_tuple(std::get<4>(key).dtype(),
                                  std::get<4>(key).dims(),
                                  std::get<4>(key).data_layout(),
                                  std::get<4>(key).lod(),
                                  std::get<4>(key).offset());
    hash_value = pir::detail::hash_combine(
        hash_value, DenseTensorTypeStorage::HashValue(tuple2));
    // hash DenseTensorType
    auto tuple3 = std::make_tuple(std::get<5>(key).dtype(),
                                  std::get<5>(key).dims(),
                                  std::get<5>(key).data_layout(),
                                  std::get<5>(key).lod(),
                                  std::get<5>(key).offset());
    hash_value = pir::detail::hash_combine(
        hash_value, DenseTensorTypeStorage::HashValue(tuple3));
    return hash_value;
  }

  ///
  /// \brief Each derived TypeStorage needs to overload operator==.
  ///
  bool operator==(const ParamKey& key) const {
    return ParamKey(dtype_,
                    dims_,
                    layout_,
                    non_zero_crows_,
                    non_zero_cols_,
                    non_zero_elements_) == key;
  }

  ParamKey GetAsKey() const {
    return ParamKey(dtype_,
                    dims_,
                    layout_,
                    non_zero_crows_,
                    non_zero_cols_,
                    non_zero_elements_);
  }

  ///
  /// \brief SparseCsrTensorTypeStorage include six parameters: dims, dtype,
  /// layout, non_zero_crows_,non_zero_cols_,non_zero_elements_.
  ///

  pir::Type dtype_;
  common::DDim dims_;
  common::DataLayout layout_;
  pir::DenseTensorType non_zero_crows_;
  pir::DenseTensorType non_zero_cols_;
  pir::DenseTensorType non_zero_elements_;
};

}  // namespace dialect
}  // namespace paddle
