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
#include "paddle/pir/core/builtin_type_storage.h"
#include "paddle/pir/core/type.h"
#include "paddle/pir/core/type_base.h"
#include "paddle/pir/core/utils.h"

namespace paddle {
namespace dialect {
using DenseTensorTypeStorage = pir::DenseTensorTypeStorage;

struct SelectedRowsTypeStorage : public pir::TypeStorage {
  using DataLayout = phi::DataLayout;
  using Dim = phi::DDim;
  using LoD = std::vector<std::vector<size_t>>;
  ///
  /// \brief Declare ParamKey according to parameter type.
  ///
  using ParamKey =
      std::tuple<pir::Type, phi::DDim, phi::DataLayout, phi::LoD, size_t>;

  SelectedRowsTypeStorage(const pir::Type& dtype,
                          const phi::DDim& dims,
                          const phi::DataLayout& layout,
                          const phi::LoD& lod,
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
    hash_value =
        pir::hash_combine(hash_value, std::hash<pir::Type>()(std::get<0>(key)));
    // hash dims
    hash_value =
        pir::hash_combine(hash_value, std::hash<phi::DDim>()(std::get<1>(key)));
    // hash layout
    hash_value = pir::hash_combine(
        hash_value,
        std::hash<std::underlying_type<phi::DataLayout>::type>()(
            static_cast<std::underlying_type<phi::DataLayout>::type>(
                std::get<2>(key))));
    // hash lod
    hash_value =
        pir::hash_combine(hash_value, std::hash<phi::LoD>()(std::get<3>(key)));
    // hash offset
    hash_value =
        pir::hash_combine(hash_value, std::hash<size_t>()(std::get<4>(key)));
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
  phi::LoD lod_;
  size_t offset_;
};

}  // namespace dialect
}  // namespace paddle
