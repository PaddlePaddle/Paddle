// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/type.h"
#include "paddle/pir/include/core/type_base.h"
#include "paddle/pir/include/core/utils.h"

namespace paddle {
namespace dialect {
///
/// \brief Define Parametric TypeStorage for SparseCooTensorType.
///
/// NOTE(risemeup1): The derived TypeStorage class needs to implement the
/// following methods: (1)declare ParamKey, (2)define Construction method,
/// (3)define HashValue method, (4)overload operator==.
///

struct SparseCooTensorTypeStorage : public pir::TypeStorage {
  ///
  /// \brief Declare ParamKey according to parameter type.
  ///
  using Dim = pir::DDim;
  using DataLayout = pir::DataLayout;
  using DataType = pir::Type;
  using DenseTensorType = pir::DenseTensorType;
  using ParamKey = std::
      tuple<DataType, Dim, DataLayout, DenseTensorType, DenseTensorType, bool>;
  SparseCooTensorTypeStorage(DataType dtype,
                             Dim dims,
                             DataLayout layout,
                             DenseTensorType non_zero_indices,
                             DenseTensorType non_zero_elements,
                             bool coalesced = false)
      : dtype_(dtype),
        dims_(dims),
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
    hash_value = pir::detail::hash_combine(hash_value,
                                           std::hash<Dim>()(std::get<1>(key)));
    // hash layout
    hash_value = pir::detail::hash_combine(
        hash_value,
        std::hash<std::underlying_type<DataLayout>::type>()(
            static_cast<std::underlying_type<DataLayout>::type>(
                std::get<2>(key))));
    // hash DenseTensorType
    hash_value = pir::detail::hash_combine(
        hash_value, std::hash<DenseTensorType>()(std::get<3>(key)));
    // hash DenseTensorType
    hash_value = pir::detail::hash_combine(
        hash_value, std::hash<DenseTensorType>()(std::get<4>(key)));

    // hash coalesced
    hash_value = pir::detail::hash_combine(hash_value,
                                           std::hash<bool>()(std::get<5>(key)));

    return hash_value;
  }

  ///
  /// \brief Each derived TypeStorage needs to overload operator==.
  ///
  bool operator==(const ParamKey& key) const {
    return ParamKey(dtype_,
                    dims_,
                    layout_,
                    non_zero_indices_,
                    non_zero_elements_,
                    coalesced_) == key;
  }

  ParamKey GetAsKey() const {
    return ParamKey(dtype_,
                    dims_,
                    layout_,
                    non_zero_indices_,
                    non_zero_elements_,
                    coalesced_);
  }

  ///
  /// \brief SparseCooTensorTypeStorage include six parameters: dims, dtype,
  /// layout, non_zero_indices_, non_zero_elements_,coalesced_.
  ///

  DataType dtype_;
  Dim dims_;
  DataLayout layout_{DataLayout::NCHW};
  DenseTensorType non_zero_indices_;
  DenseTensorType non_zero_elements_;
  bool coalesced_ = false;
};
}  // namespace dialect
}  // namespace paddle
