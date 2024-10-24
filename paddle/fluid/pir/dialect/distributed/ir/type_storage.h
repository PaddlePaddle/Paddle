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

#include <vector>

#include "paddle/fluid/pir/dialect/distributed/ir/dist_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"

namespace paddle {
namespace dialect {
///
/// \brief Define Parametric TypeStorage for DistDenseTensorType.
///
class DistDenseTensorTypeStorage : public pir::TypeStorage {
 public:
  ///
  /// \brief Declare ParamKey according to parameter type.
  ///
  using ParamKey =
      std::tuple<pir::DenseTensorType, TensorDistAttribute, common::DDim>;

  DistDenseTensorTypeStorage(pir::DenseTensorType dense_tensor_type,
                             TensorDistAttribute tensor_dist_attr,
                             const common::DDim& local_ddim)
      : dense_tensor_type(dense_tensor_type),
        tensor_dist_attr(tensor_dist_attr),
        local_ddim(local_ddim) {}

  ///
  /// \brief Each derived TypeStorage must define a Construct method, which
  /// StorageManager uses to construct a derived TypeStorage.
  ///
  static DistDenseTensorTypeStorage* Construct(ParamKey&& key) {
    return new DistDenseTensorTypeStorage(
        std::get<0>(key), std::get<1>(key), std::get<2>(key));
  }

  ///
  /// \brief Each derived TypeStorage must provide a HashValue method.
  ///
  static std::size_t HashValue(const ParamKey& key) {
    auto dense_tensor_type_hash = std::hash<pir::Type>()(std::get<0>(key));
    auto tensor_dist_attr_hash = std::hash<pir::Attribute>()(std::get<1>(key));
    auto local_ddim_hash = std::hash<common::DDim>()(std::get<2>(key));
    auto value = pir::detail::hash_combine(dense_tensor_type_hash,
                                           tensor_dist_attr_hash);
    return pir::detail::hash_combine(value, local_ddim_hash);
  }

  ///
  /// \brief Each derived TypeStorage needs to overload operator==.
  ///
  bool operator==(const ParamKey& key) const {
    return dense_tensor_type == std::get<0>(key) &&
           tensor_dist_attr == std::get<1>(key) &&
           local_ddim == std::get<2>(key);
  }

  ///
  /// \brief DistDenseTensorTypeStorage include three parameters:
  /// dense_tensor_type, tensor_dist_attr and local_ddim;
  ///
  pir::DenseTensorType dense_tensor_type;
  TensorDistAttribute tensor_dist_attr;
  common::DDim local_ddim;
};

}  // namespace dialect
}  // namespace paddle
