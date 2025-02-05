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

#include "paddle/common/ddim.h"
#include "paddle/common/hash_funcs.h"
#include "paddle/common/layout.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_attribute.h"
#include "paddle/phi/common/reduce_type.h"
#include "paddle/pir/include/core/attribute_base.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/utils.h"
#include "paddle/utils/flat_hash_map.h"

namespace paddle {
namespace dialect {

class ProcessMeshAttrStorage : public pir::AttributeStorage {
 public:
  ///
  /// \brief Declare ParamKey according to parameter type.
  ///
  using ParamKey = phi::distributed::ProcessMesh;

  ProcessMeshAttrStorage(ParamKey&& process_mesh)  // NOLINT
      : process_mesh(std::move(process_mesh)) {}

  ///
  /// \brief Each derived TypeStorage must define a Construct method, which
  /// StorageManager uses to construct a derived TypeStorage.
  ///
  static ProcessMeshAttrStorage* Construct(ParamKey&& key) {
    return new ProcessMeshAttrStorage(std::move(key));
  }

  ///
  /// \brief Each derived TypeStorage must provide a HashValue method.
  ///
  static std::size_t HashValue(const ParamKey& key) { return key.hash(); }

  ///
  /// \brief Each derived TypeStorage needs to overload operator==.
  ///
  bool operator==(const ParamKey& key) const {
    return process_mesh == key && process_mesh.dim_names() == key.dim_names();
  }

  ParamKey process_mesh;
};

// NOTE (pkuzyc): now the Placements is only used for the ``local_reshape``
// op, for the case when one tensor dimension is sharded by multiple mesh
// dimensions. In the future, the dims_mapping in TensorDistAttr will be
// replaced by this Placements.
class PlacementsAttrStorage : public pir::AttributeStorage {
 public:
  ///
  /// \brief Declare ParamKey according to parameter type.
  ///
  using ParamKey = phi::distributed::Placements;

  PlacementsAttrStorage(ParamKey&& placements)  // NOLINT
      : placements(std::move(placements)) {}

  ///
  /// \brief Each derived TypeStorage must define a Construct method, which
  /// StorageManager uses to construct a derived TypeStorage.
  ///
  static PlacementsAttrStorage* Construct(ParamKey&& key) {
    return new PlacementsAttrStorage(std::move(key));
  }

  static std::string to_string(const ParamKey& key) {
    std::string s = "[";
    for (const auto& p : key) {
      s += p->to_string() + ", ";
    }
    s.pop_back();
    s.pop_back();
    return s + "]";
  }

  ///
  /// \brief Each derived TypeStorage must provide a HashValue method.
  ///
  static std::size_t HashValue(const ParamKey& key) {
    return std::hash<std::string>()(to_string(key));
  }

  ///
  /// \brief Each derived TypeStorage needs to overload operator==.
  ///
  bool operator==(const ParamKey& key) const {
    return phi::distributed::equal_placements(placements, key);
  }

  ParamKey placements;
};

class TensorDistAttrStorage : public pir::AttributeStorage {
 public:
  ///
  /// \brief Declare ParamKey according to parameter type.
  ///
  using ParamKey = std::tuple<ProcessMeshAttribute,
                              std::vector<int64_t>,
                              flat_hash_map<int64_t, phi::ReduceType>,
                              std::optional<PlacementsAttribute>>;

  TensorDistAttrStorage(ParamKey&& param)  // NOLINT
      : mesh_attr(std::get<0>(param)),
        dims_mapping(std::move(std::get<1>(param))),
        partial_status(std::move(std::get<2>(param))),
        placements_(std::move(std::get<3>(param))) {}
  ///
  /// \brief Each derived TypeStorage must define a Construct method, which
  /// StorageManager uses to construct a derived TypeStorage.
  ///
  static TensorDistAttrStorage* Construct(ParamKey&& key) {
    return new TensorDistAttrStorage(std::move(key));
  }

  ///
  /// \brief Each derived TypeStorage must provide a HashValue method.
  ///
  static std::size_t HashValue(const ParamKey& key) {
    auto mesh_hash = std::get<0>(key).hash();
    auto dims_map_hash = std::hash<std::vector<int64_t>>()(std::get<1>(key));
    std::string partial_status_str = "[";
    for (auto& itr : std::get<2>(key)) {
      partial_status_str +=
          "Partial(dims:" + std::to_string(itr.first) + ", " +
          phi::ReduceTypeStrings[static_cast<int>(itr.second)] + "), ";
    }
    partial_status_str += "]";
    auto combine_hash = pir::detail::hash_combine(mesh_hash, dims_map_hash);
    if (std::get<3>(key).has_value()) {
      combine_hash =
          pir::detail::hash_combine(combine_hash, std::get<3>(key)->hash());
    }
    return pir::detail::hash_combine(
        combine_hash, std::hash<std::string>()(partial_status_str));
  }

  ///
  /// \brief Each derived TypeStorage needs to overload operator==.
  ///
  bool operator==(const ParamKey& key) const {
    return mesh_attr == std::get<0>(key) && dims_mapping == std::get<1>(key) &&
           partial_status == std::get<2>(key) &&
           placements_ == std::get<3>(key);
  }

  ProcessMeshAttribute mesh_attr;
  std::vector<int64_t> dims_mapping;
  // partial map would less or equal than to mesh.size.
  // iterate operation (copy and comparison) would more frequency than random
  // element access. <key: dim on mesh, value: reduce type>
  flat_hash_map<int64_t, phi::ReduceType> partial_status;
  std::optional<PlacementsAttribute> placements_;
};

class OperationDistAttrStorage : public pir::AttributeStorage {
 public:
  ///
  /// \brief Declare ParamKey according to parameter type.
  ///
  using ParamKey = std::tuple<ProcessMeshAttribute,
                              std::vector<pir::Attribute>,
                              std::vector<pir::Attribute>,
                              int64_t>;

  explicit OperationDistAttrStorage(ParamKey&& param)
      : mesh_attr(std::get<0>(param)),
        operands(std::get<1>(param)),
        results(std::get<2>(param)),
        chunk_id(std::get<3>(param)) {}

  ///
  /// \brief Each derived TypeStorage must define a Construct method, which
  /// StorageManager uses to construct a derived TypeStorage.
  ///
  static OperationDistAttrStorage* Construct(ParamKey&& key) {
    return new OperationDistAttrStorage(std::move(key));
  }

  ///
  /// \brief Each derived TypeStorage must provide a HashValue method.
  ///
  static std::size_t HashValue(const ParamKey& key) {
    auto hash_value = std::hash<pir::Attribute>()(std::get<0>(key));
    for (auto& iter : std::get<1>(key)) {
      auto tmp_value = std::hash<pir::Attribute>()(iter);
      hash_value = pir::detail::hash_combine(hash_value, tmp_value);
    }
    for (auto& iter : std::get<2>(key)) {
      auto tmp_value = std::hash<pir::Attribute>()(iter);
      hash_value = pir::detail::hash_combine(hash_value, tmp_value);
    }
    hash_value = pir::detail::hash_combine(hash_value, std::get<3>(key));
    return hash_value;
  }

  ///
  /// \brief Each derived TypeStorage needs to overload operator==.
  ///
  bool operator==(const ParamKey& key) const {
    return mesh_attr == std::get<0>(key) && operands == std::get<1>(key) &&
           results == std::get<2>(key) && chunk_id == std::get<3>(key);
  }

  ProcessMeshAttribute mesh_attr;
  std::vector<pir::Attribute> operands;
  std::vector<pir::Attribute> results;
  int64_t chunk_id;
};

}  // namespace dialect
}  // namespace paddle
