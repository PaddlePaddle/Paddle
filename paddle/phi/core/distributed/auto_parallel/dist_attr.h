/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "paddle/phi/common/reduce_type.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/utils/flat_hash_map.h"
#include "paddle/utils/test_macros.h"

namespace phi {
namespace distributed {

namespace auto_parallel {
class TensorDistAttrProto;
}

constexpr int kReplicateDim = -1;

class PlacementStatus {
 public:
  virtual ~PlacementStatus() = default;

  virtual bool is_shard(int64_t axis = -1) const { return false; }
  virtual bool is_partial() const { return false; }
  virtual bool is_replicated() const { return false; }
};

class ReplicatedStatus final : public PlacementStatus {
 public:
  bool is_replicated() const override { return true; }
};

class PartialStatus final : public PlacementStatus {
 public:
  PartialStatus(ReduceType type) : type_(type) {}
  bool is_partial() const override { return true; }
  ReduceType get_reduce_type() const { return type_; }

 private:
  ReduceType type_{ReduceType::kRedSum};
};

class ShardStatus final : public PlacementStatus {
 public:
  ShardStatus(int64_t axis) : axis_(axis) {}
  bool is_shard(int64_t axis = -1) const override {
    if (axis == -1) {
      return true;
    } else {
      return axis == axis_;
    }
  }
  int64_t get_axis() const { return axis_; }

 private:
  int64_t axis_{-1};
};

class TEST_API TensorDistAttr {
 public:
  TensorDistAttr() = default;

  explicit TensorDistAttr(const std::vector<int64_t>& tensor_shape);

  TensorDistAttr(const TensorDistAttr& dist_attr);

  TensorDistAttr& operator=(const TensorDistAttr& dist_attr);

  void copy_from(const TensorDistAttr& dist_attr);

  const ProcessMesh& process_mesh() const { return process_mesh_; }

  void set_process_mesh(const ProcessMesh& process_mesh);

  const std::vector<int64_t>& dims_mapping() const { return dims_mapping_; }

  void set_dims_mapping(const std::vector<int64_t>& dims_mapping);

  // return vector of mesh dims on which the this tensor is partial on
  const std::set<int64_t> partial_dims() const;

  const paddle::flat_hash_map<int64_t, ReduceType>& partial_status() const {
    return partial_status_;
  }

  // by map
  void set_partial_status(
      const paddle::flat_hash_map<int64_t, ReduceType>& partial_status);

  // by each dim
  void set_partial_status(const std::vector<int64_t>& dims,
                          const ReduceType& type = ReduceType::kRedSum);
  // all
  void clean_partial_status();

  // clean by dims
  void clean_partial_dims(const std::vector<int64_t>& dims);

  void set_default_dims_mapping(const std::vector<int64_t>& tensor_shape);

  int64_t batch_dim() const { return batch_dim_; }

  void set_batch_dim(int64_t batch_dim);

  const int64_t& chunk_id() const { return chunk_id_; }

  void set_chunk_id(const int64_t& chunk_id);

  const std::vector<bool>& dynamic_dims() const { return dynamic_dims_; }

  void set_dynamic_dims(const std::vector<bool>& dynamic_dims);

  void set_default_dynamic_dims(const std::vector<int64_t>& tensor_shape);

  const std::map<std::string, bool>& annotated() const { return annotated_; }

  void set_annotated(const std::map<std::string, bool>& annotated);

  bool is_annotated(const std::string& name) const {
    return annotated_.count(name) == 1 && annotated_.at(name) == true;
  }

  void mark_annotated(const std::string& name);

  void clear_annotated() { annotated_.clear(); }

  bool verify_process_mesh(const ProcessMesh& process_mesh) const;

  bool verify_dims_mapping(const std::vector<int64_t>& dims_mapping,
                           const std::vector<int64_t>& tensor_shape) const;

  bool verify_batch_dim(int64_t dim,
                        const std::vector<int64_t>& tensor_shape) const;

  bool verify_dynamic_dims(const std::vector<bool>& dynamic_dims,
                           const std::vector<int64_t>& tensor_shape) const;

  bool verify_annotated(const std::map<std::string, bool>& annotated) const;

  bool verify_partial_status() const;

  bool verify(const std::vector<int64_t>& tensor_shape) const;
  bool verify_dynamic(const std::vector<int64_t>& tensor_shape) const;

  // TensorDistAttr from_string(const std::string& dist_str);
  std::string to_string() const;
  std::string partial_status_string() const;

  // in partial-support-stage-I partial will always be a runtime attribute,
  // there is not need to serialize it. support the partial serialization in
  // future partial-support-stage-II.
  void from_proto(const auto_parallel::TensorDistAttrProto& proto);

  void to_proto(auto_parallel::TensorDistAttrProto* proto) const;

  std::string serialize_to_string();

  void parse_from_string(const std::string& data);

  bool empty() const;

  std::vector<std::shared_ptr<PlacementStatus>> to_placement() const;

  // if mesh_axis is -1, check if tensor is replicated on whole process_mesh
  // if mesh_axis is not -1, check only on specific axis.
  bool is_replicated(int64_t mesh_axis = -1) const;

  // if mesh_axis is -1, check if tensor is shard on whole process_mesh
  // if mesh_axis is not -1, check only on specific axis
  // if tensor_axis is not -1, return true only if the shard axis equal to
  // tensor_axis.
  bool is_shard(int64_t mesh_axis = -1, int64_t tensor_axis = -1) const;

  // if mesh_axis is -1, check if tensor is partial on whole process_mesh
  // if mesh_axis is not -1, check only on specific axis.
  bool is_partial(int64_t mesh_axis = -1) const;

 private:
  static std::vector<std::string> fields_;
  ProcessMesh process_mesh_;
  std::vector<int64_t> dims_mapping_;
  int64_t batch_dim_{0};
  std::vector<bool> dynamic_dims_;
  std::map<std::string, bool> annotated_;
  int64_t chunk_id_{0};
  // partial map would be small (less than mesh.size)
  // iterate operation (copy and comparision) would more frequency than random
  // element access. <key: dim on mesh, value: reduce type>
  paddle::flat_hash_map<int64_t, ReduceType> partial_status_;
};

inline std::ostream& operator<<(std::ostream& os, const TensorDistAttr& obj) {
  os << obj.to_string();
  return os;
}

bool operator==(const TensorDistAttr& lhs, const TensorDistAttr& rhs);

inline bool operator!=(const TensorDistAttr& lhs, const TensorDistAttr& rhs) {
  return !operator==(lhs, rhs);
}

}  // namespace distributed
}  // namespace phi
