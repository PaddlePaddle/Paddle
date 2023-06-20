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
#include <string>
#include <vector>

#include "paddle/phi/core/distributed/auto_parallel/auto_parallel.pb.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace distributed {
namespace auto_parallel {

constexpr const char* kDefault = "default";

class TensorDistAttr {
 public:
  TensorDistAttr() = default;

  explicit TensorDistAttr(const std::vector<int64_t>& tensor_shape);

  TensorDistAttr(const TensorDistAttr& tensor);

  TensorDistAttr& operator=(const TensorDistAttr& dist_attr);

  void copy_from(const TensorDistAttr& dist_attr);

  const ProcessMesh& process_mesh() const { return process_mesh_; }

  void set_process_mesh(const ProcessMesh& process_mesh);

  const std::vector<int64_t>& dims_mapping() const { return dims_mapping_; }

  void set_dims_mapping(const std::vector<int64_t>& dims_mapping);

  void set_default_dims_mapping(const std::vector<int64_t>& tensor_shape);

  int64_t batch_dim() const { return batch_dim_; }

  void set_batch_dim(int64_t batch_dim);

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

  bool verify(const std::vector<int64_t>& tensor_shape) const;

  // TensorDistAttr from_string(const std::string& dist_str);
  std::string to_string() const;

  void from_proto(const TensorDistAttrProto& proto);

  TensorDistAttrProto to_proto() const;

  std::string serialize_to_string();

  void parse_from_string(const std::string& data);

 private:
  static std::vector<std::string> fields_;
  ProcessMesh process_mesh_;
  std::vector<int64_t> dims_mapping_;
  int64_t batch_dim_{0};
  std::vector<bool> dynamic_dims_;
  std::map<std::string, bool> annotated_;
};

inline std::ostream& operator<<(std::ostream& os, const TensorDistAttr& obj) {
  os << obj.to_string();
  return os;
}

bool operator==(const TensorDistAttr& lhs, const TensorDistAttr& rhs);

inline bool operator!=(const TensorDistAttr& lhs, const TensorDistAttr& rhs) {
  return !operator==(lhs, rhs);
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace phi
