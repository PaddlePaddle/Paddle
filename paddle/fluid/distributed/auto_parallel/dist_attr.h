/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include <vector>

#include "paddle/fluid/distributed/auto_parallel/auto_parallel.pb.h"
#include "paddle/fluid/distributed/auto_parallel/process_mesh.h"
#include "paddle/fluid/distributed/auto_parallel/utils.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

using framework::BlockDesc;
using framework::OpDesc;
using framework::ProgramDesc;
using framework::VarDesc;

class TensorDistributedAttribute {
 public:
  TensorDistributedAttribute() = default;

  explicit TensorDistributedAttribute(const VarDesc& tensor_desc);

  const VarDesc& tensor_desc() const { return *tensor_desc_; }

  const ProcessMesh& process_mesh() const { return process_mesh_; }

  void set_process_mesh(const ProcessMesh& process_mesh) {
    if (!process_mesh_.shape().empty()) {
      for (int64_t dim_mapping : dims_mapping_) {
        PADDLE_ENFORCE(dim_mapping >= -1 && dim_mapping < process_mesh_.ndim());
      }
    }
    process_mesh_ = process_mesh;
  }

  const std::vector<int64_t>& dims_mapping() const { return dims_mapping_; }

  void set_dims_mapping(const std::vector<int64_t>& dims_mapping) {
    std::vector<int64_t> tensor_shape = tensor_desc_->GetShape();
    PADDLE_ENFORCE_EQ(
        dims_mapping.size(),
        tensor_shape.size(),
        platform::errors::InvalidArgument(
            "The dims_mapping size %d of process mesh must be equal to the "
            "shape size %d of tensor.",
            dims_mapping.size(),
            tensor_shape.size()));
    PADDLE_ENFORCE_EQ(
        has_duplicates(dims_mapping),
        false,
        platform::errors::InvalidArgument(
            "The dims_mapping [%s] must be unique.", str_join(dims_mapping)));
    if (!process_mesh_.shape().empty()) {
      for (int64_t dim_mapping : dims_mapping) {
        PADDLE_ENFORCE(dim_mapping >= -1 && dim_mapping < process_mesh_.ndim());
      }
    }
    dims_mapping_ = dims_mapping;
  }

  void set_default_dims_mapping() {
    std::vector<int64_t> tensor_shape = tensor_desc_->GetShape();
    dims_mapping_ = std::vector<int64_t>(tensor_shape.size(), -1);
  }

  int64_t batch_dim() const { return batch_dim_; }

  void set_batch_dim(int64_t batch_dim) {
    std::vector<int64_t> tensor_shape = tensor_desc_->GetShape();
    int64_t canonical_batch_dim = canonical_dim(batch_dim, tensor_shape.size());
    batch_dim_ = canonical_batch_dim;
  }

  const std::vector<bool>& dynamic_dims() const { return dynamic_dims_; }

  void set_dynamic_dims(const std::vector<bool>& dynamic_dims) {
    std::vector<int64_t> tensor_shape = tensor_desc_->GetShape();
    PADDLE_ENFORCE_EQ(
        dynamic_dims.size(),
        tensor_shape.size(),
        platform::errors::InvalidArgument(
            "The dynamic_dims size of process mesh must be equal to the "
            "shape size of tensor.",
            dynamic_dims.size(),
            tensor_shape.size()));
    dynamic_dims_ = dynamic_dims;
  }

  TensorDistributedAttribute from_string(const std::string& dist_str);
  std::string to_string() const;

  TensorDistributedAttribute from_proto(
      const TensorDistributedAttributeProto& proto);
  TensorDistributedAttributeProto to_proto() const;

 private:
  const VarDesc* tensor_desc_;
  ProcessMesh process_mesh_;
  std::vector<int64_t> dims_mapping_;
  int64_t batch_dim_;
  std::vector<bool> dynamic_dims_;
  std::map<std::string, bool> annotated_;
};

inline std::ostream& operator<<(std::ostream& os,
                                const TensorDistributedAttribute& obj) {
  os << obj.to_string();
  return os;
}

inline bool operator==(const TensorDistributedAttribute& lhs,
                       const TensorDistributedAttribute& rhs) {
  if (!(lhs.tensor_desc() == rhs.tensor_desc())) {
    return false;
  }
  if (lhs.process_mesh() != rhs.process_mesh()) {
    return false;
  }
  if (lhs.dims_mapping() != rhs.dims_mapping()) {
    return false;
  }
  if (lhs.batch_dim() != rhs.batch_dim()) {
    return false;
  }
  if (lhs.dynamic_dims() != rhs.dynamic_dims()) {
    return false;
  }
  return true;
}

inline bool operator!=(const TensorDistributedAttribute& lhs,
                       const TensorDistributedAttribute& rhs) {
  return !operator==(lhs, rhs);
}

class OperatorDistributedAttribute {
 public:
  OperatorDistributedAttribute() = default;

  explicit OperatorDistributedAttribute(const OpDesc& op_desc);

  const OpDesc& op_desc() const { return *op_desc_; }

  const VarDesc& input(const std::string& name) const {
    return *inputs_.at(name);
  }

  const VarDesc& output(const std::string& name) const {
    return *outputs_.at(name);
  }

  const std::map<std::string, TensorDistributedAttribute>& input_dist_attrs()
      const {
    return input_dist_attrs_;
  }

  const std::map<std::string, TensorDistributedAttribute>& output_dist_attrs()
      const {
    return output_dist_attrs_;
  }

  const TensorDistributedAttribute& input_dist_attr(
      const std::string& name) const {
    return input_dist_attrs_.at(name);
  }

  TensorDistributedAttribute& input_dist_attr(const std::string& name) {
    return input_dist_attrs_.at(name);
  }

  void set_input_dist_attr(const std::string& name,
                           const TensorDistributedAttribute& dist_attr) {
    if (input_dist_attrs_.count(name) == 1) {
      input_dist_attrs_[name] = dist_attr;
    }
  }

  const TensorDistributedAttribute& output_dist_attr(
      const std::string& name) const {
    return output_dist_attrs_.at(name);
  }

  TensorDistributedAttribute& output_dist_attr(const std::string& name) {
    for (const auto& item : output_dist_attrs_) {
      std::cout << item.first << std::endl;
    }
    return output_dist_attrs_.at(name);
  }

  void set_output_dist_attr(const std::string& name,
                            const TensorDistributedAttribute& dist_attr) {
    if (output_dist_attrs_.count(name) == 1) {
      output_dist_attrs_[name] = dist_attr;
    }
  }

  const ProcessMesh& process_mesh() const { return process_mesh_; }

  void set_process_mesh(const ProcessMesh& process_mesh) {
    for (auto& item : input_dist_attrs_) {
      item.second.set_process_mesh(process_mesh);
    }
    for (auto& item : output_dist_attrs_) {
      item.second.set_process_mesh(process_mesh);
    }
    process_mesh_ = process_mesh;
  }

  const std::string& impl_type() const { return impl_type_; }
  void set_impl_type(const std::string& impl_type) { impl_type_ = impl_type; }

  int64_t impl_idx() const { return impl_idx_; }
  void set_impl_idx(const int64_t& impl_idx) { impl_idx_ = impl_idx; }

  OperatorDistributedAttribute from_string(const std::string& dist_str);
  std::string to_string() const;

  OperatorDistributedAttribute from_proto(
      const OperatorDistributedAttributeProto& proto);
  OperatorDistributedAttributeProto to_proto() const;

 private:
  const OpDesc* op_desc_;
  std::map<std::string, VarDesc*> inputs_;
  std::map<std::string, VarDesc*> outputs_;
  std::map<std::string, TensorDistributedAttribute> input_dist_attrs_;
  std::map<std::string, TensorDistributedAttribute> output_dist_attrs_;
  ProcessMesh process_mesh_;
  std::string impl_type_;
  int64_t impl_idx_;
};

inline std::ostream& operator<<(std::ostream& os,
                                const OperatorDistributedAttribute& obj) {
  os << obj.to_string();
  return os;
}

inline bool operator==(const OperatorDistributedAttribute& lhs,
                       const OperatorDistributedAttribute& rhs) {
  for (auto const& item : lhs.input_dist_attrs()) {
    if (!(rhs.input(item.first) == lhs.input(item.first))) {
      return false;
    }
    if (rhs.input_dist_attrs().count(item.first) != 1) {
      return false;
    }
    if (rhs.input_dist_attrs().at(item.first) !=
        lhs.input_dist_attrs().at(item.first)) {
      return false;
    }
  }
  for (auto const& item : lhs.output_dist_attrs()) {
    if (!(rhs.output(item.first) == lhs.output(item.first))) {
      return false;
    }
    if (rhs.output_dist_attrs().count(item.first) != 1) {
      return false;
    }
    if (rhs.output_dist_attrs().at(item.first) !=
        lhs.output_dist_attrs().at(item.first)) {
      return false;
    }
  }
  if (lhs.process_mesh() != rhs.process_mesh()) {
    return false;
  }
  if (lhs.impl_type() != rhs.impl_type()) {
    return false;
  }
  if (lhs.impl_idx() != rhs.impl_idx()) {
    return false;
  }
  return true;
}

inline bool operator!=(const OperatorDistributedAttribute& lhs,
                       const OperatorDistributedAttribute& rhs) {
  return !operator==(lhs, rhs);
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
