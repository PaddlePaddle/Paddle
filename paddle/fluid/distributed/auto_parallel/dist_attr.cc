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

#include <algorithm>
#include <iostream>
#include <iterator>

#include "paddle/fluid/distributed/auto_parallel/dist_attr.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_desc.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

TensorDistAttr::TensorDistAttr(const VarDesc& tensor)
    : tensor_(&tensor), batch_dim_(0) {
  set_default_dims_mapping();
  std::vector<int64_t> tensor_shape = tensor_->GetShape();
  for (std::size_t i = 0; i < tensor_shape.size(); ++i) {
    dynamic_dims_.push_back(false);
  }
}

void TensorDistAttr::set_process_mesh(const ProcessMesh& process_mesh) {
  PADDLE_ENFORCE_EQ(verify_process_mesh(process_mesh),
                    true,
                    platform::errors::InvalidArgument(
                        "Wrong process mesh %s.", process_mesh.to_string()));
  process_mesh_ = process_mesh;
}

void TensorDistAttr::set_dims_mapping(
    const std::vector<int64_t>& dims_mapping) {
  PADDLE_ENFORCE_EQ(verify_dims_mapping(dims_mapping),
                    true,
                    platform::errors::InvalidArgument("Wrong dims_mapping %s.",
                                                      str_join(dims_mapping)));
  dims_mapping_ = dims_mapping;
}

void TensorDistAttr::set_batch_dim(int64_t batch_dim) {
  std::vector<int64_t> tensor_shape = tensor_->GetShape();
  int64_t canonical_batch_dim = canonical_dim(batch_dim, tensor_shape.size());
  batch_dim_ = canonical_batch_dim;
}

void TensorDistAttr::set_dynamic_dims(const std::vector<bool>& dynamic_dims) {
  std::vector<int64_t> tensor_shape = tensor_->GetShape();
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

void TensorDistAttr::set_default_dims_mapping() {
  std::vector<int64_t> tensor_shape = tensor_->GetShape();
  dims_mapping_ = std::vector<int64_t>(tensor_shape.size(), -1);
}

bool TensorDistAttr::verify_process_mesh(
    const ProcessMesh& process_mesh) const {
  if (!process_mesh_.empty()) {
    for (int64_t dim_mapping : dims_mapping_) {
      if (dim_mapping < -1 || dim_mapping >= process_mesh_.ndim()) {
        return false;
      }
    }
  }
  return true;
}

bool TensorDistAttr::verify_dims_mapping(
    const std::vector<int64_t>& dims_mapping) const {
  std::vector<int64_t> tensor_shape = tensor_->GetShape();
  if (dims_mapping.size() != tensor_shape.size()) {
    return false;
  }
  std::unordered_map<int64_t, int64_t> map;
  if (!process_mesh_.empty()) {
    for (int64_t i : dims_mapping) {
      if (i < -1 || i >= process_mesh_.ndim()) {
        return false;
      }
      ++map[i];
      if (i != -1 && map[i] > 1) {
        return false;
      }
    }
  } else {
    for (int64_t i : dims_mapping) {
      ++map[i];
      if (i != -1 && map[i] > 1) {
        return false;
      }
    }
  }
  return true;
}

bool TensorDistAttr::verify() const {
  if (!verify_process_mesh(process_mesh_)) {
    return false;
  }
  if (!verify_dims_mapping(dims_mapping_)) {
    return false;
  }
  return true;
}

std::string TensorDistAttr::to_string() const {
  std::string dist_str = "{tensor_name: " + tensor_->Name() + ", ";
  dist_str += "process_mesh: " + process_mesh_.to_string() + ", ";
  dist_str += "dims_mappings: [" + str_join(dims_mapping_) + "], ";
  dist_str += "batch_dim: " + std::to_string(batch_dim_) + ", ";
  dist_str += "dynamic_dims: [" + str_join(dynamic_dims_) + "]}";
  return dist_str;
}

bool operator==(const TensorDistAttr& lhs, const TensorDistAttr& rhs) {
  if (!(lhs.tensor() == rhs.tensor())) {
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

OperatorDistAttr::OperatorDistAttr(const OpDesc& op) : op_(&op) {
  for (std::string name : op_->InputArgumentNames()) {
    VarDesc* input = op_->Block()->FindVarRecursive(name);
    // inputs_[name] = input;
    input_dist_attrs_[name] = TensorDistAttr(*input);
  }
  for (std::string name : op_->OutputArgumentNames()) {
    VarDesc* output = op_->Block()->FindVarRecursive(name);
    // outputs_[name] = output;
    output_dist_attrs_[name] = TensorDistAttr(*output);
  }
}

void OperatorDistAttr::set_input_dist_attr(const std::string& name,
                                           const TensorDistAttr& dist_attr) {
  PADDLE_ENFORCE_EQ(
      dist_attr.verify(),
      true,
      platform::errors::InvalidArgument(
          "Wrong dist_attr %s for %s.", dist_attr.to_string(), name));
  if (input_dist_attrs_.count(name) == 1) {
    input_dist_attrs_[name] = dist_attr;
  }
  // Make sure the process mesh of output be same as that of the op
  if (input_dist_attrs_[name].process_mesh().empty()) {
    input_dist_attrs_[name].set_process_mesh(process_mesh_);
  }
}

void OperatorDistAttr::set_output_dist_attr(const std::string& name,
                                            const TensorDistAttr& dist_attr) {
  PADDLE_ENFORCE_EQ(
      dist_attr.verify(),
      true,
      platform::errors::InvalidArgument(
          "Wrong dist_attr %s for %s.", dist_attr.to_string(), name));
  if (output_dist_attrs_.count(name) == 1) {
    output_dist_attrs_[name] = dist_attr;
  }
  // Make sure the process mesh of output be same as that of the op
  if (output_dist_attrs_[name].process_mesh().empty()) {
    output_dist_attrs_[name].set_process_mesh(process_mesh_);
  }
}

void OperatorDistAttr::set_process_mesh(const ProcessMesh& process_mesh) {
  for (auto& item : input_dist_attrs_) {
    item.second.set_process_mesh(process_mesh);
  }
  for (auto& item : output_dist_attrs_) {
    item.second.set_process_mesh(process_mesh);
  }
  process_mesh_ = process_mesh;
}

bool OperatorDistAttr::verify() const {
  for (auto const& item : input_dist_attrs_) {
    auto input_names = op_->InputArgumentNames();
    auto found =
        std::find(std::begin(input_names), std::end(input_names), item.first);
    if (found == std::end(input_names)) {
      return false;
    }
    if (!item.second.verify()) {
      return false;
    }
  }
  for (auto const& item : output_dist_attrs_) {
    auto output_names = op_->OutputArgumentNames();
    auto found =
        std::find(std::begin(output_names), std::end(output_names), item.first);
    if (found == std::end(output_names)) {
      return false;
    }
    if (!item.second.verify()) {
      return false;
    }
  }
  return true;
}

std::string OperatorDistAttr::to_string() const {
  std::cout << "here 0" << std::endl;
  std::string str = "{op_type: " + op_->Type() + ", ";
  std::cout << "here 2" << std::endl;
  str += "impl_type: " + impl_type_ + ", ";
  str += "impl_idx: " + std::to_string(impl_idx_) + ", ";
  str += "\nprocess_mesh: " + process_mesh_.to_string() + ", ";
  std::cout << "here 3" << std::endl;
  str += "\ninput_dist_attrs: [\n";
  for (auto const& item : input_dist_attrs_) {
    str += "  " + item.second.to_string() + ",\n";
  }
  std::cout << "here 4" << std::endl;
  str.replace(str.size() - 2, 2, "]");
  str += "\noutput_dist_attrs: [\n";
  for (auto const& item : output_dist_attrs_) {
    str += "  " + item.second.to_string() + ",\n";
  }
  str.replace(str.size() - 2, 2, "]}");
  return str;
}

bool operator==(const OperatorDistAttr& lhs, const OperatorDistAttr& rhs) {
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

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
