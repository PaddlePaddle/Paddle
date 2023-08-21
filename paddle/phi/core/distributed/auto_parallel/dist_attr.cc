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

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"

#include <algorithm>
#include <iostream>
#include <iterator>

#include "glog/logging.h"

namespace phi {
namespace distributed {
using phi::distributed::auto_parallel::str_join;
using phi::distributed::auto_parallel::TensorDistAttrProto;

// partial is not allow annotated by user by now.
std::vector<std::string> TensorDistAttr::fields_{
    "process_mesh", "dims_mapping", "batch_dim", "dynamic_dims"};

TensorDistAttr::TensorDistAttr(const std::vector<int64_t>& tensor_shape) {
  set_default_dims_mapping(tensor_shape);
  set_default_dynamic_dims(tensor_shape);
}

TensorDistAttr::TensorDistAttr(const TensorDistAttr& dist_attr) {
  copy_from(dist_attr);
}

TensorDistAttr& TensorDistAttr::operator=(const TensorDistAttr& dist_attr) {
  if (this == &dist_attr) return *this;
  TensorDistAttr tmp(dist_attr);
  std::swap(this->process_mesh_, tmp.process_mesh_);
  std::swap(this->dims_mapping_, tmp.dims_mapping_);
  std::swap(this->batch_dim_, tmp.batch_dim_);
  std::swap(this->dynamic_dims_, tmp.dynamic_dims_);
  std::swap(this->annotated_, tmp.annotated_);
  std::swap(this->partial_status_, tmp.partial_status_);
  return *this;
}

void TensorDistAttr::copy_from(const TensorDistAttr& dist_attr) {
  set_process_mesh(dist_attr.process_mesh());
  set_dims_mapping(dist_attr.dims_mapping());
  set_batch_dim(dist_attr.batch_dim());
  set_dynamic_dims(dist_attr.dynamic_dims());
  set_annotated(dist_attr.annotated());
  set_partial_status(dist_attr.partial_status());
}

void TensorDistAttr::set_process_mesh(const ProcessMesh& process_mesh) {
  process_mesh_ = process_mesh;
}

void TensorDistAttr::set_dims_mapping(
    const std::vector<int64_t>& dims_mapping) {
  dims_mapping_ = dims_mapping;
}

void TensorDistAttr::set_batch_dim(int64_t batch_dim) {
  batch_dim_ = batch_dim;
}

void TensorDistAttr::set_dynamic_dims(const std::vector<bool>& dynamic_dims) {
  dynamic_dims_ = dynamic_dims;
}

void TensorDistAttr::set_annotated(
    const std::map<std::string, bool>& annotated) {
  annotated_ = annotated;
}

const std::set<int64_t> TensorDistAttr::partial_dims() const {
  std::set<int64_t> keys;
  for (auto& kv : partial_status_) {
    keys.emplace(kv.first);
  }
  return keys;
}

void TensorDistAttr::set_partial_status(
    const paddle::flat_hash_map<int64_t, ReduceType>& partial_status) {
  partial_status_ = partial_status;
}

void TensorDistAttr::set_partial_status(const std::vector<int64_t>& dims,
                                        const ReduceType& type) {
  for (const auto& dim : dims) {
    if (partial_status_.count(dim) != 0) {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Trying to Set dim %d as Partial which is already a Partial dim.",
          dim));
    }
    partial_status_.emplace(dim, type);
  }
}

void TensorDistAttr::clean_partial_status() { partial_status_.clear(); }

void TensorDistAttr::clean_partial_dims(const std::vector<int64_t>& dims) {
  for (const auto& dim : dims) {
    if (partial_status_.count(dim) == 0) {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Trying to clean Partial on dim %d but it is not Partial.", dim));
    } else {
      partial_status_.erase(dim);
    }
  }
}

void TensorDistAttr::set_default_dims_mapping(
    const std::vector<int64_t>& tensor_shape) {
  if (!tensor_shape.empty()) {
    dims_mapping_ = std::vector<int64_t>(tensor_shape.size(), -1);
  }
}

void TensorDistAttr::set_default_dynamic_dims(
    const std::vector<int64_t>& tensor_shape) {
  if (!tensor_shape.empty()) {
    dynamic_dims_ = std::vector<bool>(tensor_shape.size(), false);
  }
}

void TensorDistAttr::mark_annotated(const std::string& name) {
  auto result = std::find(std::begin(fields_), std::end(fields_), name);
  if (result != std::end(fields_)) {
    annotated_[name] = true;
  }
}

bool TensorDistAttr::verify_process_mesh(
    const ProcessMesh& process_mesh) const {
  VLOG(4) << "[TensorDistAttr verify_process_mesh] "
          << process_mesh.to_string();
  if (!process_mesh_.empty()) {
    for (int64_t dim_mapping : dims_mapping_) {
      if (dim_mapping >= process_mesh_.ndim()) {
        return false;
      }
    }
  }
  return true;
}

bool TensorDistAttr::verify_dims_mapping(
    const std::vector<int64_t>& dims_mapping,
    const std::vector<int64_t>& tensor_shape) const {
  VLOG(4) << "[TensorDistAttr verify_dims_mapping] " << str_join(dims_mapping);
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

bool TensorDistAttr::verify_batch_dim(
    int64_t dim, const std::vector<int64_t>& tensor_shape) const {
  VLOG(4) << "[TensorDistAttr verify_batch_dim] " << dim;
  int64_t ndim = tensor_shape.size();
  if (ndim > 0) {
    if (dim < 0) {
      dim = dim + ndim;
    }
    if (dim < 0 || dim >= ndim) {
      return false;
    }
  }
  return true;
}

bool TensorDistAttr::verify_dynamic_dims(
    const std::vector<bool>& dynamic_dims,
    const std::vector<int64_t>& tensor_shape) const {
  VLOG(4) << "[TensorDistAttr verify_dynamic_dims] " << str_join(dynamic_dims);
  if (!dynamic_dims.empty() && dynamic_dims.size() != tensor_shape.size()) {
    return false;
  }
  return true;
}

bool TensorDistAttr::verify_annotated(
    const std::map<std::string, bool>& annotated) const {
  VLOG(4) << "[TensorDistAttr verify_annotated] " << str_join(annotated);
  for (const auto& item : annotated) {
    auto result = std::find(std::begin(fields_), std::end(fields_), item.first);
    if (result == std::end(fields_)) {
      return false;
    }
  }
  return true;
}

bool TensorDistAttr::verify_partial_status() const {
  VLOG(4) << "[TensorDistAttr verify_partial_status] "
          << partial_status_string();
  for (auto& itr : partial_status_) {
    if (itr.first < 0 || itr.first >= process_mesh_.ndim()) {
      return false;
    }
    if (itr.second < ReduceType::kRedSum || itr.second <= ReduceType::kRedAll) {
      return false;
    }
  }
  return true;
}

bool TensorDistAttr::verify(const std::vector<int64_t>& tensor_shape) const {
  if (!verify_process_mesh(process_mesh_)) {
    return false;
  }
  if (!verify_dims_mapping(dims_mapping_, tensor_shape)) {
    return false;
  }
  if (!verify_batch_dim(batch_dim_, tensor_shape)) {
    return false;
  }
  if (!verify_dynamic_dims(dynamic_dims_, tensor_shape)) {
    return false;
  }
  if (!verify_annotated(annotated_)) {
    return false;
  }
  if (!verify_partial_status()) {
    return false;
  }
  return true;
}

std::string TensorDistAttr::to_string() const {
  std::string dist_str;
  dist_str += "{process_mesh: " + process_mesh_.to_string() + ", ";
  dist_str += "dims_mappings: [" + str_join(dims_mapping_) + "], ";
  dist_str += "batch_dim: " + std::to_string(batch_dim_) + ", ";
  dist_str += "dynamic_dims: [" + str_join(dynamic_dims_) + "], ";
  dist_str += "annotated: [" + str_join(annotated_) + "], ";
  dist_str += "partial: " + partial_status_string() + ".}";
  return dist_str;
}

void TensorDistAttr::from_proto(const TensorDistAttrProto& proto) {
  process_mesh_ = ProcessMesh::from_proto(proto.process_mesh());
  dims_mapping_.resize(proto.dims_mapping_size());
  for (int64_t i = 0; i < proto.dims_mapping_size(); ++i) {
    dims_mapping_[i] = proto.dims_mapping(i);
  }
  batch_dim_ = proto.batch_dim();
  dynamic_dims_.resize(proto.dynamic_dims_size());
  for (int64_t i = 0; i < proto.dynamic_dims_size(); ++i) {
    dynamic_dims_[i] = proto.dynamic_dims(i);
  }
}

TensorDistAttrProto TensorDistAttr::to_proto() const {
  TensorDistAttrProto proto;
  proto.mutable_process_mesh()->CopyFrom(process_mesh_.to_proto());
  for (const auto& i : dims_mapping_) {
    proto.add_dims_mapping(i);
  }
  proto.set_batch_dim(batch_dim_);
  for (const auto& i : dynamic_dims_) {
    proto.add_dynamic_dims(i);
  }
  return proto;
}

std::string TensorDistAttr::serialize_to_string() {
  std::string data;
  auto proto = to_proto();
  proto.SerializeToString(&data);
  PADDLE_ENFORCE_EQ(to_proto().SerializeToString(&data),
                    true,
                    errors::InvalidArgument(
                        "Failed to serialize tensor dist attr to string."));
  return data;
}

void TensorDistAttr::parse_from_string(const std::string& data) {
  TensorDistAttrProto proto;
  PADDLE_ENFORCE_EQ(
      proto.ParseFromString(data),
      true,
      errors::InvalidArgument(
          "Failed to parse tensor dist attr from string: %s.", data));
  from_proto(proto);
}

bool operator==(const TensorDistAttr& lhs, const TensorDistAttr& rhs) {
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
  if (lhs.partial_status() != rhs.partial_status()) {
    return false;
  }
  return true;
}

std::string TensorDistAttr::partial_status_string() const {
  std::string partial_status_str = "[";
  for (auto& itr : partial_status_) {
    partial_status_str += "Partial(dims:" + std::to_string(itr.first) + ", " +
                          ReduceTypeStrings[static_cast<int>(itr.second)] +
                          "), ";
  }
  partial_status_str += "]";
  return partial_status_str;
}

}  // namespace distributed
}  // namespace phi
