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

std::vector<std::string> TensorDistAttr::fields_{
    "process_mesh", "dims_mapping", "batch_dim", "dynamic_dims"};

TensorDistAttr::TensorDistAttr(const VarDesc& tensor) : tensor_(&tensor) {
  VLOG(4) << "[TensorDistAttr constructor] tensor name: " << tensor_->Name();
  if (tensor_->GetType() == framework::proto::VarType::READER) return;
  if (tensor_->GetType() == framework::proto::VarType::LOD_TENSOR_ARRAY) return;
  if (tensor_->GetType() == framework::proto::VarType::STEP_SCOPES) return;
  tensor_shape_ = tensor_->GetShape();
  VLOG(4) << "[TensorDistAttr constructor] tensor shape: "
          << str_join(tensor_shape_);
  set_default_dims_mapping();
  for (std::size_t i = 0; i < tensor_shape_.size(); ++i) {
    dynamic_dims_.push_back(false);
  }
}

TensorDistAttr::TensorDistAttr(const TensorDistAttr& dist_attr) {
  if (tensor_ == nullptr) {
    tensor_ = dist_attr.tensor_;
    tensor_shape_ = dist_attr.tensor_shape_;
  }
  if (tensor_ != nullptr) {
    VLOG(4) << "[TensorDistAttr copy constructor] tensor name:  "
            << tensor_->Name() << ", tensro shape: " << str_join(tensor_shape_);
  } else {
    VLOG(4) << "[TensorDistAttr copy constructor] tensor name:  None"
            << ", tensro shape: " << str_join(tensor_shape_);
  }
  copy_from(dist_attr);
}

TensorDistAttr& TensorDistAttr::operator=(const TensorDistAttr& dist_attr) {
  if (tensor_ == nullptr) {
    tensor_ = dist_attr.tensor_;
    tensor_shape_ = dist_attr.tensor_shape_;
  }
  if (tensor_ != nullptr) {
    VLOG(4) << "[TensorDistAttr assign constructor] tensor name:  "
            << tensor_->Name() << ", tensro shape: " << str_join(tensor_shape_);
  } else {
    VLOG(4) << "[TensorDistAttr assign constructor] tensor name:  None"
            << ", tensro shape: " << str_join(tensor_shape_);
  }
  copy_from(dist_attr);
  return *this;
}

void TensorDistAttr::copy_from(const TensorDistAttr& dist_attr) {
  set_process_mesh(dist_attr.process_mesh());
  set_dims_mapping(dist_attr.dims_mapping());
  set_batch_dim(dist_attr.batch_dim());
  set_dynamic_dims(dist_attr.dynamic_dims());
  set_annotated(dist_attr.annotated());
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
  PADDLE_ENFORCE_EQ(
      verify_batch_dim(batch_dim),
      true,
      platform::errors::InvalidArgument(
          "Wrong batch_dim %d in this distributed attribute.", batch_dim));
  if (tensor_ != nullptr && tensor_shape_.size() > 0) {
    int64_t canonical_batch_dim =
        canonical_dim(batch_dim, tensor_shape_.size());
    batch_dim_ = canonical_batch_dim;
  } else {
    batch_dim_ = batch_dim;
  }
}

void TensorDistAttr::set_dynamic_dims(const std::vector<bool>& dynamic_dims) {
  PADDLE_ENFORCE_EQ(
      verify_dynamic_dims(dynamic_dims),
      true,
      platform::errors::InvalidArgument("The dynamic_dims [%s] is wrong.",
                                        str_join(dynamic_dims)));
  dynamic_dims_ = dynamic_dims;
}

void TensorDistAttr::set_annotated(
    const std::map<std::string, bool>& annotated) {
  PADDLE_ENFORCE_EQ(verify_annotated(annotated),
                    true,
                    platform::errors::InvalidArgument(
                        "The annotated [%s] is wrong.", str_join(annotated)));
  annotated_ = annotated;
}

void TensorDistAttr::set_default_dims_mapping() {
  if (tensor_ != nullptr) {
    dims_mapping_ = std::vector<int64_t>(tensor_shape_.size(), -1);
  }
}

void TensorDistAttr::annotate(const std::string& name) {
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
      if (dim_mapping < -1 || dim_mapping >= process_mesh_.ndim()) {
        return false;
      }
    }
  }
  return true;
}

bool TensorDistAttr::verify_dims_mapping(
    const std::vector<int64_t>& dims_mapping) const {
  VLOG(4) << "[TensorDistAttr verify_dims_mapping] " << str_join(dims_mapping);
  if (dims_mapping.size() != tensor_shape_.size()) {
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

bool TensorDistAttr::verify_batch_dim(int64_t dim) const {
  VLOG(4) << "[TensorDistAttr verify_batch_dim] " << dim;
  int64_t ndim = tensor_shape_.size();
  if (tensor_ != nullptr && ndim > 0) {
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
    const std::vector<bool>& dynamic_dims) const {
  VLOG(4) << "[TensorDistAttr verify_dynamic_dims] " << str_join(dynamic_dims);
  if (dynamic_dims.size() != tensor_shape_.size()) {
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

bool TensorDistAttr::verify() const {
  if (!verify_process_mesh(process_mesh_)) {
    return false;
  }
  if (!verify_dims_mapping(dims_mapping_)) {
    return false;
  }
  if (!verify_batch_dim(batch_dim_)) {
    return false;
  }
  if (!verify_dynamic_dims(dynamic_dims_)) {
    return false;
  }
  if (!verify_annotated(annotated_)) {
    return false;
  }
  return true;
}

std::string TensorDistAttr::to_string() const {
  std::string dist_str;
  if (tensor_ != nullptr) {
    dist_str = "{tensor_name: " + tensor_->Name() + ", ";
  } else {
    dist_str = "{tensor_name: None, ";
  }
  dist_str += "process_mesh: " + process_mesh_.to_string() + ", ";
  dist_str += "dims_mappings: [" + str_join(dims_mapping_) + "], ";
  dist_str += "batch_dim: " + std::to_string(batch_dim_) + ", ";
  dist_str += "dynamic_dims: [" + str_join(dynamic_dims_) + "], ";
  dist_str += "annotated: [" + str_join(annotated_) + "]}";
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
                    platform::errors::InvalidArgument(
                        "Failed to serialize tensor dist attr to string."));
  return data;
}

void TensorDistAttr::parse_from_string(const std::string& data) {
  TensorDistAttrProto proto;
  PADDLE_ENFORCE_EQ(proto.ParseFromString(data),
                    true,
                    platform::errors::InvalidArgument(
                        "Failed to parse tensor dist attr from string."));
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
  return true;
}

std::vector<std::string> OperatorDistAttr::fields_{
    "process_mesh", "impl_type", "impl_idx"};

OperatorDistAttr::OperatorDistAttr(const OpDesc& op) : op_(&op) {
  VLOG(4) << "[OperatorDistAttr constructor] op type: " << op_->Type();
  initialize();
}

OperatorDistAttr::OperatorDistAttr(const OperatorDistAttr& dist_attr) {
  if (op_ == nullptr) {
    op_ = dist_attr.op();
  }
  if (op_ != nullptr) {
    VLOG(4) << "[OperatorDistAttr copy constructor] op type: " << op_->Type();
  } else {
    VLOG(4) << "[OperatorDistAttr copy constructor] op type: None";
  }
  initialize();
  copy_from(dist_attr);
}

OperatorDistAttr& OperatorDistAttr::operator=(
    const OperatorDistAttr& dist_attr) {
  if (op_ == nullptr) {
    op_ = dist_attr.op();
  }
  if (op_ != nullptr) {
    VLOG(4) << "[OperatorDistAttr assign constructor] op type: " << op_->Type();
  } else {
    VLOG(4) << "[OperatorDistAttr assign constructor] op type: None";
  }
  initialize();
  copy_from(dist_attr);
  return *this;
}

void OperatorDistAttr::initialize() {
  if (op_ == nullptr) return;
  for (std::string name : op_->InputArgumentNames()) {
    VarDesc* input = op_->Block()->FindVarRecursive(name);
    VLOG(4) << "[OperatorDistAttr create input dist attr] " << name;
    inputs_[name] = input;
    if (input == nullptr || op_->Type() == "create_py_reader") {
      input_dist_attrs_[name] = TensorDistAttr();
    } else {
      input_dist_attrs_[name] = TensorDistAttr(*input);
    }
  }
  for (std::string name : op_->OutputArgumentNames()) {
    VarDesc* output = op_->Block()->FindVarRecursive(name);
    VLOG(4) << "[OperatorDistAttr create output dist attr] " << name;
    outputs_[name] = output;
    if (output == nullptr) {
      output_dist_attrs_[name] = TensorDistAttr();
    } else {
      output_dist_attrs_[name] = TensorDistAttr(*output);
    }
  }
  impl_type_ = "default";
  impl_idx_ = 0;
}

void OperatorDistAttr::copy_from(const OperatorDistAttr& dist_attr) {
  set_input_dist_attrs(dist_attr.input_dist_attrs());
  set_output_dist_attrs(dist_attr.output_dist_attrs());
  set_process_mesh(dist_attr.process_mesh());
  set_impl_type(dist_attr.impl_type());
  set_impl_idx(dist_attr.impl_idx());
  set_annotated(dist_attr.annotated());
  impl_type_ = dist_attr.impl_type();
  impl_idx_ = dist_attr.impl_idx();
}

void OperatorDistAttr::set_input_dist_attrs(
    const std::map<std::string, TensorDistAttr>& dist_attrs) {
  if (op_ == nullptr) {
    for (const auto& item : dist_attrs) {
      set_input_dist_attr(item.first, item.second);
    }
  } else {
    for (const auto& item : input_dist_attrs_) {
      if (dist_attrs.count(item.first) == 1) {
        set_input_dist_attr(item.first, dist_attrs.at(item.first));
      }
    }
  }
}

void OperatorDistAttr::set_output_dist_attrs(
    const std::map<std::string, TensorDistAttr>& dist_attrs) {
  if (op_ == nullptr) {
    for (const auto& item : dist_attrs) {
      set_output_dist_attr(item.first, item.second);
    }
  } else {
    for (const auto& item : output_dist_attrs_) {
      if (dist_attrs.count(item.first) == 1) {
        set_output_dist_attr(item.first, dist_attrs.at(item.first));
      }
    }
  }
}

void OperatorDistAttr::set_input_dist_attr(const std::string& name,
                                           const TensorDistAttr& dist_attr) {
  PADDLE_ENFORCE_EQ(
      verify_input_dist_attr(name, dist_attr),
      true,
      platform::errors::InvalidArgument("Wrong dist_attr %s for %s. %s",
                                        dist_attr.to_string(),
                                        name,
                                        to_string()));
  input_dist_attrs_[name] = dist_attr;
  // Make sure the process mesh of input be same as that of the op
  input_dist_attrs_[name].set_process_mesh(process_mesh_);
}

void OperatorDistAttr::set_output_dist_attr(const std::string& name,
                                            const TensorDistAttr& dist_attr) {
  PADDLE_ENFORCE_EQ(
      verify_output_dist_attr(name, dist_attr),
      true,
      platform::errors::InvalidArgument(
          "Wrong dist_attr %s for %s.", dist_attr.to_string(), name));
  output_dist_attrs_[name] = dist_attr;
  // Make sure the process mesh of output be same as that of the op
  output_dist_attrs_[name].set_process_mesh(process_mesh_);
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

void OperatorDistAttr::annotate(const std::string& name) {
  auto result = std::find(std::begin(fields_), std::end(fields_), name);
  if (result != std::end(fields_)) {
    annotated_[name] = true;
  }
  if (name == "process_mesh") {
    for (auto& item : input_dist_attrs_) {
      item.second.annotate(name);
    }
    for (auto& item : output_dist_attrs_) {
      item.second.annotate(name);
    }
  }
}

void OperatorDistAttr::set_annotated(
    const std::map<std::string, bool>& annotated) {
  PADDLE_ENFORCE_EQ(verify_annotated(annotated),
                    true,
                    platform::errors::InvalidArgument(
                        "The annotated [%s] is wrong.", str_join(annotated)));
  annotated_ = annotated;
}

const std::vector<int64_t>& OperatorDistAttr::input_dims_mapping(
    const std::string& name) const {
  return input_dist_attr(name).dims_mapping();
}

void OperatorDistAttr::set_input_dims_mapping(
    const std::string& name, const std::vector<int64_t>& dims_mapping) {
  input_dist_attr(name).set_dims_mapping(dims_mapping);
}

const std::vector<int64_t>& OperatorDistAttr::output_dims_mapping(
    const std::string& name) {
  return output_dist_attr(name).dims_mapping();
}

void OperatorDistAttr::set_output_dims_mapping(
    const std::string& name, const std::vector<int64_t>& dims_mapping) {
  output_dist_attr(name).set_dims_mapping(dims_mapping);
}

bool OperatorDistAttr::verify_input_dist_attr(
    const std::string& name, const TensorDistAttr& dist_attr) const {
  VLOG(4) << "[OperatorDistAttr verify_input_dist_attr] " << name << " "
          << dist_attr.to_string();
  if (!dist_attr.verify()) {
    return false;
  }
  if (op_ != nullptr) {
    if (dist_attr.tensor() != nullptr) {
      if (name != dist_attr.tensor()->Name()) {
        return false;
      }
    }
    if (input_dist_attrs_.count(name) == 0) {
      return false;
    }
  }
  return true;
}

bool OperatorDistAttr::verify_output_dist_attr(
    const std::string& name, const TensorDistAttr& dist_attr) const {
  VLOG(4) << "[OperatorDistAttr verify_output_dist_attr] " << name << " "
          << dist_attr.to_string();
  if (!dist_attr.verify()) {
    return false;
  }
  if (op_ != nullptr) {
    if (dist_attr.tensor() != nullptr) {
      if (name != dist_attr.tensor()->Name()) {
        return false;
      }
    }
    if (output_dist_attrs_.count(name) == 0) {
      return false;
    }
  }
  return true;
}

bool OperatorDistAttr::verify_process_mesh(
    const ProcessMesh& process_mesh) const {
  VLOG(4) << "[OperatorDistAttr verify_process_mesh] "
          << process_mesh.to_string();
  if (process_mesh != process_mesh_) {
    return false;
  }
  for (auto& item : input_dist_attrs_) {
    if (item.second.process_mesh() != process_mesh) {
      return false;
    }
  }
  for (auto& item : output_dist_attrs_) {
    if (item.second.process_mesh() != process_mesh) {
      return false;
    }
  }
  return true;
}

bool OperatorDistAttr::verify_annotated(
    const std::map<std::string, bool>& annotated) const {
  VLOG(4) << "[OperatorDistAttr verify_annotated] " << str_join(annotated);
  for (const auto& item : annotated) {
    auto result = std::find(std::begin(fields_), std::end(fields_), item.first);
    if (result == std::end(fields_)) {
      return false;
    }
  }
  for (auto& item : input_dist_attrs_) {
    VLOG(4) << "[OperatorDistAttr verify_annotated input] "
            << str_join(item.second.annotated());
    if (!item.second.verify_annotated(item.second.annotated())) {
      return false;
    }
  }
  for (auto& item : output_dist_attrs_) {
    VLOG(4) << "[OperatorDistAttr verify_annotated output] "
            << str_join(item.second.annotated());
    if (!item.second.verify_annotated(item.second.annotated())) {
      return false;
    }
  }
  return true;
}

bool OperatorDistAttr::verify() const {
  if (op_ == nullptr) {
    return false;
  }
  if (!verify_process_mesh(process_mesh_)) {
    return false;
  }
  for (auto const& item : input_dist_attrs_) {
    auto input_names = op_->InputArgumentNames();
    auto found =
        std::find(std::begin(input_names), std::end(input_names), item.first);
    if (found == std::end(input_names)) {
      return false;
    }
    if (!verify_input_dist_attr(item.first, item.second)) {
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
    if (!verify_output_dist_attr(item.first, item.second)) {
      return false;
    }
  }
  return true;
}

void OperatorDistAttr::rename_input(const std::string& old_name,
                                    const std::string& new_name) {
  for (auto& item : input_dist_attrs_) {
    if (item.first == old_name) {
      VarDesc* new_input = op_->Block()->FindVarRecursive(new_name);
      inputs_[new_name] = new_input;
      if (new_input == nullptr) {
        input_dist_attrs_[new_name] = TensorDistAttr();
      } else {
        input_dist_attrs_[new_name] = TensorDistAttr(*new_input);
        input_dist_attrs_[new_name].copy_from(input_dist_attrs_[old_name]);
      }
      inputs_.erase(old_name);
      input_dist_attrs_.erase(old_name);
      break;
    }
  }
}

void OperatorDistAttr::rename_output(const std::string& old_name,
                                     const std::string& new_name) {
  for (auto& item : output_dist_attrs_) {
    if (item.first == old_name) {
      VarDesc* new_output = op_->Block()->FindVarRecursive(new_name);
      outputs_[new_name] = new_output;
      if (new_output == nullptr) {
        output_dist_attrs_[new_name] = TensorDistAttr();
      } else {
        output_dist_attrs_[new_name] = TensorDistAttr(*new_output);
        output_dist_attrs_[new_name].copy_from(output_dist_attrs_[old_name]);
      }
      outputs_.erase(old_name);
      output_dist_attrs_.erase(old_name);
      break;
    }
  }
}

std::string OperatorDistAttr::to_string() const {
  std::string str;
  if (op_ != nullptr) {
    str += "{op_type: " + op_->Type() + ", ";
  } else {
    str += "{op_type: None, ";
  }
  str += "impl_type: " + impl_type_ + ", ";
  str += "impl_idx: " + std::to_string(impl_idx_) + ", ";
  str += "annotated: [" + str_join(annotated_) + "], ";
  str += "\nprocess_mesh: " + process_mesh_.to_string() + ", ";
  str += "\ninput_dist_attrs: [\n";
  for (auto const& item : input_dist_attrs_) {
    str += "  " + item.second.to_string() + ",\n";
  }
  str.replace(str.size() - 2, 2, "]");
  str += "\noutput_dist_attrs: [\n";
  for (auto const& item : output_dist_attrs_) {
    str += "  " + item.second.to_string() + ",\n";
  }
  str.replace(str.size() - 2, 2, "]}");
  return str;
}

void OperatorDistAttr::from_proto(const OperatorDistAttrProto& proto) {
  for (int64_t i = 0; i < proto.input_dist_attrs_size(); ++i) {
    TensorDistAttr dist_attr;
    std::string name = proto.input_dist_attrs(i).name();
    dist_attr.from_proto(proto.input_dist_attrs(i).tensor_dist_attr());
    input_dist_attrs_[name] = dist_attr;
  }
  for (int64_t i = 0; i < proto.output_dist_attrs_size(); ++i) {
    TensorDistAttr dist_attr;
    std::string name = proto.output_dist_attrs(i).name();
    dist_attr.from_proto(proto.output_dist_attrs(i).tensor_dist_attr());
    output_dist_attrs_[name] = dist_attr;
  }
  process_mesh_ = ProcessMesh::from_proto(proto.process_mesh());
  impl_type_ = proto.impl_type();
  impl_idx_ = proto.impl_idx();
}

OperatorDistAttrProto OperatorDistAttr::to_proto() const {
  OperatorDistAttrProto proto;
  for (const auto& item : input_dist_attrs_) {
    auto proto_item = proto.mutable_input_dist_attrs()->Add();
    proto_item->set_name(item.first);
    proto_item->mutable_tensor_dist_attr()->CopyFrom(item.second.to_proto());
  }
  for (const auto& item : output_dist_attrs_) {
    auto proto_item = proto.mutable_output_dist_attrs()->Add();
    proto_item->set_name(item.first);
    proto_item->mutable_tensor_dist_attr()->CopyFrom(item.second.to_proto());
  }
  proto.mutable_process_mesh()->CopyFrom(process_mesh_.to_proto());
  proto.set_impl_type(impl_type_);
  proto.set_impl_idx(impl_idx_);
  return proto;
}

std::string OperatorDistAttr::serialize_to_string() {
  std::string data;
  auto proto = to_proto();
  proto.SerializeToString(&data);
  PADDLE_ENFORCE_EQ(to_proto().SerializeToString(&data),
                    true,
                    platform::errors::InvalidArgument(
                        "Failed to serialize op dist attr to string."));
  return data;
}

void OperatorDistAttr::parse_from_string(const std::string& data) {
  OperatorDistAttrProto proto;
  PADDLE_ENFORCE_EQ(proto.ParseFromString(data),
                    true,
                    platform::errors::InvalidArgument(
                        "Failed to parse op dist attr from string."));
  from_proto(proto);
}

bool operator==(const OperatorDistAttr& lhs, const OperatorDistAttr& rhs) {
  if (lhs.process_mesh() != rhs.process_mesh()) {
    return false;
  }
  if (lhs.impl_type() != rhs.impl_type()) {
    return false;
  }
  if (lhs.impl_idx() != rhs.impl_idx()) {
    return false;
  }
  for (auto const& item : lhs.input_dist_attrs()) {
    if (rhs.input_dist_attrs().count(item.first) != 1) {
      return false;
    }
    if (rhs.input_dist_attrs().at(item.first) !=
        lhs.input_dist_attrs().at(item.first)) {
      return false;
    }
  }
  for (auto const& item : lhs.output_dist_attrs()) {
    if (rhs.output_dist_attrs().count(item.first) != 1) {
      return false;
    }
    if (rhs.output_dist_attrs().at(item.first) !=
        lhs.output_dist_attrs().at(item.first)) {
      return false;
    }
  }
  return true;
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
