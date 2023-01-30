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

<<<<<<< HEAD
static inline std::vector<int64_t> get_tensor_shape(const VarDesc* tensor) {
  if (tensor == nullptr) return std::vector<int64_t>();
  switch (tensor->GetType()) {
    case framework::proto::VarType::READER:
    case framework::proto::VarType::LOD_TENSOR_ARRAY:
    case framework::proto::VarType::STEP_SCOPES:
    case framework::proto::VarType::FEED_MINIBATCH:
    case framework::proto::VarType::FETCH_LIST:
      return std::vector<int64_t>();
    default:
      return tensor->GetShape();
  }
}

TensorDistAttr::TensorDistAttr(const VarDesc& tensor) {
  VLOG(4) << "[TensorDistAttr constructor] tensor name: " << tensor.Name();
  std::vector<int64_t> tensor_shape = get_tensor_shape(&tensor);
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
  return *this;
}

void TensorDistAttr::copy_from(const TensorDistAttr& dist_attr) {
=======
TensorDistAttr::TensorDistAttr(const VarDesc& tensor)
    : tensor_(&tensor), batch_dim_(0) {
  set_default_dims_mapping();
  std::vector<int64_t> tensor_shape = tensor_->GetShape();
  for (std::size_t i = 0; i < tensor_shape.size(); ++i) {
    dynamic_dims_.push_back(false);
  }
}

TensorDistAttr::TensorDistAttr(const TensorDistAttr& dist_attr) {
  if (tensor_ == nullptr) {
    tensor_ = dist_attr.tensor();
  }
  set_process_mesh(dist_attr.process_mesh());
  set_dims_mapping(dist_attr.dims_mapping());
  set_batch_dim(dist_attr.batch_dim());
  set_dynamic_dims(dist_attr.dynamic_dims());
  set_annotated(dist_attr.annotated());
}

TensorDistAttr& TensorDistAttr::operator=(const TensorDistAttr& dist_attr) {
  if (tensor_ == nullptr) {
    tensor_ = dist_attr.tensor();
  }
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  set_process_mesh(dist_attr.process_mesh());
  set_dims_mapping(dist_attr.dims_mapping());
  set_batch_dim(dist_attr.batch_dim());
  set_dynamic_dims(dist_attr.dynamic_dims());
  set_annotated(dist_attr.annotated());
<<<<<<< HEAD
}

void TensorDistAttr::set_process_mesh(const ProcessMesh& process_mesh) {
=======
  return *this;
}

void TensorDistAttr::set_process_mesh(const ProcessMesh& process_mesh) {
  PADDLE_ENFORCE_EQ(verify_process_mesh(process_mesh),
                    true,
                    platform::errors::InvalidArgument(
                        "Wrong process mesh %s.", process_mesh.to_string()));
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  process_mesh_ = process_mesh;
}

void TensorDistAttr::set_dims_mapping(
    const std::vector<int64_t>& dims_mapping) {
<<<<<<< HEAD
=======
  PADDLE_ENFORCE_EQ(verify_dims_mapping(dims_mapping),
                    true,
                    platform::errors::InvalidArgument("Wrong dims_mapping %s.",
                                                      str_join(dims_mapping)));
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  dims_mapping_ = dims_mapping;
}

void TensorDistAttr::set_batch_dim(int64_t batch_dim) {
<<<<<<< HEAD
  batch_dim_ = batch_dim;
}

void TensorDistAttr::set_dynamic_dims(const std::vector<bool>& dynamic_dims) {
=======
  PADDLE_ENFORCE_EQ(
      verify_batch_dim(batch_dim),
      true,
      platform::errors::InvalidArgument(
          "Wrong batch_dim %d in this distributed attribute.", batch_dim));
  if (tensor_ != nullptr) {
    std::vector<int64_t> tensor_shape = tensor_->GetShape();
    int64_t canonical_batch_dim = canonical_dim(batch_dim, tensor_shape.size());
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  dynamic_dims_ = dynamic_dims;
}

void TensorDistAttr::set_annotated(
    const std::map<std::string, bool>& annotated) {
<<<<<<< HEAD
  annotated_ = annotated;
}

void TensorDistAttr::set_default_dims_mapping(
    const std::vector<int64_t>& tensor_shape) {
  if (tensor_shape.size() != 0) {
=======
  PADDLE_ENFORCE_EQ(verify_annotated(annotated),
                    true,
                    platform::errors::InvalidArgument(
                        "The annotated [%s] is wrong.", str_join(annotated)));
  annotated_ = annotated;
}

void TensorDistAttr::set_default_dims_mapping() {
  if (tensor_ != nullptr) {
    std::vector<int64_t> tensor_shape = tensor_->GetShape();
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    dims_mapping_ = std::vector<int64_t>(tensor_shape.size(), -1);
  }
}

<<<<<<< HEAD
void TensorDistAttr::set_default_dynamic_dims(
    const std::vector<int64_t>& tensor_shape) {
  if (tensor_shape.size() != 0) {
    dynamic_dims_ = std::vector<bool>(tensor_shape.size(), false);
  }
}

void TensorDistAttr::mark_annotated(const std::string& name) {
=======
void TensorDistAttr::annotate(const std::string& name) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  auto result = std::find(std::begin(fields_), std::end(fields_), name);
  if (result != std::end(fields_)) {
    annotated_[name] = true;
  }
}

bool TensorDistAttr::verify_process_mesh(
    const ProcessMesh& process_mesh) const {
<<<<<<< HEAD
  VLOG(4) << "[TensorDistAttr verify_process_mesh] "
          << process_mesh.to_string();
  if (!process_mesh_.empty()) {
    for (int64_t dim_mapping : dims_mapping_) {
      if (dim_mapping >= process_mesh_.ndim()) {
=======
  if (!process_mesh_.empty()) {
    for (int64_t dim_mapping : dims_mapping_) {
      if (dim_mapping < -1 || dim_mapping >= process_mesh_.ndim()) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return false;
      }
    }
  }
  return true;
}

bool TensorDistAttr::verify_dims_mapping(
<<<<<<< HEAD
    const std::vector<int64_t>& dims_mapping,
    const std::vector<int64_t>& tensor_shape) const {
  VLOG(4) << "[TensorDistAttr verify_dims_mapping] " << str_join(dims_mapping);
  if (dims_mapping.size() != tensor_shape.size()) {
    return false;
=======
    const std::vector<int64_t>& dims_mapping) const {
  if (tensor_ != nullptr) {
    std::vector<int64_t> tensor_shape = tensor_->GetShape();
    if (dims_mapping.size() != tensor_shape.size()) {
      return false;
    }
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

<<<<<<< HEAD
bool TensorDistAttr::verify_batch_dim(
    int64_t dim, const std::vector<int64_t>& tensor_shape) const {
  VLOG(4) << "[TensorDistAttr verify_batch_dim] " << dim;
  int64_t ndim = tensor_shape.size();
  if (ndim > 0) {
=======
bool TensorDistAttr::verify_batch_dim(int64_t dim) const {
  if (tensor_ != nullptr) {
    std::vector<int64_t> tensor_shape = tensor_->GetShape();
    int64_t ndim = tensor_shape.size();
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
    const std::vector<bool>& dynamic_dims,
    const std::vector<int64_t>& tensor_shape) const {
  VLOG(4) << "[TensorDistAttr verify_dynamic_dims] " << str_join(dynamic_dims);
  if (dynamic_dims.size() > 0 && dynamic_dims.size() != tensor_shape.size()) {
    return false;
=======
    const std::vector<bool>& dynamic_dims) const {
  if (tensor_ != nullptr) {
    std::vector<int64_t> tensor_shape = tensor_->GetShape();
    if (dynamic_dims.size() != tensor_shape.size()) {
      return false;
    }
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  }
  return true;
}

bool TensorDistAttr::verify_annotated(
    const std::map<std::string, bool>& annotated) const {
<<<<<<< HEAD
  VLOG(4) << "[TensorDistAttr verify_annotated] " << str_join(annotated);
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  for (const auto& item : annotated) {
    auto result = std::find(std::begin(fields_), std::end(fields_), item.first);
    if (result == std::end(fields_)) {
      return false;
    }
  }
  return true;
}

<<<<<<< HEAD
bool TensorDistAttr::verify(const VarDesc* tensor) const {
  auto tensor_shape = get_tensor_shape(tensor);
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
=======
bool TensorDistAttr::verify() const {
  if (tensor_ == nullptr) {
    return false;
  }
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    return false;
  }
  if (!verify_annotated(annotated_)) {
    return false;
  }
  return true;
}

std::string TensorDistAttr::to_string() const {
  std::string dist_str;
<<<<<<< HEAD
  dist_str += "{process_mesh: " + process_mesh_.to_string() + ", ";
=======
  if (tensor_ != nullptr) {
    dist_str = "{tensor_name: " + tensor_->Name() + ", ";
  } else {
    dist_str = "{tensor_name: None, ";
  }
  dist_str += "process_mesh: " + process_mesh_.to_string() + ", ";
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  dist_str += "dims_mappings: [" + str_join(dims_mapping_) + "], ";
  dist_str += "batch_dim: " + std::to_string(batch_dim_) + ", ";
  dist_str += "dynamic_dims: [" + str_join(dynamic_dims_) + "], ";
  dist_str += "annotated: [" + str_join(annotated_) + "]}";
  return dist_str;
}

<<<<<<< HEAD
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
=======
TensorDistAttr TensorDistAttr::from_proto(const TensorDistAttrProto& proto) {
  TensorDistAttr dist_attr;
  dist_attr.process_mesh_ = ProcessMesh::from_proto(proto.process_mesh());
  dist_attr.dims_mapping_.resize(proto.dims_mapping_size());
  for (int64_t i = 0; i < proto.dims_mapping_size(); ++i) {
    dist_attr.dims_mapping_[i] = proto.dims_mapping(i);
  }
  dist_attr.batch_dim_ = proto.batch_dim();
  dist_attr.dynamic_dims_.resize(proto.dynamic_dims_size());
  for (int64_t i = 0; i < proto.dynamic_dims_size(); ++i) {
    dist_attr.dynamic_dims_[i] = proto.dynamic_dims(i);
  }
  return dist_attr;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

<<<<<<< HEAD
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

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

<<<<<<< HEAD
std::vector<std::string> OperatorDistAttr::fields_{"process_mesh",
                                                   "impl_type",
                                                   "impl_idx",
                                                   "is_recompute",
                                                   "execution_stream",
                                                   "stream_priority",
                                                   "scheduling_priority"};

OperatorDistAttr::OperatorDistAttr(const OpDesc& op) {
  VLOG(4) << "[OperatorDistAttr constructor] op type: " << op.Type();
  initialize(&op);
}

OperatorDistAttr::OperatorDistAttr(const OperatorDistAttr& dist_attr) {
  VLOG(4) << "[OperatorDistAttr copy constructor]";
  copy_from(dist_attr);
}

OperatorDistAttr& OperatorDistAttr::operator=(
    const OperatorDistAttr& dist_attr) {
  VLOG(4) << "[OperatorDistAttr assign constructor]";
  if (this == &dist_attr) return *this;
  OperatorDistAttr tmp(dist_attr);
  std::swap(this->input_dist_attrs_, tmp.input_dist_attrs_);
  std::swap(this->output_dist_attrs_, tmp.output_dist_attrs_);
  std::swap(this->process_mesh_, tmp.process_mesh_);
  std::swap(this->op_type_, tmp.op_type_);
  std::swap(this->impl_type_, tmp.impl_type_);
  std::swap(this->impl_idx_, tmp.impl_idx_);
  std::swap(this->is_recompute_, tmp.is_recompute_);
  std::swap(this->execution_stream_, tmp.execution_stream_);
  std::swap(this->stream_priority_, tmp.stream_priority_);
  std::swap(this->scheduling_priority_, tmp.scheduling_priority_);
  std::swap(this->annotated_, tmp.annotated_);
  // Note: Make sure all tensor dist attr has the same process_mesh
  set_process_mesh(this->process_mesh_);
  return *this;
}

void OperatorDistAttr::initialize(const OpDesc* op) {
  if (op == nullptr) return;
  for (std::string name : op->InputArgumentNames()) {
    VarDesc* input = op->Block()->FindVarRecursive(name);
    VLOG(4) << "[OperatorDistAttr create input dist attr] " << name;
    if (input == nullptr || op->Type() == "create_py_reader") {
      input_dist_attrs_[name] = TensorDistAttr();
    } else {
      input_dist_attrs_[name] = TensorDistAttr(*input);
    }
  }
  for (std::string name : op->OutputArgumentNames()) {
    VarDesc* output = op->Block()->FindVarRecursive(name);
    VLOG(4) << "[OperatorDistAttr create output dist attr] " << name;
    if (output == nullptr) {
      output_dist_attrs_[name] = TensorDistAttr();
    } else {
      output_dist_attrs_[name] = TensorDistAttr(*output);
    }
  }
  op_type_ = op->Type();
  impl_type_ = kDefault;
  impl_idx_ = 0;
  is_recompute_ = false;
  execution_stream_ = kDefault;
  stream_priority_ = 0;
  scheduling_priority_ = 0;
}

void OperatorDistAttr::copy_from(const OperatorDistAttr& dist_attr) {
  set_input_dist_attrs(dist_attr.input_dist_attrs());
  set_output_dist_attrs(dist_attr.output_dist_attrs());
  set_process_mesh(dist_attr.process_mesh());
  set_op_type(dist_attr.op_type());
  set_impl_type(dist_attr.impl_type());
  set_impl_idx(dist_attr.impl_idx());
  set_is_recompute(dist_attr.is_recompute());
  set_execution_stream(dist_attr.execution_stream());
  set_stream_priority(dist_attr.stream_priority());
  set_scheduling_priority(dist_attr.scheduling_priority());
  set_annotated(dist_attr.annotated());
}

void OperatorDistAttr::set_input_dist_attrs(
    const std::map<std::string, TensorDistAttr>& dist_attrs) {
  for (const auto& item : dist_attrs) {
    set_input_dist_attr(item.first, item.second);
  }
}

void OperatorDistAttr::set_output_dist_attrs(
    const std::map<std::string, TensorDistAttr>& dist_attrs) {
  for (const auto& item : dist_attrs) {
    set_output_dist_attr(item.first, item.second);
  }
=======
std::vector<std::string> OperatorDistAttr::fields_{
    "process_mesh", "impl_type", "impl_idx"};

OperatorDistAttr::OperatorDistAttr(const OpDesc& op) : op_(&op) {
  for (std::string name : op_->InputArgumentNames()) {
    VarDesc* input = op_->Block()->FindVarRecursive(name);
    inputs_[name] = input;
    input_dist_attrs_[name] = TensorDistAttr(*input);
  }
  for (std::string name : op_->OutputArgumentNames()) {
    VarDesc* output = op_->Block()->FindVarRecursive(name);
    outputs_[name] = output;
    output_dist_attrs_[name] = TensorDistAttr(*output);
  }
  impl_type_ = "default";
  impl_idx_ = 0;
}

OperatorDistAttr::OperatorDistAttr(const OperatorDistAttr& dist_attr) {
  if (op_ == nullptr) {
    op_ = dist_attr.op();
  }
  for (const auto& item : dist_attr.input_dist_attrs()) {
    set_input_dist_attr(item.first, item.second);
  }
  for (const auto& item : dist_attr.output_dist_attrs()) {
    set_output_dist_attr(item.first, item.second);
  }
  set_process_mesh(dist_attr.process_mesh());
  set_impl_type(dist_attr.impl_type());
  set_impl_idx(dist_attr.impl_idx());
  set_annotated(dist_attr.annotated());
}

OperatorDistAttr& OperatorDistAttr::operator=(
    const OperatorDistAttr& dist_attr) {
  if (op_ == nullptr) {
    op_ = dist_attr.op();
  }
  for (const auto& item : dist_attr.input_dist_attrs()) {
    set_input_dist_attr(item.first, item.second);
  }
  for (const auto& item : dist_attr.output_dist_attrs()) {
    set_output_dist_attr(item.first, item.second);
  }
  set_process_mesh(dist_attr.process_mesh());
  set_impl_type(dist_attr.impl_type());
  set_impl_idx(dist_attr.impl_idx());
  set_annotated(dist_attr.annotated());
  return *this;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
}

void OperatorDistAttr::set_input_dist_attr(const std::string& name,
                                           const TensorDistAttr& dist_attr) {
<<<<<<< HEAD
=======
  PADDLE_ENFORCE_EQ(
      verify_input_dist_attr(name, dist_attr),
      true,
      platform::errors::InvalidArgument(
          "Wrong dist_attr %s for %s.", dist_attr.to_string(), name));
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  input_dist_attrs_[name] = dist_attr;
  // Make sure the process mesh of input be same as that of the op
  input_dist_attrs_[name].set_process_mesh(process_mesh_);
}

void OperatorDistAttr::set_output_dist_attr(const std::string& name,
                                            const TensorDistAttr& dist_attr) {
<<<<<<< HEAD
=======
  PADDLE_ENFORCE_EQ(
      verify_output_dist_attr(name, dist_attr),
      true,
      platform::errors::InvalidArgument(
          "Wrong dist_attr %s for %s.", dist_attr.to_string(), name));
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

<<<<<<< HEAD
void OperatorDistAttr::set_annotated(
    const std::map<std::string, bool>& annotated) {
  annotated_ = annotated;
}

void OperatorDistAttr::mark_annotated(const std::string& name) {
=======
void OperatorDistAttr::annotate(const std::string& name) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  auto result = std::find(std::begin(fields_), std::end(fields_), name);
  if (result != std::end(fields_)) {
    annotated_[name] = true;
  }
  if (name == "process_mesh") {
    for (auto& item : input_dist_attrs_) {
<<<<<<< HEAD
      item.second.mark_annotated(name);
    }
    for (auto& item : output_dist_attrs_) {
      item.second.mark_annotated(name);
=======
      item.second.annotate(name);
    }
    for (auto& item : output_dist_attrs_) {
      item.second.annotate(name);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    }
  }
}

<<<<<<< HEAD
void OperatorDistAttr::clear_annotated() {
  annotated_.clear();
  for (auto& item : input_dist_attrs_) {
    item.second.clear_annotated();
  }
  for (auto& item : output_dist_attrs_) {
    item.second.clear_annotated();
  }
}

const std::vector<int64_t>& OperatorDistAttr::input_dims_mapping(
    const std::string& name) const {
  return input_dist_attr(name).dims_mapping();
}

void OperatorDistAttr::set_input_dims_mapping(
    const std::string& name, const std::vector<int64_t>& dims_mapping) {
  input_dist_attrs_[name].set_dims_mapping(dims_mapping);
  input_dist_attrs_[name].set_process_mesh(process_mesh_);
}

const std::vector<int64_t>& OperatorDistAttr::output_dims_mapping(
    const std::string& name) {
  return output_dist_attr(name).dims_mapping();
}

void OperatorDistAttr::set_output_dims_mapping(
    const std::string& name, const std::vector<int64_t>& dims_mapping) {
  output_dist_attrs_[name].set_dims_mapping(dims_mapping);
  output_dist_attrs_[name].set_process_mesh(process_mesh_);
}

bool OperatorDistAttr::verify_input_dist_attr(const std::string& name,
                                              const TensorDistAttr& dist_attr,
                                              const VarDesc* tensor) const {
  VLOG(4) << "[OperatorDistAttr verify_input_dist_attr] " << name << " "
          << dist_attr.to_string();
  if (!dist_attr.verify(tensor)) {
    return false;
  }
  if (tensor != nullptr) {
    if (name != tensor->Name()) {
      return false;
    }
  }
  if (input_dist_attrs_.count(name) == 0) {
    return false;
  }
  return true;
}

bool OperatorDistAttr::verify_output_dist_attr(const std::string& name,
                                               const TensorDistAttr& dist_attr,
                                               const VarDesc* tensor) const {
  VLOG(4) << "[OperatorDistAttr verify_output_dist_attr] " << name << " "
          << dist_attr.to_string();
  if (!dist_attr.verify(tensor)) {
    return false;
  }
  if (tensor != nullptr) {
    if (name != tensor->Name()) {
      return false;
    }
  }
  if (output_dist_attrs_.count(name) == 0) {
    return false;
  }
=======
void OperatorDistAttr::set_annotated(
    const std::map<std::string, bool>& annotated) {
  PADDLE_ENFORCE_EQ(verify_annotated(annotated),
                    true,
                    platform::errors::InvalidArgument(
                        "The annotated [%s] is wrong.", str_join(annotated)));
  annotated_ = annotated;
}

bool OperatorDistAttr::verify_input_dist_attr(
    const std::string& name, const TensorDistAttr& dist_attr) const {
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  return true;
}

bool OperatorDistAttr::verify_process_mesh(
    const ProcessMesh& process_mesh) const {
<<<<<<< HEAD
  VLOG(4) << "[OperatorDistAttr verify_process_mesh] "
          << process_mesh.to_string();
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
  VLOG(4) << "[OperatorDistAttr verify_annotated] " << str_join(annotated);
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  for (const auto& item : annotated) {
    auto result = std::find(std::begin(fields_), std::end(fields_), item.first);
    if (result == std::end(fields_)) {
      return false;
    }
  }
  for (auto& item : input_dist_attrs_) {
<<<<<<< HEAD
    VLOG(4) << "[OperatorDistAttr verify_annotated input] "
            << str_join(item.second.annotated());
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    if (!item.second.verify_annotated(item.second.annotated())) {
      return false;
    }
  }
  for (auto& item : output_dist_attrs_) {
<<<<<<< HEAD
    VLOG(4) << "[OperatorDistAttr verify_annotated output] "
            << str_join(item.second.annotated());
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    if (!item.second.verify_annotated(item.second.annotated())) {
      return false;
    }
  }
  return true;
}

<<<<<<< HEAD
bool OperatorDistAttr::verify(const OpDesc* op) const {
=======
bool OperatorDistAttr::verify() const {
  if (op_ == nullptr) {
    return false;
  }
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  if (!verify_process_mesh(process_mesh_)) {
    return false;
  }
  for (auto const& item : input_dist_attrs_) {
<<<<<<< HEAD
    auto input_names = op->InputArgumentNames();
=======
    auto input_names = op_->InputArgumentNames();
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    auto found =
        std::find(std::begin(input_names), std::end(input_names), item.first);
    if (found == std::end(input_names)) {
      return false;
    }
<<<<<<< HEAD
    auto tensor = op->Block()->FindVarRecursive(item.first);
    if (!verify_input_dist_attr(item.first, item.second, tensor)) {
=======
    if (!verify_input_dist_attr(item.first, item.second)) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
      return false;
    }
  }
  for (auto const& item : output_dist_attrs_) {
<<<<<<< HEAD
    auto output_names = op->OutputArgumentNames();
=======
    auto output_names = op_->OutputArgumentNames();
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    auto found =
        std::find(std::begin(output_names), std::end(output_names), item.first);
    if (found == std::end(output_names)) {
      return false;
    }
<<<<<<< HEAD
    auto tensor = op->Block()->FindVarRecursive(item.first);
    if (!verify_output_dist_attr(item.first, item.second, tensor)) {
=======
    if (!verify_output_dist_attr(item.first, item.second)) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
      return false;
    }
  }
  return true;
}

<<<<<<< HEAD
void OperatorDistAttr::rename_input(const std::string& old_name,
                                    const std::string& new_name) {
  if (old_name == new_name) return;
  for (auto& item : input_dist_attrs_) {
    if (item.first == old_name) {
      input_dist_attrs_[new_name].copy_from(input_dist_attrs_[old_name]);
      input_dist_attrs_.erase(old_name);
      break;
    }
  }
}

void OperatorDistAttr::rename_output(const std::string& old_name,
                                     const std::string& new_name) {
  if (old_name == new_name) return;
  for (auto& item : output_dist_attrs_) {
    if (item.first == old_name) {
      output_dist_attrs_[new_name].copy_from(output_dist_attrs_[old_name]);
      output_dist_attrs_.erase(old_name);
      break;
    }
  }
}

std::string OperatorDistAttr::to_string() const {
  std::string str;
  str += "{impl_type: " + impl_type_ + ", ";
  str += "impl_idx: " + std::to_string(impl_idx_) + ", ";
  str += "execution_stream: " + execution_stream_ + ", ";
  str += "stream_priority: " + std::to_string(stream_priority_) + ", ";
  str += "scheduling_priority: " + std::to_string(scheduling_priority_) + ", ";
=======
std::string OperatorDistAttr::to_string() const {
  std::string str;
  if (op_ != nullptr) {
    str += "{op_type: " + op_->Type() + ", ";
  } else {
    str += "{op_type: None, ";
  }
  str += "impl_type: " + impl_type_ + ", ";
  str += "impl_idx: " + std::to_string(impl_idx_) + ", ";
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  str += "annotated: [" + str_join(annotated_) + "], ";
  str += "\nprocess_mesh: " + process_mesh_.to_string() + ", ";
  str += "\ninput_dist_attrs: [\n";
  for (auto const& item : input_dist_attrs_) {
<<<<<<< HEAD
    str += "  " + item.first + ": " + item.second.to_string() + ",\n";
=======
    str += "  " + item.second.to_string() + ",\n";
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  }
  str.replace(str.size() - 2, 2, "]");
  str += "\noutput_dist_attrs: [\n";
  for (auto const& item : output_dist_attrs_) {
<<<<<<< HEAD
    str += "  " + item.first + ": " + item.second.to_string() + ",\n";
=======
    str += "  " + item.second.to_string() + ",\n";
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  }
  str.replace(str.size() - 2, 2, "]}");
  return str;
}

<<<<<<< HEAD
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
=======
OperatorDistAttr OperatorDistAttr::from_proto(
    const OperatorDistAttrProto& proto) {
  OperatorDistAttr dist_attr;
  for (int64_t i = 0; i < proto.input_dist_attrs_size(); ++i) {
    dist_attr.input_dist_attrs_[proto.input_dist_attrs(i).name()] =
        TensorDistAttr::from_proto(
            proto.input_dist_attrs(i).tensor_dist_attr());
  }
  for (int64_t i = 0; i < proto.output_dist_attrs_size(); ++i) {
    dist_attr.output_dist_attrs_[proto.output_dist_attrs(i).name()] =
        TensorDistAttr::from_proto(
            proto.output_dist_attrs(i).tensor_dist_attr());
  }
  dist_attr.process_mesh_ = ProcessMesh::from_proto(proto.process_mesh());
  dist_attr.impl_type_ = proto.impl_type();
  dist_attr.impl_idx_ = proto.impl_idx();
  return dist_attr;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

<<<<<<< HEAD
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

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
  if (lhs.execution_stream() != rhs.execution_stream()) {
    return false;
  }
  if (lhs.stream_priority() != rhs.stream_priority()) {
    return false;
  }
  if (lhs.scheduling_priority() != rhs.scheduling_priority()) {
    return false;
  }
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
