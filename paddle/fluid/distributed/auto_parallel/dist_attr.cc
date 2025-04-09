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
#include "paddle/phi/core/distributed/auto_parallel/proto_helper.h"

namespace paddle::distributed::auto_parallel {

using phi::distributed::auto_parallel::str_join;

std::vector<int64_t> get_tensor_shape(const VarDesc* tensor) {
  if (tensor == nullptr) return std::vector<int64_t>();
  switch (tensor->GetType()) {
    case framework::proto::VarType::READER:
    case framework::proto::VarType::DENSE_TENSOR_ARRAY:
    case framework::proto::VarType::STEP_SCOPES:
    case framework::proto::VarType::FEED_MINIBATCH:
    case framework::proto::VarType::FETCH_LIST:
      return std::vector<int64_t>();
    default:
      return tensor->GetShape();
  }
}

std::vector<std::string> OperatorDistAttr::fields_{"process_mesh",
                                                   "impl_type",
                                                   "impl_idx",
                                                   "chunk_id",
                                                   "is_recompute",
                                                   "execution_stream",
                                                   "stream_priority",
                                                   "scheduling_priority"};

OperatorDistAttr::OperatorDistAttr(const OpDesc& op) { initialize(&op); }

OperatorDistAttr::OperatorDistAttr(const OperatorDistAttr& dist_attr) {
  copy_from(dist_attr);
}

OperatorDistAttr& OperatorDistAttr::operator=(
    const OperatorDistAttr& dist_attr) {
  if (this == &dist_attr) return *this;
  OperatorDistAttr tmp(dist_attr);
  std::swap(this->input_dist_attrs_, tmp.input_dist_attrs_);
  std::swap(this->output_dist_attrs_, tmp.output_dist_attrs_);
  std::swap(this->process_mesh_, tmp.process_mesh_);
  std::swap(this->op_type_, tmp.op_type_);
  std::swap(this->impl_type_, tmp.impl_type_);
  std::swap(this->impl_idx_, tmp.impl_idx_);
  std::swap(this->chunk_id_, tmp.chunk_id_);
  std::swap(this->is_recompute_, tmp.is_recompute_);
  std::swap(this->execution_stream_, tmp.execution_stream_);
  std::swap(this->stream_priority_, tmp.stream_priority_);
  std::swap(this->scheduling_priority_, tmp.scheduling_priority_);
  std::swap(this->annotated_, tmp.annotated_);
  std::swap(this->run_time_us_, tmp.run_time_us_);
  // Note: Make sure all tensor dist attr has the same process_mesh
  set_process_mesh(this->process_mesh_);
  return *this;
}

void OperatorDistAttr::initialize(const OpDesc* op) {
  if (op == nullptr) return;
  for (std::string const& name : op->InputArgumentNames()) {
    VarDesc* input = op->Block()->FindVarRecursive(name);
    VLOG(4) << "[OperatorDistAttr create input dist attr] " << name;
    if (input == nullptr || op->Type() == "create_py_reader") {
      input_dist_attrs_[name] = TensorDistAttr();
    } else {
      input_dist_attrs_[name] = TensorDistAttr(get_tensor_shape(input));
    }
  }
  for (std::string const& name : op->OutputArgumentNames()) {
    VarDesc* output = op->Block()->FindVarRecursive(name);
    VLOG(4) << "[OperatorDistAttr create output dist attr] " << name;
    if (output == nullptr) {
      output_dist_attrs_[name] = TensorDistAttr();
    } else {
      output_dist_attrs_[name] = TensorDistAttr(get_tensor_shape(output));
    }
  }
  op_type_ = op->Type();
  impl_type_ = kDefault;
  impl_idx_ = 0;
  chunk_id_ = 0;
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
  set_chunk_id(dist_attr.chunk_id());
  set_is_recompute(dist_attr.is_recompute());
  set_execution_stream(dist_attr.execution_stream());
  set_stream_priority(dist_attr.stream_priority());
  set_force_record_event(dist_attr.force_record_event());
  set_event_to_record(dist_attr.event_to_record());
  set_events_to_wait(dist_attr.events_to_wait());
  set_scheduling_priority(dist_attr.scheduling_priority());
  set_annotated(dist_attr.annotated());
  set_run_time_us(dist_attr.run_time_us());
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
}

void OperatorDistAttr::set_input_dist_attr(const std::string& name,
                                           const TensorDistAttr& dist_attr) {
  input_dist_attrs_[name] = dist_attr;
  // Make sure the process mesh of input be same as that of the op
  input_dist_attrs_[name].set_process_mesh(process_mesh_);
}

void OperatorDistAttr::set_output_dist_attr(const std::string& name,
                                            const TensorDistAttr& dist_attr) {
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

void OperatorDistAttr::set_annotated(
    const std::map<std::string, bool>& annotated) {
  annotated_ = annotated;
}

void OperatorDistAttr::mark_annotated(const std::string& name) {
  auto result = std::find(std::begin(fields_), std::end(fields_), name);
  if (result != std::end(fields_)) {
    annotated_[name] = true;
  }
  if (name == "process_mesh") {
    for (auto& item : input_dist_attrs_) {
      item.second.mark_annotated(name);
    }
    for (auto& item : output_dist_attrs_) {
      item.second.mark_annotated(name);
    }
  }
}

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
  auto tensor_shape = get_tensor_shape(tensor);
  if (!dist_attr.verify(tensor_shape)) {
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
  auto tensor_shape = get_tensor_shape(tensor);
  if (!dist_attr.verify(tensor_shape)) {
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

bool OperatorDistAttr::verify(const OpDesc* op) const {
  if (!verify_process_mesh(process_mesh_)) {
    return false;
  }
  for (auto const& item : input_dist_attrs_) {
    auto input_names = op->InputArgumentNames();
    auto found =
        std::find(std::begin(input_names), std::end(input_names), item.first);
    if (found == std::end(input_names)) {
      return false;
    }
    auto tensor = op->Block()->FindVarRecursive(item.first);
    if (!verify_input_dist_attr(item.first, item.second, tensor)) {
      return false;
    }
  }
  for (auto const& item : output_dist_attrs_) {
    auto output_names = op->OutputArgumentNames();
    auto found =
        std::find(std::begin(output_names), std::end(output_names), item.first);
    if (found == std::end(output_names)) {
      return false;
    }
    auto tensor = op->Block()->FindVarRecursive(item.first);
    if (!verify_output_dist_attr(item.first, item.second, tensor)) {
      return false;
    }
  }
  return true;
}

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
  str += "chunk_id: " + std::to_string(chunk_id_) + ", ";
  str += "execution_stream: " + execution_stream_ + ", ";
  str += "stream_priority: " + std::to_string(stream_priority_) + ", ";
  str += "scheduling_priority: " + std::to_string(scheduling_priority_) + ", ";
  str += "annotated: [" + str_join(annotated_) + "], ";
  str += "\nprocess_mesh: " + process_mesh_.to_string() + ", ";
  str += "\ninput_dist_attrs: [\n";
  for (auto const& item : input_dist_attrs_) {
    str += "  " + item.first + ": " + item.second.to_string() + ",\n";
  }
  str.replace(str.size() - 2, 2, "]");
  str += "\noutput_dist_attrs: [\n";
  for (auto const& item : output_dist_attrs_) {
    str += "  " + item.first + ": " + item.second.to_string() + ",\n";
  }
  str.replace(str.size() - 2, 2, "]}");
  return str;
}

void OperatorDistAttr::from_proto(const OperatorDistAttrProto& proto) {
  for (int i = 0; i < proto.input_dist_attrs_size(); ++i) {
    TensorDistAttr dist_attr;
    std::string name = proto.input_dist_attrs(i).name();
    dist_attr.from_proto(proto.input_dist_attrs(i).tensor_dist_attr());
    input_dist_attrs_[name] = dist_attr;
  }
  for (int i = 0; i < proto.output_dist_attrs_size(); ++i) {
    TensorDistAttr dist_attr;
    std::string name = proto.output_dist_attrs(i).name();
    dist_attr.from_proto(proto.output_dist_attrs(i).tensor_dist_attr());
    output_dist_attrs_[name] = dist_attr;
  }
  process_mesh_ = ProcessMesh::from_proto(proto.process_mesh());
  impl_type_ = proto.impl_type();
  impl_idx_ = proto.impl_idx();
  chunk_id_ = proto.chunk_id();
}

OperatorDistAttrProto OperatorDistAttr::to_proto() const {
  OperatorDistAttrProto proto;
  for (const auto& item : input_dist_attrs_) {
    auto proto_item = proto.mutable_input_dist_attrs()->Add();
    proto_item->set_name(item.first);
    proto_item->mutable_tensor_dist_attr()->CopyFrom(
        phi::distributed::to_proto(item.second));
  }
  for (const auto& item : output_dist_attrs_) {
    auto proto_item = proto.mutable_output_dist_attrs()->Add();
    proto_item->set_name(item.first);
    proto_item->mutable_tensor_dist_attr()->CopyFrom(
        phi::distributed::to_proto(item.second));
  }
  proto.mutable_process_mesh()->CopyFrom(
      phi::distributed::to_proto(process_mesh_));
  proto.set_impl_type(impl_type_);
  proto.set_impl_idx(impl_idx_);
  proto.set_chunk_id(chunk_id_);
  return proto;
}

std::string OperatorDistAttr::serialize_to_string() {
  std::string data;
  auto proto = to_proto();
  proto.SerializeToString(&data);
  PADDLE_ENFORCE_EQ(to_proto().SerializeToString(&data),
                    true,
                    common::errors::InvalidArgument(
                        "Failed to serialize op dist attr to string."));
  return data;
}

void OperatorDistAttr::parse_from_string(const std::string& data) {
  OperatorDistAttrProto proto;
  PADDLE_ENFORCE_EQ(proto.ParseFromString(data),
                    true,
                    common::errors::InvalidArgument(
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
  if (lhs.chunk_id() != rhs.chunk_id()) {
    return false;
  }
  if (lhs.execution_stream() != rhs.execution_stream()) {
    return false;
  }
  if (lhs.stream_priority() != rhs.stream_priority()) {
    return false;
  }
  if (lhs.scheduling_priority() != rhs.scheduling_priority()) {
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

}  // namespace paddle::distributed::auto_parallel
