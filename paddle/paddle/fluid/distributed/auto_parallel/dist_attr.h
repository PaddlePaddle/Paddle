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
#include <string>
#include <vector>

#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/core/distributed/auto_parallel/auto_parallel.pb.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"

namespace paddle {

// Forward Declaration
namespace framework {

class BlockDesc;
class OpDesc;
class ProgramDesc;
class VarDesc;

}  // namespace framework

namespace distributed {

using phi::distributed::ProcessMesh;
using phi::distributed::TensorDistAttr;

namespace auto_parallel {

using framework::BlockDesc;
using framework::OpDesc;
using framework::ProgramDesc;
using framework::VarDesc;

using phi::distributed::auto_parallel::OperatorDistAttrProto;

constexpr const char* kDefault = "default";

std::vector<int64_t> get_tensor_shape(const VarDesc* tensor);

class OperatorDistAttr {
 public:
  OperatorDistAttr() = default;

  explicit OperatorDistAttr(const OpDesc& op);

  OperatorDistAttr(const OperatorDistAttr& dist_attr);

  OperatorDistAttr& operator=(const OperatorDistAttr& dist_attr);

  void initialize(const OpDesc* op = nullptr);

  void copy_from(const OperatorDistAttr& dist_attr);

  const std::map<std::string, TensorDistAttr>& input_dist_attrs() const {
    return input_dist_attrs_;
  }

  std::map<std::string, TensorDistAttr>& input_dist_attrs() {
    return input_dist_attrs_;
  }

  void set_input_dist_attrs(
      const std::map<std::string, TensorDistAttr>& dist_attrs);

  const std::map<std::string, TensorDistAttr>& output_dist_attrs() const {
    return output_dist_attrs_;
  }

  std::map<std::string, TensorDistAttr>& output_dist_attrs() {
    return output_dist_attrs_;
  }

  void set_output_dist_attrs(
      const std::map<std::string, TensorDistAttr>& dist_attrs);

  const TensorDistAttr& input_dist_attr(const std::string& name) const {
    return input_dist_attrs_.at(name);
  }

  TensorDistAttr& input_dist_attr(const std::string& name) {
    return input_dist_attrs_.at(name);
  }

  void set_input_dist_attr(const std::string& name,
                           const TensorDistAttr& dist_attr);

  const TensorDistAttr& output_dist_attr(const std::string& name) const {
    return output_dist_attrs_.at(name);
  }

  TensorDistAttr& output_dist_attr(const std::string& name) {
    return output_dist_attrs_.at(name);
  }

  void set_output_dist_attr(const std::string& name,
                            const TensorDistAttr& dist_attr);

  const ProcessMesh& process_mesh() const { return process_mesh_; }

  void set_process_mesh(const ProcessMesh& process_mesh);

  const std::string& op_type() const { return op_type_; }

  void set_op_type(const std::string& op_type) { op_type_ = op_type; }

  const std::string& impl_type() const { return impl_type_; }

  void set_impl_type(const std::string& impl_type) { impl_type_ = impl_type; }

  int64_t impl_idx() const { return impl_idx_; }

  void set_impl_idx(const int64_t& impl_idx) { impl_idx_ = impl_idx; }

  int64_t chunk_id() const { return chunk_id_; }

  void set_chunk_id(const int64_t& chunk_id) { chunk_id_ = chunk_id; }

  bool is_recompute() const { return is_recompute_; }

  void set_is_recompute(bool is_recompute) { is_recompute_ = is_recompute; }

  const std::string& execution_stream() const { return execution_stream_; }

  void set_execution_stream(const std::string& execution_stream) {
    execution_stream_ = execution_stream;
  }

  void set_event_to_record(const std::string& event_name) {
    event_to_record_ = event_name;
  }

  void set_force_record_event(bool force_record_event) {
    force_record_event_ = force_record_event;
  }

  void set_events_to_wait(const std::vector<std::string>& events_to_wait) {
    events_to_wait_ = events_to_wait;
  }

  bool force_record_event() const { return force_record_event_; }

  const std::string& event_to_record() const { return event_to_record_; }

  const std::vector<std::string>& events_to_wait() const {
    return events_to_wait_;
  }

  int stream_priority() const { return stream_priority_; }

  void set_stream_priority(int stream_priority) {
    stream_priority_ = stream_priority;
  }

  int64_t scheduling_priority() const { return scheduling_priority_; }

  void set_scheduling_priority(int64_t scheduling_priority) {
    scheduling_priority_ = scheduling_priority;
  }

  const std::map<std::string, bool>& annotated() const { return annotated_; }

  void set_annotated(const std::map<std::string, bool>& annotated);

  bool is_annotated(const std::string& name) const {
    return annotated_.count(name) == 1 && annotated_.at(name) == true;
  }

  void mark_annotated(const std::string& name);

  void clear_annotated();

  const std::vector<int64_t>& input_dims_mapping(const std::string& name) const;

  void set_input_dims_mapping(const std::string& name,
                              const std::vector<int64_t>& dims_mapping);

  const std::vector<int64_t>& output_dims_mapping(const std::string& name);

  void set_output_dims_mapping(const std::string& name,
                               const std::vector<int64_t>& dims_mapping);

  bool verify_input_dist_attr(const std::string& name,
                              const TensorDistAttr& dist_attr,
                              const VarDesc* tensor) const;

  bool verify_output_dist_attr(const std::string& name,
                               const TensorDistAttr& dist_attr,
                               const VarDesc* tensor) const;

  bool verify_process_mesh(const ProcessMesh& process_mesh) const;

  bool verify_annotated(const std::map<std::string, bool>& annotated) const;

  bool verify(const OpDesc* op = nullptr) const;

  void rename_input(const std::string& old_name, const std::string& new_name);

  void rename_output(const std::string& old_name, const std::string& new_name);

  // OperatorDistAttr from_string(const std::string& dist_str);
  std::string to_string() const;

  void from_proto(const OperatorDistAttrProto& proto);

  OperatorDistAttrProto to_proto() const;

  std::string serialize_to_string();

  void parse_from_string(const std::string& data);

  static std::string unique_name(std::string key) {
    static std::atomic<int> id_{0};
    return key + "_" + std::to_string(id_++);
  }

  double run_time_us() const { return this->run_time_us_; }
  void set_run_time_us(const double& us) { this->run_time_us_ = us; }

 private:
  static std::vector<std::string> fields_;
  std::map<std::string, TensorDistAttr> input_dist_attrs_;
  std::map<std::string, TensorDistAttr> output_dist_attrs_;
  ProcessMesh process_mesh_;
  std::string op_type_;
  std::string impl_type_ = kDefault;
  int64_t impl_idx_ = 0;
  int64_t chunk_id_ = 0;
  bool is_recompute_ = false;
  std::string execution_stream_ = kDefault;
  bool force_record_event_ = false;
  std::vector<std::string> events_to_wait_;
  std::string event_to_record_ = unique_name("event");  // event_idx
  int stream_priority_ = 0;          // lower value, higher priority
  int64_t scheduling_priority_ = 0;  // lower value, higher priority
  std::map<std::string, bool> annotated_;
  double run_time_us_ = -1.0;  // stores the actual run time (us) of relevant
                               // op, negative value means invalid.
};

inline std::ostream& operator<<(std::ostream& os, const OperatorDistAttr& obj) {
  os << obj.to_string();
  return os;
}

bool operator==(const OperatorDistAttr& lhs, const OperatorDistAttr& rhs);

inline bool operator!=(const OperatorDistAttr& lhs,
                       const OperatorDistAttr& rhs) {
  return !operator==(lhs, rhs);
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
