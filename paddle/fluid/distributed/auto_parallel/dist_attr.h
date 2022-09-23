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

#include "paddle/fluid/distributed/auto_parallel/auto_parallel.pb.h"
#include "paddle/fluid/distributed/auto_parallel/process_mesh.h"
#include "paddle/fluid/distributed/auto_parallel/utils.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {

// Forward Declaration
namespace framework {

class BlockDesc;
class OpDesc;
class ProgramDesc;
class VarDesc;

}  // namespace framework

namespace distributed {
namespace auto_parallel {

using framework::BlockDesc;
using framework::OpDesc;
using framework::ProgramDesc;
using framework::VarDesc;

class TensorDistAttr {
 public:
  TensorDistAttr() = default;

  explicit TensorDistAttr(const VarDesc& tensor);

  TensorDistAttr(const TensorDistAttr& tensor);

  TensorDistAttr& operator=(const TensorDistAttr& dist_attr);

  const VarDesc* tensor() const { return tensor_; }

  const ProcessMesh& process_mesh() const { return process_mesh_; }

  void set_process_mesh(const ProcessMesh& process_mesh);

  const std::vector<int64_t>& dims_mapping() const { return dims_mapping_; }

  void set_dims_mapping(const std::vector<int64_t>& dims_mapping);

  int64_t batch_dim() const { return batch_dim_; }

  void set_batch_dim(int64_t batch_dim);

  const std::vector<bool>& dynamic_dims() const { return dynamic_dims_; }

  void set_dynamic_dims(const std::vector<bool>& dynamic_dims);

  const std::map<std::string, bool>& annotated() const { return annotated_; }

  void set_annotated(const std::map<std::string, bool>& annotated);

  void set_default_dims_mapping();

  bool is_annotated(const std::string& name) const {
    return annotated_.count(name) == 1;
  }

  void annotate(const std::string& name);

  bool verify_process_mesh(const ProcessMesh& process_mesh) const;

  bool verify_dims_mapping(const std::vector<int64_t>& dims_mapping) const;

  bool verify_batch_dim(int64_t dim) const;

  bool verify_dynamic_dims(const std::vector<bool>& dynamic_dims) const;

  bool verify_annotated(const std::map<std::string, bool>& annotated) const;

  bool verify() const;

  // TensorDistAttr from_string(const std::string& dist_str);
  std::string to_string() const;

  static TensorDistAttr from_proto(const TensorDistAttrProto& proto);

  TensorDistAttrProto to_proto() const;

 private:
  static std::vector<std::string> fields_;
  const VarDesc* tensor_{nullptr};
  ProcessMesh process_mesh_;
  std::vector<int64_t> dims_mapping_;
  int64_t batch_dim_;
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

class OperatorDistAttr {
 public:
  OperatorDistAttr() = default;

  explicit OperatorDistAttr(const OpDesc& op);

  OperatorDistAttr(const OperatorDistAttr& dist_attr);

  OperatorDistAttr& operator=(const OperatorDistAttr& dist_attr);

  const OpDesc* op() const { return op_; }

  const VarDesc& input(const std::string& name) const {
    return *inputs_.at(name);
  }

  const VarDesc& output(const std::string& name) const {
    return *outputs_.at(name);
  }

  const std::map<std::string, TensorDistAttr>& input_dist_attrs() const {
    return input_dist_attrs_;
  }

  const std::map<std::string, TensorDistAttr>& output_dist_attrs() const {
    return output_dist_attrs_;
  }

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

  const std::string& impl_type() const { return impl_type_; }

  void set_impl_type(const std::string& impl_type) { impl_type_ = impl_type; }

  int64_t impl_idx() const { return impl_idx_; }

  void set_impl_idx(const int64_t& impl_idx) { impl_idx_ = impl_idx; }

  const std::map<std::string, bool>& annotated() const { return annotated_; }

  void set_annotated(const std::map<std::string, bool>& annotated);

  bool is_annotated(const std::string& name) const {
    return annotated_.count(name) == 1;
  }

  void annotate(const std::string& name);

  bool verify_input_dist_attr(const std::string& name,
                              const TensorDistAttr& dist_attr) const;

  bool verify_output_dist_attr(const std::string& name,
                               const TensorDistAttr& dist_attr) const;

  bool verify_process_mesh(const ProcessMesh& process_mesh) const;

  bool verify_annotated(const std::map<std::string, bool>& annotated) const;

  bool verify() const;

  // OperatorDistAttr from_string(const std::string& dist_str);
  std::string to_string() const;

  static OperatorDistAttr from_proto(const OperatorDistAttrProto& proto);

  OperatorDistAttrProto to_proto() const;

 private:
  static std::vector<std::string> fields_;
  const OpDesc* op_{nullptr};
  std::map<std::string, VarDesc*> inputs_;
  std::map<std::string, VarDesc*> outputs_;
  std::map<std::string, TensorDistAttr> input_dist_attrs_;
  std::map<std::string, TensorDistAttr> output_dist_attrs_;
  ProcessMesh process_mesh_;
  std::string impl_type_;
  int64_t impl_idx_ = -1;
  std::map<std::string, bool> annotated_;
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
