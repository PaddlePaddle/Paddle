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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "paddle/fluid/distributed/auto_parallel/auto_parallel.pb.h"
#include "paddle/fluid/distributed/auto_parallel/utils.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

class ProcessMesh {
 public:
  ProcessMesh() = default;

  ProcessMesh(const std::vector<int64_t>& shape,
              const std::vector<int64_t>& process_ids,
              const std::vector<std::string>& dim_names,
              const std::string& device_type);

  const std::vector<int64_t>& shape() const { return shape_; }

  const std::vector<int64_t>& process_ids() const { return process_ids_; }

  const std::vector<std::string>& dim_names() const { return dim_names_; }

  std::string device_type() const { return device_type_; }

  void set_device_type(const std::string& device_type) {
    device_type_ = device_type;
  }

  int64_t size() const;

  int64_t ndim() const { return shape_.size(); }

  int64_t dim_size(int64_t dim) const {
    int64_t cdim = canonical_dim(dim, shape_.size());
    return shape_[cdim];
  }

  int64_t dim_size(std::string dim_name) const {
    for (std::size_t i = 0; i < dim_names_.size(); ++i) {
      if (dim_names_[i] == dim_name) {
        return shape_[i];
      }
    }
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Cannot find the dimension of %s in this process mesh.", dim_name));
  }

  ProcessMesh from_string(const std::string& mesh_str);
  std::string to_string() const;

  ProcessMesh from_proto(const ProcessMeshProto& proto);
  ProcessMeshProto to_proto() const;

 private:
  std::vector<int64_t> shape_;
  std::vector<int64_t> process_ids_;
  std::vector<std::string> dim_names_;
  std::string device_type_;
};

inline std::ostream& operator<<(std::ostream& os, const ProcessMesh& obj) {
  os << obj.to_string();
  return os;
}

inline bool operator==(const ProcessMesh& lhs, const ProcessMesh& rhs) {
  if (lhs.shape() != rhs.shape()) {
    return false;
  }
  if (lhs.process_ids() != rhs.process_ids()) {
    return false;
  }
  if (lhs.device_type() != rhs.device_type()) {
    return false;
  }
  return true;
}

inline bool operator!=(const ProcessMesh& lhs, const ProcessMesh& rhs) {
  return !operator==(lhs, rhs);
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
