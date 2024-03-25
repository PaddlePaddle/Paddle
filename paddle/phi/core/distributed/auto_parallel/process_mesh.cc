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

#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"

#include <algorithm>
#include <iterator>

#include "paddle/phi/core/distributed/auto_parallel/proto_helper.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"

namespace phi {
namespace distributed {

using phi::distributed::auto_parallel::has_duplicates;
using phi::distributed::auto_parallel::ProcessMeshProto;
using phi::distributed::auto_parallel::str_join;

ProcessMesh::ProcessMesh(const std::vector<int64_t> &shape,
                         const std::vector<int64_t> &process_ids,
                         const std::vector<std::string> &dim_names) {
  shape_ = shape;
  int64_t size = this->size();
  PADDLE_ENFORCE_EQ(
      size,
      process_ids.size(),
      errors::InvalidArgument("The size of this process mesh must be "
                              "equal to the size of its process ids.",
                              size,
                              process_ids.size()));
  PADDLE_ENFORCE_EQ(
      has_duplicates(process_ids),
      false,
      errors::InvalidArgument("The process ids [%s] must be unique.",
                              str_join(process_ids_)));
  process_ids_ = process_ids;

  PADDLE_ENFORCE_EQ(shape_.size(),
                    dim_names.size(),
                    errors::InvalidArgument(
                        "The size of mesh shape must be equal to the size "
                        "of the dimension names.",
                        shape_.size(),
                        dim_names_.size()));
  PADDLE_ENFORCE_EQ(has_duplicates(dim_names),
                    false,
                    errors::InvalidArgument(
                        "The names [%s] of each dimension must be unique.",
                        str_join(dim_names)));
  dim_names_ = dim_names;
}

int64_t ProcessMesh::size() const {
  if (shape_.empty()) return 0;
  int64_t size = 1;
  for (const int64_t dim_size : shape_) size *= dim_size;
  return size;
}

bool ProcessMesh::contains(int64_t process_id) const {
  auto result =
      std::find(std::begin(process_ids_), std::end(process_ids_), process_id);
  if (result != std::end(process_ids_)) {
    return true;
  } else {
    return false;
  }
}

std::string ProcessMesh::to_string() const {
  std::string mesh_str = "{shape: [" + str_join(shape_) + "], ";
  mesh_str += "process_ids: [" + str_join(process_ids_) + "], ";
  mesh_str += "dim_names: [" + str_join(dim_names_) + "]}";
  return mesh_str;
}

ProcessMesh ProcessMesh::from_proto(const ProcessMeshProto &proto) {
  ProcessMesh mesh;

  mesh.shape_.resize(proto.shape_size());
  for (int i = 0; i < proto.shape_size(); ++i) {
    mesh.shape_[i] = proto.shape(i);
  }

  mesh.process_ids_.resize(proto.process_ids_size());
  for (int i = 0; i < proto.process_ids_size(); ++i) {
    mesh.process_ids_[i] = proto.process_ids(i);
  }

  mesh.dim_names_.resize(proto.dim_names_size());
  for (int i = 0; i < proto.dim_names_size(); ++i) {
    mesh.dim_names_[i] = proto.dim_names(i);
  }

  return mesh;
}

void ProcessMesh::to_proto(ProcessMeshProto *proto) const {
  for (const auto &i : shape_) {
    proto->add_shape(i);
  }

  for (const auto &i : process_ids_) {
    proto->add_process_ids(i);
  }

  for (const auto &i : dim_names_) {
    proto->add_dim_names(i);
  }
}

bool operator==(const ProcessMesh &lhs, const ProcessMesh &rhs) {
  if (lhs.shape() != rhs.shape()) {
    return false;
  }
  if (lhs.process_ids() != rhs.process_ids()) {
    return false;
  }
  return true;
}

}  // namespace distributed
}  // namespace phi
