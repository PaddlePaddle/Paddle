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
#include <numeric>
#include <set>

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
  if (lhs.shape() == rhs.shape() && lhs.process_ids() == rhs.process_ids()) {
    return true;
  }
  if (lhs.process_ids() != rhs.process_ids()) {
    return false;
  }

  // if the process ids are the same, and the shapes after
  // removing all `1` are the same, then they are equal.
  std::vector<int64_t> new_lhs_shape = lhs.shape();
  std::vector<int64_t> new_rhs_shape = rhs.shape();
  new_lhs_shape.erase(
      std::remove(new_lhs_shape.begin(), new_lhs_shape.end(), 1),
      new_lhs_shape.end());
  new_rhs_shape.erase(
      std::remove(new_rhs_shape.begin(), new_rhs_shape.end(), 1),
      new_rhs_shape.end());

  return new_lhs_shape == new_rhs_shape;
}

std::vector<ProcessMesh> SplitMesh(const ProcessMesh &mesh, int axis) {
  std::vector<int64_t> mesh_shape = mesh.shape();
  std::vector<int64_t> process_ids = mesh.process_ids();
  std::vector<ProcessMesh> result;

  int64_t total_elements = process_ids.size();

  int64_t num_splits = mesh_shape[axis];
  int64_t prod_before = std::accumulate(mesh_shape.begin(),
                                        mesh_shape.begin() + axis,
                                        1,
                                        std::multiplies<int64_t>());
  int64_t prod_after = std::accumulate(mesh_shape.begin() + axis + 1,
                                       mesh_shape.end(),
                                       1,
                                       std::multiplies<int64_t>());

  for (int i = 0; i < num_splits; ++i) {
    std::vector<int64_t> new_shape = mesh_shape;
    new_shape[axis] = 1;

    std::vector<int64_t> new_process_ids;
    for (int64_t j = 0; j < prod_before; ++j) {
      for (int64_t k = 0; k < prod_after; ++k) {
        int64_t index = j * mesh_shape[axis] * prod_after + i * prod_after + k;
        if (index < total_elements) {
          new_process_ids.push_back(process_ids[index]);
        }
      }
    }

    result.emplace_back(
        ProcessMesh(new_shape, new_process_ids, mesh.dim_names()));
  }

  return result;
}

int SubMeshDim(const ProcessMesh &global_mesh, const ProcessMesh &sub_mesh) {
  std::set<int64_t> global_ids(global_mesh.process_ids().begin(),
                               global_mesh.process_ids().end());
  std::set<int64_t> sub_ids(sub_mesh.process_ids().begin(),
                            sub_mesh.process_ids().end());
  if (!std::includes(
          global_ids.begin(), global_ids.end(), sub_ids.begin(), sub_ids.end()))
    return -1;

  int sub_dim = -1;
  std::vector<int64_t> global_shape = global_mesh.shape();
  std::vector<int64_t> sub_shape = sub_mesh.shape();

  if (global_mesh.ndim() == sub_mesh.ndim() + 1) {
    // for the case that the `1` is not explicitly specified in the shape
    // only supports the case that the sub_mesh is splitted from the 0-th
    // from global_mesh now.
    // e.g.
    //  global_mesh: shape = [2,3], process_ids = [0,1,2,3,4,5]
    //  sub_mesh: shape = [3], process_ids = [0,1,2]
    if (std::equal(global_shape.begin() + 1,
                   global_shape.end(),
                   sub_shape.begin(),
                   sub_shape.end())) {
      sub_dim = 0;
    }
    return sub_dim;
  } else if (global_mesh.ndim() != sub_mesh.ndim()) {
    return -1;
  }

  auto it = std::find(sub_shape.begin(), sub_shape.end(), 1);
  if (it == sub_shape.end()) {
    return -1;
  }

  sub_dim = it - sub_shape.begin();
  std::vector<ProcessMesh> sub_meshes = SplitMesh(global_mesh, sub_dim);
  if (std::find(sub_meshes.begin(), sub_meshes.end(), sub_mesh) !=
      sub_meshes.end()) {
    return sub_dim;
  }

  return -1;
}

}  // namespace distributed
}  // namespace phi
