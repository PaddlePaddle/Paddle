/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/dist_tensor_spec.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

DistTensorSpec::DistTensorSpec(const std::vector<int64_t>& shape,
                               const TensorDistAttr& dist_attr) {
  shape_.assign(shape.begin(), shape.end());
  // we should merge the new distributed attributes with the original one
  // after inferencing, thus we get a copy of the original one
  dist_attr_.copy_from(dist_attr);
}

DistTensorSpec::DistTensorSpec(const DistTensorSpec& spec) {
  std::vector<int64_t> spec_shape = spec.get_shape();
  shape_.assign(spec_shape.begin(), spec_shape.end());
  dist_attr_.copy_from(spec.get_dist_attr());
}

DistTensorSpec::~DistTensorSpec() {}

DistTensorSpec::DistTensorSpec(const Tensor& tensor) {
  shape_ = tensor.shape();

  // std::vector<int64_t> pm_shape, pm_ids;
  // pm_shape = {4};
  // pm_ids = {0, 1, 2, 3};
  // std::vector<std::string> dim_name = {"mp"};

  // ProcessMesh pm(pm_shape, pm_ids, dim_name);
  // std::vector<int64_t> dims_mapping = {-1, 0};
  // TensorDistAttr dist_attr;
  // dist_attr.set_process_mesh(pm);
  // dist_attr.set_dims_mapping(dims_mapping);

  // dist_attr_.copy_from(dist_attr);

  // std::cout << dist_attr_;
}

DistTensorSpec& DistTensorSpec::operator=(const DistTensorSpec& spec) {
  std::vector<int64_t> spec_shape = spec.get_shape();
  shape_ = spec_shape;
  dist_attr_.copy_from(spec.get_dist_attr());
  return *this;
}

const std::vector<int64_t>& DistTensorSpec::get_dims_mapping() const {
  return dist_attr_.dims_mapping();
}

void DistTensorSpec::set_dims_mapping(
    const std::vector<int64_t>& dims_mapping) {
  dist_attr_.set_dims_mapping(dims_mapping);
}

const ProcessMesh& DistTensorSpec::get_process_mesh() const {
  return dist_attr_.process_mesh();
}

void DistTensorSpec::set_process_mesh(const ProcessMesh& process_mesh) {
  dist_attr_.set_process_mesh(process_mesh);
}

const std::vector<int64_t>& DistTensorSpec::get_shape() const { return shape_; }

void DistTensorSpec::set_shape(const std::vector<int64_t>& shape) {
  shape_ = shape;
}
const TensorDistAttr& DistTensorSpec::get_dist_attr() const {
  return dist_attr_;
}

void DistTensorSpec::set_dist_attr(const TensorDistAttr& dist_attr) {
  dist_attr_ = dist_attr;
}

std::string DistTensorSpec::to_string() const {
  using phi::distributed::auto_parallel::str_join;
  std::string spec_str = "{tensor_shape:[" + str_join(shape_) + "], ";
  spec_str += "dist_attr:" + dist_attr_.to_string() + "}";
  return spec_str;
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
