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

#pragma once

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/dist_tensor_spec.h"

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

DistTensorSpec::~DistTensorSpec() {}

const std::vector<int64_t>& DistTensorSpec::get_dims_mapping() {
  return dist_attr_.dims_mapping();
}

void DistTensorSpec::set_dims_mapping(
    const std::vector<int64_t>& dims_mapping) {
  dist_attr_.set_dims_mapping(dims_mapping);
}

const ProcessMesh& DistTensorSpec::get_process_mesh() {
  return dist_attr_.process_mesh();
}

void DistTensorSpec::set_process_mesh(const ProcessMesh& process_mesh) {
  dist_attr_.set_process_mesh(process_mesh);
}

const std::vector<int64_t>& DistTensorSpec::get_shape() { return shape_; }

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
