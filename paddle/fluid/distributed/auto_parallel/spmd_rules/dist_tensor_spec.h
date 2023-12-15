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

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"

namespace paddle {
namespace distributed {
using phi::distributed::ProcessMesh;
using phi::distributed::TensorDistAttr;

namespace auto_parallel {

/**
 * A unified data class for inferring distributed attributes
 * in both dygraph mode and static mode
 */
class DistTensorSpec {
 public:
  DistTensorSpec() = default;

  DistTensorSpec(const std::vector<int64_t>& shape,
                 const TensorDistAttr& dist_attr);

  DistTensorSpec(const DistTensorSpec& spec);

  // temp function, only for test in dygraph mode
  explicit DistTensorSpec(const Tensor& tensor);

  ~DistTensorSpec();

  DistTensorSpec& operator=(const DistTensorSpec& spec);

  // get dims_mapping from dist_attr_
  const std::vector<int64_t>& dims_mapping() const;

  // set dims_mapping in dist_attr_
  void set_dims_mapping(const std::vector<int64_t>& dims_mapping);

  // get process_mesh from dist_attr_
  const ProcessMesh& process_mesh() const;

  // set process_mesh in dist_attr_
  void set_process_mesh(const ProcessMesh& process_mesh);

  const TensorDistAttr& dist_attr() const;

  void set_dist_attr(const TensorDistAttr& dist_attr);

  const std::vector<int64_t>& shape() const;

  void set_shape(const std::vector<int64_t>& shape);

  std::string to_string() const;

 private:
  std::vector<int64_t> shape_;
  // distributed attributes of the corresponding tensor
  TensorDistAttr dist_attr_;
};
}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
