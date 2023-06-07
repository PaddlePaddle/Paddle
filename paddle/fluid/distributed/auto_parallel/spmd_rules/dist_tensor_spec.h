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

#include "paddle/fluid/framework/type_defs.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

using paddle::framework::Attribute;
using paddle::framework::AttributeMap;
using phi::distributed::auto_parallel::ProcessMesh;
using phi::distributed::auto_parallel::TensorDistAttr;

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
  const std::vector<int64_t>& get_dims_mapping() const;

  // set dims_mapping in dist_attr_
  void set_dims_mapping(const std::vector<int64_t>& dims_mapping);

  // get process_mesh from dist_attr_
  const ProcessMesh& get_process_mesh() const;

  // set process_mesh in dist_attr_
  void set_process_mesh(const ProcessMesh& process_mesh);

  const TensorDistAttr& get_dist_attr() const;

  void set_dist_attr(const TensorDistAttr& dist_attr);

  const std::vector<int64_t>& get_shape() const;

  std::string to_string() const;

  // only for testing AttributeMap
  void test_attr_map(const AttributeMap& attr_map);

  // only for testing AttributeMap
  const AttributeMap& get_attr_map() const;

  // only for testing AttributeMap
  void validate_attr_map();

 private:
  std::vector<int64_t> shape_;
  // distributed attributes of the corresponding tensor
  TensorDistAttr dist_attr_;
  AttributeMap attrs_;
};
}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
