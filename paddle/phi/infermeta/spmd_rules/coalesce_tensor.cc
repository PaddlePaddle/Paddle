/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/spmd_rules/coalesce_tensor.h"

namespace phi {
namespace distributed {
SpmdInfo CoalesceTensorInferSpmd(const std::vector<DistMetaTensor>& inputs,
                                 DataType dtype,
                                 bool copy_data,
                                 bool set_constant,
                                 bool persist_output,
                                 float constant,
                                 bool use_align,
                                 int align_size,
                                 int size_of_dtype,
                                 const std::vector<int64_t>& concated_shapes,
                                 const std::vector<int64_t>& concated_ranks) {
  PADDLE_ENFORCE_GT(
      inputs.size(),
      0u,
      common::errors::InvalidArgument("CoalesceTensor input can't be empty."));
  auto dist_attr = inputs[0].dist_attr();
  std::vector<TensorDistAttr> inputs_attrs;
  auto mesh = dist_attr.process_mesh();
  for (auto input : inputs) {
    inputs_attrs.push_back(input.dist_attr());
  }
  TensorDistAttr fused_out_dist_attr;
  fused_out_dist_attr.set_process_mesh(mesh);
  fused_out_dist_attr.set_dims_mapping({-1});
  return {{inputs_attrs}, {inputs_attrs, fused_out_dist_attr}};
}

}  // namespace distributed
}  // namespace phi
