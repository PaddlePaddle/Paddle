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

#include "paddle/phi/infermeta/spmd_rules/coalesce.h"

namespace phi {
namespace distributed {
SpmdInfo CoalesceTensorInferSpmd(const std::vector<DistMetaTensor>& input,
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
      input.size(),
      0u,
      common::errors::InvalidArgument("CoalesceTensor input can't be empty."));
  auto dist_attr = input[0].dist_attr();
  std::vector<TensorDistAttr> input_attrs{dist_attr};
  auto partial_status = dist_attr.partial_status();
  auto mesh = dist_attr.process_mesh();
  for (size_t idx = 1u; idx < input.size(); ++idx) {
    auto& sub_dist_attr = input[idx].dist_attr();
    PADDLE_ENFORCE_EQ(
        mesh,
        sub_dist_attr.process_mesh(),
        common::errors::InvalidArgument(
            "All input of CoalesceTensor mush have the same mesh."));
    if (partial_status != sub_dist_attr.partial_status()) {
      partial_status.clear();
    }
    input_attrs.push_back(sub_dist_attr);
  }
  TensorDistAttr fused_out_dist_attr;
  fused_out_dist_attr.set_process_mesh(mesh);
  fused_out_dist_attr.set_dims_mapping({-1});
  fused_out_dist_attr.set_partial_status(partial_status);
  return {{input_attrs}, {input_attrs, fused_out_dist_attr}};
}

}  // namespace distributed
}  // namespace phi
