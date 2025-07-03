/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/spmd_rules/label_smooth.h"
#include "glog/logging.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"
#include "paddle/phi/infermeta/spmd_rules/elementwise.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

std::vector<int64_t> ReplicateWithExcludeAxes(
    const std::vector<int64_t>& x, const std::vector<int64_t>& exclude_axes) {
  std::vector<int64_t> res(x);
  for (size_t i = 0; i < x.size(); ++i) {
    if (res[i] == -1) {
      continue;
    }

    if (std::find(exclude_axes.begin(), exclude_axes.end(), res[i]) !=
        exclude_axes.end()) {
      res[i] = -1;
    }
  }
  return res;
}

SpmdInfo LabelSmoothInferSpmd(const DistMetaTensor& label,
                              const DistMetaTensor& prior_dist,
                              float epsilon) {
  if (prior_dist.initialized()) {
    const auto& label_dims_attr = label.dist_attr();
    const auto& prior_dims_attr = prior_dist.dist_attr();
    const auto& label_dims_mapping = label_dims_attr.dims_mapping();
    const auto& prior_dims_mapping = prior_dims_attr.dims_mapping();
    size_t label_rank = label.dims().size();
    size_t prior_rank = prior_dist.dims().size();

    if (label_rank > prior_rank) {
      size_t extra_dims_count = label_rank - prior_rank;
      std::vector<std::int64_t> extra_dims_mapping_values;
      extra_dims_mapping_values.reserve(extra_dims_count);
      for (size_t i = 0; i < extra_dims_count; ++i) {
        extra_dims_mapping_values.push_back(label_dims_mapping[i]);
      }
      const auto& new_prior_dims_mapping = ReplicateWithExcludeAxes(
          prior_dims_mapping, extra_dims_mapping_values);
      TensorDistAttr new_prior_dist_attr = prior_dims_attr;
      new_prior_dist_attr.set_dims_mapping(new_prior_dims_mapping);
      DistMetaTensor modified_prior_dist(prior_dist.dims(),
                                         new_prior_dist_attr);
      VLOG(4) << "LabelSmoothInferSpmd call ElementwiseBinaryInferSpmd:";
      return ElementwiseBinaryInferSpmd(label, modified_prior_dist);
    }
    VLOG(4) << "LabelSmoothInferSpmd call ElementwiseBinaryInferSpmd:";
    return ElementwiseBinaryInferSpmd(label, prior_dist);
  }
  VLOG(4) << "LabelSmoothInferSpmd call ElementwiseUnaryInferSpmd:";
  SpmdInfo unary_spmd_info = ElementwiseUnaryInferSpmd(label);
  unary_spmd_info.first.push_back(TensorDistAttr());
  return unary_spmd_info;
}

SpmdInfo LabelSmoothGradInferSpmd(const DistMetaTensor& out_grad,
                                  float epsilon) {
  VLOG(4) << "LabelSmoothGradInferSpmd call ElementwiseUnaryGradInferSpmd:";
  return ElementwiseUnaryGradInferSpmd(out_grad);
}
}  // namespace distributed
}  // namespace phi
