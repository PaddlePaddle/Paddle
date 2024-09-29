// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/infermeta/spmd_rules/amp_ops.h"

#include <vector>
#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {
// TODO(zhiqiu): support xs on different mesh.
SpmdInfo CheckFiniteAndUnscaleSpmd(const std::vector<DistMetaTensor>& xs,
                                   const DistMetaTensor& scale) {
  std::vector<TensorDistAttr> xs_attrs;
  paddle::flat_hash_map<int64_t, ReduceType> partial_on_dims;
  auto scale_mesh = scale.dist_attr().process_mesh();
  auto offset = 0;
  for (auto& x : xs) {
    auto dist_attr = x.dist_attr();
    dist_attr.clean_partial_status();
    xs_attrs.emplace_back(dist_attr);
    auto dims_mapping = dist_attr.dims_mapping();
    auto mesh = dist_attr.process_mesh();
    if (scale_mesh.ndim() > 1 && IsSubMesh(scale_mesh, mesh)) {
      partial_on_dims[0] = ReduceType::kRedMax;
      offset = 1;
    }
    for (auto& m : dims_mapping) {
      if (m != -1 && partial_on_dims.count(m) == 0) {
        partial_on_dims[m + offset] = ReduceType::kRedMax;
      }
    }
  }
  TensorDistAttr found_infinite_attr =
      CopyTensorDistAttrForOutput(scale.dist_attr());
  found_infinite_attr.set_partial_status(partial_on_dims);
  found_infinite_attr.set_dims_mapping({-1});
  return {{xs_attrs, scale.dist_attr()}, {xs_attrs, found_infinite_attr}};
}

SpmdInfo UpdateLossScalingSpmd(const std::vector<DistMetaTensor>& xs,
                               const DistMetaTensor& found_infinite,
                               const DistMetaTensor& prev_loss_scaling,
                               const DistMetaTensor& in_good_steps,
                               const DistMetaTensor& in_bad_steps,
                               int incr_every_n_steps,
                               int decr_every_n_nan_or_inf,
                               float incr_ratio,
                               float decr_ratio,
                               Scalar stop_update) {
  std::vector<TensorDistAttr> xs_attrs;
  for (auto& x : xs) {
    auto dist_attr = x.dist_attr();
    dist_attr.clean_partial_status();
    xs_attrs.emplace_back(dist_attr);
  }
  TensorDistAttr found_infinite_attr =
      CopyTensorDistAttrForOutput(found_infinite.dist_attr());
  found_infinite_attr.set_dims_mapping({-1});
  return {{xs_attrs,
           found_infinite_attr,
           prev_loss_scaling.dist_attr(),
           in_good_steps.dist_attr(),
           in_bad_steps.dist_attr()},
          {xs_attrs,
           found_infinite_attr,
           in_good_steps.dist_attr(),
           in_bad_steps.dist_attr()}};
}

}  // namespace distributed
}  // namespace phi
