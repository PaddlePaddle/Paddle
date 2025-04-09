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

#include "paddle/phi/infermeta/spmd_rules/numel.h"

#include "glog/logging.h"

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {
using phi::distributed::auto_parallel::str_join;

SpmdInfo NumelInferSpmd(const DistMetaTensor& x) {
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  auto x_shape = common::vectorize(x.dims());
  int x_ndim = x_shape.size();
  auto x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(x_ndim,
                    x_dims_mapping.size(),
                    common::errors::InvalidArgument(
                        "The Tensor Input's rank [%d] and Input's "
                        "dims_mapping size [%d] are not matched.",
                        x_ndim,
                        x_dims_mapping.size()));
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr.set_dims_mapping({});
  std::vector<int64_t> partial_on_dims;
  const auto& dim_mapping = x_dims_mapping;
  for (int i = 0; i < x_ndim; ++i) {
    auto mapping = dim_mapping[i];
    if (mapping != -1) {
      partial_on_dims.push_back(mapping);
    }
  }
  out_dist_attr.set_partial_status(partial_on_dims);
  return SpmdInfo({x_dist_attr_src}, {out_dist_attr});
}

}  // namespace phi::distributed
