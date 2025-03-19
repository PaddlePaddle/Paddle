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

#include "paddle/phi/infermeta/spmd_rules/pad.h"
#include <numeric>

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/rules.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

SpmdInfo PadInferSpmd(const DistMetaTensor& x,
                      const std::vector<int>& paddings,
                      int pad_value) {
  auto x_shape = phi::vectorize(x.dims());
  int x_ndim = x_shape.size();
  auto x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping.size(),
      common::errors::InvalidArgument("The Tensor X's rank [%d] and X's "
                                      "dims_mapping size [%d] are not matched.",
                                      x_ndim,
                                      x_dims_mapping.size()));
  std::vector<int64_t> dims_to_unshard;
  for (size_t i = 0; i < paddings.size(); i += 2) {
    if (paddings[i] != 0 || paddings[i + 1] != 0) {
      dims_to_unshard.push_back(i / 2);
    }
  }
  auto x_dist_attr = UnShardTensorDims(x_dist_attr_src, dims_to_unshard);
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr);
  out_dist_attr.set_dims_mapping(x_dist_attr.dims_mapping());

  VLOG(4) << "PadInferSpmd: X shape: [" << str_join(x_shape) << "]";
  VLOG(4) << "X dims_mapping: [" << str_join(x_dist_attr.dims_mapping())
          << "] Out dims_mapping: [" << str_join(out_dist_attr.dims_mapping())
          << "]";

  return {{x_dist_attr}, {out_dist_attr}};
}

SpmdInfo PadGradInferSpmd(const DistMetaTensor& x,
                          const DistMetaTensor& out,
                          const std::vector<int>& paddings,
                          int pad_value) {
  auto out_shape = phi::vectorize(out.dims());
  int out_ndim = out_shape.size();
  auto out_dist_attr_src = out.dist_attr();
  std::vector<int64_t> out_dims_mapping = out_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      out_ndim,
      out_dims_mapping.size(),
      common::errors::InvalidArgument("The Tensor Out's rank [%d] and Out's "
                                      "dims_mapping size [%d] are not matched.",
                                      out_ndim,
                                      out_dims_mapping.size()));

  std::vector<int64_t> dims_to_unshard;
  for (size_t i = 0; i < paddings.size(); i += 2) {
    if (paddings[i] != 0 || paddings[i + 1] != 0) {
      dims_to_unshard.push_back(i / 2);
    }
  }
  auto out_dist_attr = UnShardTensorDims(out_dist_attr_src, dims_to_unshard);
  TensorDistAttr x_dist_attr = CopyTensorDistAttrForOutput(out_dist_attr);
  x_dist_attr.set_dims_mapping(out_dist_attr.dims_mapping());

  VLOG(4) << "PadInferSpmdReverse: Out shape: [" << str_join(out_shape) << "]";
  VLOG(4) << "Out dims_mapping: [" << str_join(x_dist_attr.dims_mapping())
          << "] X dims_mapping: [" << str_join(x_dist_attr.dims_mapping())
          << "]";

  return {{x_dist_attr}, {out_dist_attr}};
}

SpmdInfo PadInferSpmdDynamic(const DistMetaTensor& x,
                             const std::vector<int>& paddings,
                             const Scalar& pad_value) {
  return PadInferSpmd(x, paddings, pad_value.to<int32_t>());
}

SpmdInfo PadGradInferSpmdDynamic(const DistMetaTensor& out_grad,
                                 const std::vector<int>& paddings,
                                 const Scalar& pad_value) {
  return PadInferSpmd(out_grad, paddings, pad_value.to<int32_t>());
}

}  // namespace distributed
}  // namespace phi
