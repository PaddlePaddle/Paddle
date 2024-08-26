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

#include "paddle/phi/infermeta/spmd_rules/fused_dropout_add.h"
#include <numeric>

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/dim_trans.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

SpmdInfo FusedDropoutAddSpmd(const DistMetaTensor& x, const DistMetaTensor& y) {
  SpmdInfo out_info = ElementwiseBinaryInferSpmd(x, y);

  TensorDistAttr seed_offset_dist_attr({2});
  seed_offset_dist_attr.set_dims_mapping({-1});

  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(y);
  VLOG(4) << "out dist_attr: [" << out_info.second[0].to_string() << "]";
  VLOG(4) << "seed_offset dist_attr: [" << seed_offset_dist_attr.to_string()
          << "]";
  return {{x_dist_attr_dst.dist_attr(), y_dist_attr_dst.dist_attr()},
          {out_info.second[0], seed_offset_dist_attr}};
}

SpmdInfo FusedDropoutAddSpmdReverse(const DistMetaTensor& x,
                                    const DistMetaTensor& y,
                                    const DistMetaTensor& out,
                                    const DistMetaTensor& seed_offset) {
  SpmdInfo reverse_info = ElementwiseBinaryInferSpmdReverse(x, y, out);
  LOG_SPMD_INPUT(out);
  LOG_SPMD_INPUT(seed_offset);
  VLOG(4) << "x dist_attr: [" << reverse_info.first[0].to_string() << "]";
  VLOG(4) << "y dist_attr: [" << reverse_info.first[1].to_string() << "]";
  return {reverse_info.first,
          {reverse_info.second[0], seed_offset.dist_attr()}};
}

SpmdInfo FusedDropoutAddGradInferSpmd(const DistMetaTensor& seed_offset,
                                      const DistMetaTensor& out_grad) {
  LOG_SPMD_INPUT(seed_offset);
  LOG_SPMD_INPUT(out_grad);
  VLOG(4) << "x_grad dist_attr: [" << out_grad.dist_attr().to_string() << "]";
  VLOG(4) << "y_grad dist_attr: [" << out_grad.dist_attr().to_string() << "]";
  return {{seed_offset.dist_attr(), out_grad.dist_attr()},
          {out_grad.dist_attr(), out_grad.dist_attr()}};
}

}  // namespace phi::distributed
