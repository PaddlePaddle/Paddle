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
#include "paddle/phi/infermeta/spmd_rules/rules.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {
SpmdInfo FusedDropoutAddSpmdBase(const DistMetaTensor& x,
                                 const DistMetaTensor& y) {
  SpmdInfo out_info = ElementwiseBinaryInferSpmd(x, y);

  TensorDistAttr seed_offset_dist_attr({2});
  seed_offset_dist_attr.set_process_mesh(x.dist_attr().process_mesh());
  seed_offset_dist_attr.set_dims_mapping({-1});

  VLOG(4) << "x dist_attr: [" << x.dist_attr().to_string() << "]";
  VLOG(4) << "y dist_attr: [" << y.dist_attr().to_string() << "]";
  VLOG(4) << "out dist_attr: ["
          << paddle::get<0>(out_info.second[0]).to_string() << "]";
  VLOG(4) << "seed_offset dist_attr: [" << seed_offset_dist_attr.to_string()
          << "]";
  return {{x.dist_attr(), y.dist_attr()},
          {out_info.second[0], seed_offset_dist_attr}};
}

SpmdInfo FusedDropoutAddSpmdReverseBase(const DistMetaTensor& x,
                                        const DistMetaTensor& y,
                                        const DistMetaTensor& out,
                                        const DistMetaTensor& seed_offset) {
  SpmdInfo reverse_info = ElementwiseBinaryInferSpmdReverse(x, y, out);

  VLOG(4) << "out dist_attr: [" << out.dist_attr().to_string() << "]";
  VLOG(4) << "x dist_attr: ["
          << paddle::get<0>(reverse_info.first[0]).to_string() << "]";
  VLOG(4) << "y dist_attr: ["
          << paddle::get<0>(reverse_info.first[1]).to_string() << "]";
  return {reverse_info.first,
          {reverse_info.second[0], seed_offset.dist_attr()}};
}

SpmdInfo FusedDropoutAddGradInferSpmdBase(const DistMetaTensor& seed_offset,
                                          const DistMetaTensor& out_grad) {
  VLOG(4) << "seed_offset dist_attr: [" << seed_offset.dist_attr().to_string()
          << "]";
  VLOG(4) << "out_grad dist_attr: [" << out_grad.dist_attr().to_string() << "]";
  VLOG(4) << "x_grad dist_attr: [" << out_grad.dist_attr().to_string() << "]";
  VLOG(4) << "y_grad dist_attr: [" << out_grad.dist_attr().to_string() << "]";
  return {{seed_offset.dist_attr(), out_grad.dist_attr()},
          {out_grad.dist_attr(), out_grad.dist_attr()}};
}

SpmdInfo FusedDropoutAddSpmd(const DistMetaTensor& x,
                             const DistMetaTensor& y,
                             const DistMetaTensor& seed_tensor,
                             const Scalar& p,
                             bool is_test,
                             const std::string& mode,
                             int seed,
                             bool fix_seed) {
  auto dropout_info = FusedDropoutAddSpmdBase(x, y);
  dropout_info.first.push_back(seed_tensor.dist_attr());
  return dropout_info;
}

SpmdInfo FusedDropoutAddSpmdReverse(const DistMetaTensor& x,
                                    const DistMetaTensor& y,
                                    const DistMetaTensor& seed_tensor,
                                    const DistMetaTensor& out,
                                    const DistMetaTensor& seed_offset,
                                    const Scalar& p,
                                    bool is_test,
                                    const std::string& mode,
                                    int seed,
                                    bool fix_seed) {
  return FusedDropoutAddSpmdReverseBase(x, y, out, seed_offset);
}

SpmdInfo FusedDropoutAddGradInferSpmd(const DistMetaTensor& seed_offset,
                                      const DistMetaTensor& out_grad,
                                      const Scalar& p,
                                      bool is_test,
                                      std::string mode,
                                      bool fix_seed) {
  return FusedDropoutAddGradInferSpmdBase(seed_offset, out_grad);
}

}  // namespace phi::distributed
