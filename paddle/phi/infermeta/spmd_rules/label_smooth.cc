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
SpmdInfo LabelSmoothInferSpmd(const DistMetaTensor& label,
                              const DistMetaTensor& prior_dist,
                              float epsilon) {
  if (prior_dist.initialized()) {
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
