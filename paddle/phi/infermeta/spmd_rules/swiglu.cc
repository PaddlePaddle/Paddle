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

#include "paddle/phi/infermeta/spmd_rules/elementwise.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

SpmdInfo SwiGLUInferSpmd(const DistMetaTensor& x, const DistMetaTensor& y) {
  // y.dist_attr() is empty means y is None
  if (y.dist_attr() == TensorDistAttr()) {
    auto x_dims_mapping = x.dist_attr().dims_mapping();
    if (x_dims_mapping.back() != -1) {
      PADDLE_THROW(
          phi::errors::Unimplemented("The input y is none and input x's last "
                                     "dim is sharded is not supported"));
    }
    return ElementwiseUnaryInferSpmd(x);
  } else {
    return ElementwiseBinaryInferSpmd(x, y);
  }
}

SpmdInfo SwiGLUInferSpmdReverse(const DistMetaTensor& x,
                                const DistMetaTensor& y,
                                const DistMetaTensor& out) {
  if (y.dist_attr() == TensorDistAttr()) {
    auto x_dims_mapping = x.dist_attr().dims_mapping();
    if (x_dims_mapping.back() != -1) {
      PADDLE_THROW(
          phi::errors::Unimplemented("The input y is none and input x's last "
                                     "dim is sharded is not supported"));
    }
    return ElementwiseUnaryInferSpmdReverse(x, out);
  } else {
    return ElementwiseBinaryInferSpmdReverse(x, y, out);
  }
}

SpmdInfo SwiGLUGradInferSpmd(const DistMetaTensor& x,
                             const DistMetaTensor& y,
                             const DistMetaTensor& out_grad) {
  if (y.dist_attr() == TensorDistAttr()) {
    auto x_dims_mapping = x.dist_attr().dims_mapping();
    if (x_dims_mapping.back() != -1) {
      PADDLE_THROW(
          phi::errors::Unimplemented("The input y is none and input x's last "
                                     "dim is sharded is not supported"));
    }
    return ElementwiseUnaryGradInferSpmd(x, out_grad);
  } else {
    return ElementwiseBinaryGradInferSpmd(x, y, out_grad);
  }
}

}  // namespace distributed
}  // namespace phi
