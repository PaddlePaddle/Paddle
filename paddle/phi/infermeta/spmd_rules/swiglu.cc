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
  if (y) {
    return ElementwiseBinaryInferSpmd(x, y);
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("The input y is not allowed to be None"));
  }
}

SpmdInfo SwiGLUInferSpmdReverse(const DistMetaTensor& x,
                                const DistMetaTensor& y,
                                const DistMetaTensor& out) {
  if (y) {
    return ElementwiseBinaryInferSpmdReverse(x, y, out);
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("The input y is not allowed to be None"));
  }
}

SpmdInfo SwiGLUGradInferSpmd(const DistMetaTensor& x,
                             const DistMetaTensor& y,
                             const DistMetaTensor& out_grad) {
  if (y) {
    return ElementwiseBinaryGradInferSpmd(x, y, out_grad);
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("The input y is not allowed to be None"));
  }
}

}  // namespace distributed
}  // namespace phi
