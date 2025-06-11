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

#include "paddle/phi/infermeta/spmd_rules/cummin.h"
#include "paddle/phi/infermeta/spmd_rules/topk.h"

namespace phi {
namespace distributed {

SpmdInfo CumminInferSpmd(const DistMetaTensor& x, int axis, DataType dtype) {
  return TopkInferSpmdBase(x, axis);
}

SpmdInfo CumminGradInferSpmd(const DistMetaTensor& x,
                             const DistMetaTensor& indices,
                             const DistMetaTensor& out_grad,
                             int axis,
                             DataType dtype) {
  return TopkGradInferSpmdBase(x, indices, out_grad, axis);
}

}  // namespace distributed
}  // namespace phi
