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

#include "paddle/phi/infermeta/spmd_rules/full_like.h"
#include "paddle/phi/infermeta/spmd_rules/elementwise.h"

namespace phi::distributed {
SpmdInfo FullLikeInferSpmd(const DistMetaTensor& x,
                           const Scalar& y,
                           phi::DataType dtype) {
  TensorDistAttr out_dist_attr = x.dist_attr();
  out_dist_attr.clean_partial_status();
  return {{x.dist_attr()}, {out_dist_attr}};
}
}  // namespace phi::distributed
