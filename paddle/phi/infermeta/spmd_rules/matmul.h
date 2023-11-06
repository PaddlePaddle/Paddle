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

#pragma once

#include <vector>

#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"

namespace phi {
namespace distributed {

SpmdInfo MatmulInferSpmd(const DistMetaTensor& x,
                         const DistMetaTensor& y,
                         bool trans_x,
                         bool trans_y);

SpmdInfo MatmulInferSpmdReverse(const DistMetaTensor& x,
                                const DistMetaTensor& y,
                                const DistMetaTensor& out,
                                bool trans_x,
                                bool trans_y);

// TODO(chenweihang): This rule is currently incomplete, and we should
// polish this rule after fixed Matmul infermeta's existing bug
SpmdInfo MatmulGradInferSpmd(const DistMetaTensor& x,
                             const DistMetaTensor& y,
                             const DistMetaTensor& out_grad,
                             bool trans_x,
                             bool trans_y);

}  // namespace distributed
}  // namespace phi
