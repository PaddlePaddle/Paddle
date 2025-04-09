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

#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"

namespace phi {
namespace distributed {

// (TODO) Support 2 kind of parallel:
// 1. sharding on batch axes (any axis that is not to be softmax normalized) of
// tensor.
// 2. sharding on normalized axis of tensor. (naive support by now, efficient
// support need to change the softmax kernel).
SpmdInfo SoftmaxInferSpmd(const DistMetaTensor& x, int axis);

SpmdInfo SoftmaxInferSpmdReverse(const DistMetaTensor& x,
                                 const DistMetaTensor& out,
                                 int axis);

SpmdInfo SoftmaxGradInferSpmd(const DistMetaTensor& out,
                              const DistMetaTensor& out_grad,
                              int axis);
}  // namespace distributed
}  // namespace phi
