/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

// The deformed product of operator iterative upgrade, there is no strict 2.0
// API corresponding to it! In 2.0 API paddle.nn.functional.cross_entropy,
// use_softmax has become an optional argument, which may be called
// CrossEntropyWithSoftmax more accurately, here we keep this kernel arguments
// same as original OpMaker, and if need a CrossEntropyKernel like
// paddle.nn.functional.cross_entropy, we can reuse this kernel
template <typename T, typename Context>
void CrossEntropyWithSoftmaxKernel(const Context& dev_ctx,
                                   const DenseTensor& logits,
                                   const DenseTensor& label,
                                   bool soft_label,
                                   bool use_softmax,
                                   bool numeric_stable_mode,
                                   int ignore_index,
                                   int axis,
                                   DenseTensor* softmax,
                                   DenseTensor* loss);

}  // namespace phi
