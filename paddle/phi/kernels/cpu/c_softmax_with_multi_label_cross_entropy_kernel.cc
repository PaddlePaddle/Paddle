// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void CSoftmaxWithMultiLabelCrossEntropyKernel(const Context &dev_ctx UNUSED,
                                              const DenseTensor &logits UNUSED,
                                              const DenseTensor &label UNUSED,
                                              const DenseTensor &smooth_weight
                                                  UNUSED,
                                              int64_t ignore_index UNUSED,
                                              bool sum_multi_label_loss UNUSED,
                                              int rank UNUSED,
                                              int nranks UNUSED,
                                              DenseTensor *softmax UNUSED,
                                              DenseTensor *loss UNUSED) {
  PADDLE_THROW(common::errors::Unavailable(
      "Do not support c_softmax_with_multi_label_cross_entropy for cpu kernel "
      "now."));
}

}  // namespace phi
PD_REGISTER_KERNEL(c_softmax_with_multi_label_cross_entropy,
                   CPU,
                   ALL_LAYOUT,
                   phi::CSoftmaxWithMultiLabelCrossEntropyKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
