// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/margin_cross_entropy_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void MarginCrossEntropyKernel(const Context& dev_ctx,
                              const DenseTensor& logits,
                              const DenseTensor& labels,
                              bool return_softmax,
                              int ring_id,
                              int rank,
                              int nranks,
                              float margin1,
                              float margin2,
                              float margin3,
                              float scale,
                              DenseTensor* softmax,
                              DenseTensor* loss) {
  PADDLE_THROW(
      errors::Unavailable("Do not support margin_cross_entropy for cpu kernel "
                          "now."));
}

}  // namespace phi

PD_REGISTER_KERNEL(margin_cross_entropy,
                   CPU,
                   ALL_LAYOUT,
                   phi::MarginCrossEntropyKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
