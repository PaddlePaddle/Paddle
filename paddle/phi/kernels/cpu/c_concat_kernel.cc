// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/c_concat_kernel.h"

#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void CConcatKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   int nranks,
                   int rank,
                   bool use_model_parallel,
                   DenseTensor* out) {
  PADDLE_THROW(common::errors::Unavailable(
      "Do not support c_concat for cpu kernel now."));
}
}  // namespace phi

PD_REGISTER_KERNEL(c_concat_v2,
                   CPU,
                   ALL_LAYOUT,
                   ops::CConcatKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
