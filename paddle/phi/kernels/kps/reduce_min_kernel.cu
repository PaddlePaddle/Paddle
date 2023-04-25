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

#include "paddle/phi/kernels/reduce_min_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/reduce.h"

namespace phi {

template <typename T, typename Context>
void MinRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const IntArray& dims,
                  bool keep_dim,
                  bool reduce_all,
                  DenseTensor* out) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  auto out_dtype = x.dtype();
  phi::Reduce<T, kps::MinFunctor, kps::IdentityFunctor>(
      dev_ctx, x, reduce_all, dims.GetData(), keep_dim, out_dtype, out);
}

}  // namespace phi

#ifdef PADDLE_WITH_XPU_KP
PD_REGISTER_KERNEL(min_raw, KPS, ALL_LAYOUT, phi::MinRawKernel, float) {}
#else
PD_REGISTER_KERNEL(min_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::MinRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif
