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

#include "paddle/phi/kernels/squared_l2_norm_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
namespace phi {
template <typename T, typename Context>
void SquaredL2NormKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  std::vector<int> origin_reduce_dims;
  for (size_t i = 0; i < x.dims().size(); i++) {
    origin_reduce_dims.push_back(i);
  }
  phi::funcs::ReduceKernel<T, T, kps::AddFunctor, kps::SquareFunctor<T, T>>(
      dev_ctx, x, out, kps::SquareFunctor<T, T>(), origin_reduce_dims);
}

}  // namespace phi

PD_REGISTER_KERNEL(squared_l2_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::SquaredL2NormKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
