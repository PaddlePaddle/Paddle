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

#include "paddle/phi/kernels/squared_l2_norm_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"

namespace phi {
/**
 * x*y*2.0
 */
template <typename T>
struct DoubleMulFunctor {
  __device__ __forceinline__ T operator()(const T a, const T b) const {
    return b * a * static_cast<T>(2.0f);
  }
};

template <typename T, typename Context>
void SquaredL2NormGradKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& dout,
                             DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  PADDLE_ENFORCE_EQ(
      dout.numel(),
      1,
      common::errors::InvalidArgument(
          "Input(GRAD@Out) of SquaredL2NormGradOP should be a scalar."));
  std::vector<const DenseTensor*> ins{&x, &dout};
  std::vector<DenseTensor*> outs{dx};

  funcs::BroadcastKernel<T>(dev_ctx, ins, &outs, phi::DoubleMulFunctor<T>());
}
}  // namespace phi

PD_REGISTER_KERNEL(squared_l2_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SquaredL2NormGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
