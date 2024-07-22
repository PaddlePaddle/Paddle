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

#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/fake_quantize_grad_kernel.h"
#include "paddle/phi/kernels/funcs/fake_quantize_grad_functor.h"

namespace phi {

template <typename T, typename Context>
void FakeQuantizeDequantizeLSQGradKernel(const Context& dev_ctx,
                                         const DenseTensor& x,
                                         const DenseTensor& scale,
                                         const DenseTensor& out_grad,
                                         const float lsq_factor,
                                         const int bit_length,
                                         const int round_type,
                                         DenseTensor* x_grad,
                                         DenseTensor* scale_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  dev_ctx.template Alloc<T>(scale_grad);

  int bin_cnt = std::pow(2, bit_length - 1) - 1;

  phi::funcs::FakeQuantizeDequantizeGradLSQFunctor<Context, T>()(
    dev_ctx, x, scale, out_grad, lsq_factor, bin_cnt, round_type, x_grad, scale_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(fake_quantize_dequantize_lsq_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FakeQuantizeDequantizeLSQGradKernel,
                   float,
                   phi::dtype::float16,
                   double) {}
