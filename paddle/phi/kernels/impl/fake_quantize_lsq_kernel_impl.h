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

#pragma once

#include "paddle/phi/kernels/fake_quantize_kernel.h"
#include "paddle/phi/kernels/funcs/fake_quantize_functor.h"

namespace phi {
// LSQ fakequat impl
template <typename T, typename Context>
void FakeQuantizeDequantizeLsqplusKernel(const Context &dev_ctx,
                                         const DenseTensor &x,
                                         const DenseTensor &alpha,
                                         const DenseTensor &beta,
                                         const DenseTensor &g_scale,
                                         int bit_length,
                                         bool is_sign,
                                         int round_type,
                                         DenseTensor *out) {
  dev_ctx.template Alloc<T>(out);
  int Qn = 0;
  int Qp = 255;
  if (is_sign) {
    Qn = -std::pow(2, bit_length - 1);
    Qp = std::pow(2, bit_length - 1) - 1;
  } else {
    Qn = 0;
    Qp = std::pow(2, bit_length) - 1;
  }

  phi::funcs::LsqplusFakeQuantDequantFunctor<Context, T>()(
      dev_ctx, x, alpha, beta, g_scale, Qn, Qp, round_type, out);
}
}  // namespace phi
