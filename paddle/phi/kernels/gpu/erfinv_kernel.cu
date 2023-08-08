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

#include "paddle/phi/kernels/erfinv_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"

namespace phi {

template <typename T>
struct ErfinvFunctor {
  HOSTDEVICE inline T operator()(const T x) const { return erfinv(x); }
};
template <>
struct ErfinvFunctor<float16> {
  HOSTDEVICE inline float16 operator()(const float16 x) const {
    auto x_ = static_cast<float>(x);
    return static_cast<float16>(erfinv(x_));
  }
};

template <>
struct ErfinvFunctor<bfloat16> {
  HOSTDEVICE inline bfloat16 operator()(const bfloat16 x) const {
    auto x_ = static_cast<float>(x);
    return static_cast<bfloat16>(erfinv(x_));
  }
};
template <typename T, typename Context>
void ErfinvKernel(const Context& ctx, const DenseTensor& x, DenseTensor* out) {
  ctx.template Alloc<T>(out);
  std::vector<const DenseTensor*> ins = {&x};
  std::vector<DenseTensor*> outs = {out};
  phi::funcs::ElementwiseKernel<T>(ctx, ins, &outs, ErfinvFunctor<T>());
}

}  // namespace phi

PD_REGISTER_KERNEL(erfinv,
                   GPU,
                   ALL_LAYOUT,
                   phi::ErfinvKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
