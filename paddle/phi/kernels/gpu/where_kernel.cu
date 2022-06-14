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

#include "paddle/phi/kernels/where_kernel.h"

#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"

namespace phi {

// Cond
template <typename T>
struct CondFunctor {
  inline HOSTDEVICE T operator()(const bool cond, const T x, const T y) const {
    return cond ? x : y;
  }
};

template <typename T, typename Context>
void WhereKernel(const Context& ctx,
                 const DenseTensor& condition,
                 const DenseTensor& x,
                 const DenseTensor& y,
                 DenseTensor* out) {
  std::vector<const DenseTensor*> ins = {&condition, &x, &y};
  std::vector<DenseTensor*> outs = {out};
  ctx.template Alloc<T>(out);

  CondFunctor<T> func;
  funcs::BroadcastKernel<ElementwiseType::kTernary, T, T>(
      ctx, ins, &outs, -1, func);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    where, GPU, ALL_LAYOUT, phi::WhereKernel, float, double, int, int64_t) {}
