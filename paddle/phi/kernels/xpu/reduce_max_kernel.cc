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

#include "paddle/phi/kernels/reduce_max_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/xpu/reduce.h"

namespace phi {

template <typename T, typename Context>
void MaxKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const IntArray& dims,
               bool keep_dim,
               DenseTensor* out) {
  if (x.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  bool reduce_all = recompute_reduce_all(x, dims);
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto f = [](xpu::Context* ctx,
              const T* x,
              T* y,
              const std::vector<int64_t>& xdims,
              const std::vector<int64_t>& reduce_dims) {
#ifndef PADDLE_WITH_XPU_PLUGIN
    return xpu::reduce_max<XPUType>(ctx,
                                    reinterpret_cast<const XPUType*>(x),
                                    reinterpret_cast<XPUType*>(y),
                                    xdims,
                                    reduce_dims);
#else
    return xpu::plugin::fast_reduce_max<XPUType>(
        ctx,
        reinterpret_cast<const XPUType*>(x),
        reinterpret_cast<XPUType*>(y),
        std::vector<int>(xdims.begin(), xdims.end()),
        std::vector<int>(reduce_dims.begin(), reduce_dims.end()));
#endif
  };

  int r = XPUReduce<Context, T>(
      dev_ctx, x, dims.GetData(), keep_dim, reduce_all, out, f);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_max");
}

}  // namespace phi

PD_REGISTER_KERNEL(max,
                   XPU,
                   ALL_LAYOUT,
                   phi::MaxKernel,
                   int,
                   int64_t,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
