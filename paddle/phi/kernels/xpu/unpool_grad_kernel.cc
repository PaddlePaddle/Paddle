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

#include "paddle/phi/kernels/unpool_grad_kernel.h"

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

// XPU implementation of max_unpool2d backward (gradient).
//
// The forward kernel converts XPU local indices (position within each
// kH x kW window) to global flat indices before scattering.  The backward
// kernel must apply the same conversion and then gather:
//
//   x_grad[b, c, i] = out_grad[b, c, global_flat(i)]
//
// where global_flat is computed identically to the forward pass.
template <typename T, typename Context>
void UnpoolGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& indices,
                      const DenseTensor& out UNUSED,
                      const DenseTensor& out_grad,
                      const std::vector<int>& ksize,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings UNUSED,
                      const IntArray& output_size UNUSED,
                      const std::string& data_format UNUSED,
                      DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  if (x_grad->numel() == 0) {
    return;
  }

  const int64_t n = x.dims()[0];
  const int64_t c = x.dims()[1];
  const int64_t xh = x.dims()[2];
  const int64_t xw = x.dims()[3];
  const int64_t yh = out_grad.dims()[2];
  const int64_t yw = out_grad.dims()[3];

  const int64_t in_spatial = xh * xw;
  const int64_t out_spatial = yh * yw;

  const int64_t kW = ksize[1];
  const int64_t sH = strides[0];
  const int64_t sW = strides[1];

  // Step 1: cast int64 indices to int32 if needed.
  const DenseTensor* indices_to_copy = &indices;
  DenseTensor indices_int32_buf;
  if (indices.dtype() == phi::DataType::INT64) {
    indices_int32_buf.Resize(indices.dims());
    dev_ctx.template Alloc<int>(&indices_int32_buf);
    int r = xpu::cast<int64_t, int>(dev_ctx.x_context(),
                                    indices.data<int64_t>(),
                                    indices_int32_buf.data<int>(),
                                    indices.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast int64->int32 for unpool_grad indices");
    indices_to_copy = &indices_int32_buf;
  }

  // Step 2: copy tensors to CPU.
  phi::CPUPlace cpu_place;
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* cpu_ctx = static_cast<phi::CPUContext*>(pool.Get(cpu_place));

  DenseTensor idx_cpu, out_grad_cpu, x_grad_cpu;
  phi::Copy<Context>(
      dev_ctx, *indices_to_copy, cpu_place, /*blocking=*/true, &idx_cpu);
  phi::Copy<Context>(
      dev_ctx, out_grad, cpu_place, /*blocking=*/true, &out_grad_cpu);

  // Step 3: gather on CPU with local→global index conversion.
  x_grad_cpu.Resize(x_grad->dims());
  cpu_ctx->template Alloc<T>(&x_grad_cpu);
  T* x_grad_ptr = x_grad_cpu.data<T>();
  std::fill(x_grad_ptr, x_grad_ptr + x_grad_cpu.numel(), static_cast<T>(0));

  const int* idx_ptr = idx_cpu.data<int>();
  const T* out_grad_ptr = out_grad_cpu.data<T>();

  for (int64_t b = 0; b < n; ++b) {
    for (int64_t ch = 0; ch < c; ++ch) {
      int64_t base = (b * c + ch);
      T* xg_slice = x_grad_ptr + base * in_spatial;
      const int* idx_slice = idx_ptr + base * in_spatial;
      const T* og_slice = out_grad_ptr + base * out_spatial;

      for (int64_t i = 0; i < in_spatial; ++i) {
        int64_t local_idx = static_cast<int64_t>(idx_slice[i]);
        int64_t local_r = local_idx / kW;
        int64_t local_c = local_idx % kW;
        int64_t wr = i / xw;
        int64_t wc = i % xw;
        int64_t global_r = wr * sH + local_r;
        int64_t global_c = wc * sW + local_c;
        int64_t global_flat = global_r * yw + global_c;

        PADDLE_ENFORCE_LT(
            global_flat,
            out_spatial,
            common::errors::InvalidArgument(
                "Unpool_grad global index %lld is out of range [0, %lld).",
                global_flat,
                out_spatial));
        xg_slice[i] = og_slice[global_flat];
      }
    }
  }

  // Step 4: copy result back to XPU.
  phi::Copy<Context>(dev_ctx,
                     x_grad_cpu,
                     dev_ctx.GetPlace(),
                     /*blocking=*/true,
                     x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(unpool_grad, XPU, ALL_LAYOUT, phi::UnpoolGradKernel, float) {
  kernel->InputAt(1).SetDataType(phi::DataType::INT32);
}
