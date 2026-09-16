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

#include "paddle/phi/kernels/unpool_kernel.h"

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

// XPU implementation of max_unpool2d.
//
// Index format difference between XPU and CPU/GPU:
//   XPU's max_pool2d produces LOCAL indices: the flat position of the maximum
//   element within the corresponding spatial window [0, kH * kW).
//
//   CPU/GPU's max_pool2d produces GLOBAL flat indices: the flat position of
//   the maximum element within the entire OUTPUT tensor spatial extent
//   [0, out_H * out_W).
//
//   The unpool operation requires global flat indices. This kernel converts
//   XPU local indices to global flat indices before scattering.
//
// Conversion formula:
//   For pool output element at position (wr, wc) in pool output grid:
//     local_r = local_idx / kW
//     local_c = local_idx % kW
//     global_r = wr * stride_H + local_r
//     global_c = wc * stride_W + local_c
//     global_flat = global_r * out_W + global_c
//
// Implementation:
//   1. Copy x and indices from XPU to CPU.
//   2. Convert local indices to global flat indices.
//   3. Zero-fill a CPU output buffer.
//   4. Scatter: out[b, c, global_flat] = x[b, c, i].
//   5. Copy result back to XPU.
template <typename T, typename Context>
void UnpoolKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& indices,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings UNUSED,
                  const IntArray& output_size UNUSED,
                  const std::string& data_format UNUSED,
                  DenseTensor* out) {
  // Allocate output.
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  const int64_t n = x.dims()[0];
  const int64_t c = x.dims()[1];
  const int64_t xh = x.dims()[2];  // pool output height = unpool input height
  const int64_t xw = x.dims()[3];  // pool output width  = unpool input width
  const int64_t yh =
      out->dims()[2];  // unpool output height = pool input height
  const int64_t yw = out->dims()[3];  // unpool output width  = pool input width

  const int64_t in_spatial = xh * xw;
  const int64_t out_spatial = yh * yw;

  // ksize and strides are from the pooling that generated the indices.
  const int64_t kH = ksize[0];
  const int64_t kW = ksize[1];
  const int64_t sH = strides[0];
  const int64_t sW = strides[1];

  // Step 1: cast int64 indices to int32 if needed (XPU-side).
  const DenseTensor* indices_to_copy = &indices;
  DenseTensor indices_int32_buf;
  if (indices.dtype() == phi::DataType::INT64) {
    indices_int32_buf.Resize(indices.dims());
    dev_ctx.template Alloc<int>(&indices_int32_buf);
    int r = xpu::cast<int64_t, int>(dev_ctx.x_context(),
                                    indices.data<int64_t>(),
                                    indices_int32_buf.data<int>(),
                                    indices.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast int64->int32 for unpool indices");
    indices_to_copy = &indices_int32_buf;
  }

  // Step 2: copy x and int32-indices to CPU.
  phi::CPUPlace cpu_place;
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* cpu_ctx = static_cast<phi::CPUContext*>(pool.Get(cpu_place));

  DenseTensor x_cpu, idx_cpu, out_cpu;
  phi::Copy<Context>(dev_ctx, x, cpu_place, /*blocking=*/true, &x_cpu);
  phi::Copy<Context>(
      dev_ctx, *indices_to_copy, cpu_place, /*blocking=*/true, &idx_cpu);

  // Step 3: scatter on CPU, converting XPU local indices to global flat
  // indices.
  out_cpu.Resize(out->dims());
  cpu_ctx->template Alloc<T>(&out_cpu);
  T* out_ptr = out_cpu.data<T>();
  std::fill(out_ptr, out_ptr + out_cpu.numel(), static_cast<T>(0));

  const T* x_ptr = x_cpu.data<T>();
  const int* idx_ptr = idx_cpu.data<int>();

  for (int64_t b = 0; b < n; ++b) {
    for (int64_t ch = 0; ch < c; ++ch) {
      int64_t base = (b * c + ch);
      const T* x_slice = x_ptr + base * in_spatial;
      const int* idx_slice = idx_ptr + base * in_spatial;
      T* out_slice = out_ptr + base * out_spatial;

      // i is the linear index in the pool output (= unpool input)
      // wr, wc: row and column position in the pool output grid
      for (int64_t i = 0; i < in_spatial; ++i) {
        // XPU local index: flat position within the kH x kW window
        int64_t local_idx = static_cast<int64_t>(idx_slice[i]);

        // Convert to row, col within the window
        int64_t local_r = local_idx / kW;
        int64_t local_c = local_idx % kW;

        // Pool output position
        int64_t wr = i / xw;
        int64_t wc = i % xw;

        // Global position in pool input (= unpool output)
        int64_t global_r = wr * sH + local_r;
        int64_t global_c = wc * sW + local_c;
        int64_t global_flat = global_r * yw + global_c;

        PADDLE_ENFORCE_LT(
            global_flat,
            out_spatial,
            common::errors::InvalidArgument(
                "Unpool global index %lld is out of range [0, %lld). "
                "local_idx=%lld, wr=%lld, wc=%lld, global_r=%lld, "
                "global_c=%lld, "
                "kH=%lld, kW=%lld, sH=%lld, sW=%lld, yh=%lld, yw=%lld.",
                global_flat,
                out_spatial,
                local_idx,
                wr,
                wc,
                global_r,
                global_c,
                kH,
                kW,
                sH,
                sW,
                yh,
                yw));
        out_slice[global_flat] = x_slice[i];
      }
    }
  }

  // Step 4: copy result back to XPU.
  phi::Copy<Context>(dev_ctx,
                     out_cpu,
                     dev_ctx.GetPlace(),
                     /*blocking=*/true,
                     out);
}

}  // namespace phi

// Register float32.
// XPU max_pool2d_with_index produces int32 local indices (position within
// the kH x kW window). The kernel body converts these to global flat indices
// matching the format expected by CPU/GPU unpool. When a caller passes int64
// indices, the dispatch layer (SetDataType INT32) will attempt to cast first;
// the kernel also contains an explicit fallback cast.
PD_REGISTER_KERNEL(unpool, XPU, ALL_LAYOUT, phi::UnpoolKernel, float) {
  kernel->InputAt(1).SetDataType(phi::DataType::INT32);
}
