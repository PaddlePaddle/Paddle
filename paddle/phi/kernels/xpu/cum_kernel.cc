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

#include "paddle/phi/kernels/cum_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/cumprod.h"

namespace phi {

template <typename T, typename Context>
void CumsumKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const Scalar& axis,
                  bool flatten,
                  bool exclusive,
                  bool reverse,
                  DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  dev_ctx.template Alloc<T>(out);

  if (x.numel() == 1) {
    int r = xpu::copy<XPUType>(dev_ctx.x_context(),
                               reinterpret_cast<const XPUType*>(x.data<T>()),
                               reinterpret_cast<XPUType*>(out->data<T>()),
                               x.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
    return;
  }

  // prepare for call xdnn api
  std::vector<int64_t> x_shape = vectorize<int64_t>(x.dims());
  int axis_as_int = axis.to<int>();

  if (flatten) {
    x_shape = {x.numel()};
    axis_as_int = 0;
  } else {
    auto out_dims = out->dims();
    PADDLE_ENFORCE_EQ(
        axis_as_int < out_dims.size() && axis_as_int >= (0 - out_dims.size()),
        true,
        common::errors::OutOfRange(
            "Attr(axis) is out of range, It's expected "
            "to be in range of [-%d, %d]. But received Attr(axis) = %d.",
            out_dims.size(),
            out_dims.size() - 1,
            axis_as_int));
    if (axis_as_int < 0) {
      axis_as_int += out_dims.size();
    }
  }

  // For float32 tensors with large scan axis (>4096), use block-based
  // decomposition to bound float32 accumulation error in xpu::cumsum.
  // The XDNN cumsum implementation can accumulate significant floating-point
  // error over long sequences because it lacks compensated summation.
  // By splitting into blocks of 4096 elements, the per-block cumsum
  // error is bounded (same as a small tensor), and the prefix propagation
  // across O(sqrt(N)) blocks also stays within tolerance.
  if constexpr (std::is_same_v<T, float>) {
    int64_t scan_size = x_shape[axis_as_int];
    constexpr int64_t BLOCK_SIZE = 1024;

    size_t outer_dim, mid_dim, inner_dim;
    auto out_dims = out->dims();
    GetCumprodDimInfo(out_dims, axis_as_int, &outer_dim, &mid_dim, &inner_dim);

    if (mid_dim > BLOCK_SIZE && inner_dim == 1) {
      int64_t num_rows = outer_dim;
      int64_t num_blocks = (mid_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
      int64_t padded_mid = num_blocks * BLOCK_SIZE;
      int64_t total_padded = num_rows * padded_mid;

      xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
      const XPUType* x_data =
          reinterpret_cast<const XPUType*>(x.data<T>());
      XPUType* out_data = reinterpret_cast<XPUType*>(out->data<T>());

      // Handle reverse: flip input first, compute forward, flip output back
      XPUType* flipped_x = nullptr;
      const XPUType* work_x = x_data;
      if (reverse) {
        flipped_x = RAII_GUARD.alloc_l3_or_gm<XPUType>(x.numel());
        std::vector<int64_t> flip_axes = {axis_as_int};
        int r = xpu::flip<XPUType>(
            dev_ctx.x_context(), x_data, flipped_x, x_shape, flip_axes);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "flip input");
        work_x = flipped_x;
      }

      // Allocate padded buffers
      XPUType* padded_x = RAII_GUARD.alloc_l3_or_gm<XPUType>(total_padded);
      XPUType* padded_y = RAII_GUARD.alloc_l3_or_gm<XPUType>(total_padded);
      XPUType* block_sums =
          RAII_GUARD.alloc_l3_or_gm<XPUType>(num_rows * num_blocks);
      XPUType* block_prefix =
          RAII_GUARD.alloc_l3_or_gm<XPUType>(num_rows * num_blocks);

      // Copy input rows with zero-padding to multiple of BLOCK_SIZE.
      // For inner_dim==1 (contiguous rows), copy each row from work_x
      // to padded_x with zero-fill for the padding region.
      int r;
      if (num_rows == 1) {
        // Single row: simple copy + zero pad tail
        r = xpu::copy<XPUType>(
            dev_ctx.x_context(), work_x, padded_x, mid_dim);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy padded_x");
        int64_t pad_len = total_padded - mid_dim;
        if (pad_len > 0) {
          r = xpu::constant<XPUType>(dev_ctx.x_context(),
                                     padded_x + mid_dim,
                                     pad_len,
                                     static_cast<XPUType>(0));
          PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant pad");
        }
      } else {
        // Multi-row: zero-init then copy each row's scan_size elements
        r = xpu::constant<XPUType>(dev_ctx.x_context(),
                                   padded_x,
                                   total_padded,
                                   static_cast<XPUType>(0));
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant zero padded_x");
        for (int64_t row = 0; row < num_rows; ++row) {
          r = xpu::copy<XPUType>(dev_ctx.x_context(),
                                 work_x + row * mid_dim,
                                 padded_x + row * padded_mid,
                                 mid_dim);
          PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy row");
        }
      }

      // Step 1: block-local cumsum
      // Shape: [num_rows * num_blocks, BLOCK_SIZE] — each block is contiguous
      std::vector<int64_t> block_shape = {num_rows * num_blocks, BLOCK_SIZE};
      r = xpu::cumsum<XPUType>(dev_ctx.x_context(),
                               padded_x,
                               padded_y,
                               block_shape,
                               /*reverse=*/false,
                               exclusive,
                               /*axis=*/1);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cumsum blocks");

      // Step 2: compute block sums from original padded input
      // reduce_sum along axis=1 (BLOCK_SIZE dim) -> [num_rows * num_blocks]
      std::vector<int64_t> reduce_dims = {1};
      r = xpu::reduce_sum<XPUType>(dev_ctx.x_context(),
                                   padded_x,
                                   block_sums,
                                   block_shape,
                                   reduce_dims);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum block sums");

      // Step 3: exclusive cumsum on block sums to get per-block prefix
      // Each row has num_blocks blocks; compute prefix independently per row
      // Shape for cumsum: [num_rows, num_blocks]
      std::vector<int64_t> bs_2d_shape = {num_rows, num_blocks};
      // exclusive cumsum: prefix[b] = sum of blocks 0..b-1
      r = xpu::cumsum<XPUType>(dev_ctx.x_context(),
                               block_sums,
                               block_prefix,
                               bs_2d_shape,
                               /*reverse=*/false,
                               /*exclusive=*/true,
                               /*axis=*/1);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cumsum block prefix");

      // Step 4: broadcast-add prefix to each block
      // prefix shape: [num_rows, num_blocks, 1]
      // padded_y shape: [num_rows * num_blocks, BLOCK_SIZE]
      std::vector<int64_t> prefix_bshape = {num_rows * num_blocks, 1};
      r = xpu::broadcast_add<XPUType>(dev_ctx.x_context(),
                                      block_prefix,
                                      padded_y,
                                      padded_y,
                                      prefix_bshape,
                                      block_shape);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add prefix");

      // Step 5: extract unpadded result back to output tensor
      XPUType* result = padded_y;
      if (reverse) {
        // Need to flip output back
        XPUType* flipped_out =
            RAII_GUARD.alloc_l3_or_gm<XPUType>(x.numel());
        // Copy unpadded result to flipped_out first
        if (num_rows == 1) {
          r = xpu::copy<XPUType>(
              dev_ctx.x_context(), padded_y, flipped_out, mid_dim);
        } else {
          for (int64_t row = 0; row < num_rows; ++row) {
            r = xpu::copy<XPUType>(dev_ctx.x_context(),
                                   padded_y + row * padded_mid,
                                   flipped_out + row * mid_dim,
                                   mid_dim);
            PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy unpadded row");
          }
        }
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy unpadded");
        // Flip back along the original axis
        std::vector<int64_t> flip_axes = {axis_as_int};
        r = xpu::flip<XPUType>(
            dev_ctx.x_context(), flipped_out, out_data, x_shape, flip_axes);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "flip output");
      } else {
        // Non-reverse: copy unpadded result to output
        if (num_rows == 1) {
          r = xpu::copy<XPUType>(
              dev_ctx.x_context(), padded_y, out_data, mid_dim);
        } else {
          for (int64_t row = 0; row < num_rows; ++row) {
            r = xpu::copy<XPUType>(dev_ctx.x_context(),
                                   padded_y + row * padded_mid,
                                   out_data + row * mid_dim,
                                   mid_dim);
            PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy unpadded row");
          }
        }
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy output");
      }
      return;
    }
  }

  // Standard path for non-float types or small tensors
  // template<typename T> DLL_EXPORT int cumsum(Context* xpu_ctx, const T* x, T*
  // y, const std::vector<int>& xshape, bool reverse, bool exclusive, int
  // axis);
  int r = xpu::cumsum<XPUType>(dev_ctx.x_context(),
                               reinterpret_cast<const XPUType*>(x.data<T>()),
                               reinterpret_cast<XPUType*>(out->data<T>()),
                               x_shape,
                               reverse,
                               exclusive,
                               axis_as_int);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "cumsum");
}

}  // namespace phi

PD_REGISTER_KERNEL(cumsum,
                   XPU,
                   ALL_LAYOUT,
                   phi::CumsumKernel,
                   float,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16) {}
