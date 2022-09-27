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

#include "paddle/phi/kernels/triu_indices_kernel.h"

#include <algorithm>
#include <tuple>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
__device__ inline int resolve_root_int(int b, int cX4, int x, int32_t sign) {
  int64_t bXb_cX4 = b * b - cX4;
  double sr = ::sqrt(static_cast<double>(bXb_cX4));
  T res = ::__double2ll_rd((-b + sign * sr) / 2);
  if (bXb_cX4 != static_cast<int>(sr * sr)) {
    int llsr = ::__double2ll_rd(sr);
    int diff = ::__double2ll_ru(
        ::sqrt(::fabs(static_cast<double>(bXb_cX4 - llsr * llsr))));
    auto l = res > diff ? res - diff : 0;
    auto r = res + diff + 1;
    x <<= 1;
    while (l + 1 < r) {
      auto m = (l + r) >> 1;
      if (sign * (b + m) * m > x) {
        r = m;
      } else {
        l = m;
      }
    }
    res = l;
  }
  return res;
}

template <typename T>
__device__ inline void get_coordinate_in_triu_trapezoid(int f,
                                                        int x,
                                                        T* row,
                                                        T* col) {
  f <<= 1;  // all statements use 2f, so only calculate it once here.
  auto b = -1 - f;
  auto cX4 = x << 3;  // 4 * c = 4 * (2x) = 8x;
  *row = resolve_root_int<T>(b, cX4, x, -1);
  *col = (x - (((f - *row + 1) * *row) >> 1)) + *row;
}

template <typename T>
__global__ void triu_indices_kernel(T* out_data,
                                    int col_offset,
                                    int m_first_row,
                                    int col,
                                    int rectangle_size,
                                    int triu_size) {
  int linear_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (linear_index < triu_size) {
    T r, c;
    if (linear_index < rectangle_size) {
      // the coordinate is within the top rectangle
      r = linear_index / col;
      c = linear_index % col;
    } else {
      // the coordinate falls in the bottom trapezoid
      get_coordinate_in_triu_trapezoid<T>(
          m_first_row, linear_index - rectangle_size, &r, &c);
      r += rectangle_size / col;
    }

    c += col_offset;
    out_data[linear_index] = r;
    out_data[linear_index + triu_size] = c;
  }
}

template <typename T, typename Context>
void TriuIndicesKernel(const Context& dev_ctx,
                       int row,
                       int col,
                       int offset,
                       DataType dtype,
                       DenseTensor* out) {
  T* out_data = dev_ctx.template Alloc<T>(out);
  auto out_dims = out->dims();
  int triu_size = out_dims[1];
  //  auto tensor = empty_cuda({2, triu_size}, dtype_opt, layout_opt,
  //  device_opt, pin_memory_opt);

  if (triu_size > 0) {
    // # of triu elements in the first row
    auto m_first_row = offset > 0 ? std::max<int>(col - offset, 0)
                                  :  // upper bounded by col
                           col;

    // size of the top rectangle
    int rectangle_size = 0;
    if (offset < 0) {
      rectangle_size = std::min<int>(row, -offset) * col;
    }

    //  using gpu_launch_config to get grid_size and block_size
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, triu_size);

    triu_indices_kernel<T><<<config.block_per_grid.x,
                             config.thread_per_block.x,
                             0,
                             dev_ctx.stream()>>>(out_data,
                                                 std::max<int>(0, offset),
                                                 m_first_row,
                                                 col,
                                                 rectangle_size,
                                                 triu_size);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    triu_indices, GPU, ALL_LAYOUT, phi::TriuIndicesKernel, int, int64_t) {}
