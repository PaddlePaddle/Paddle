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

#include "paddle/phi/kernels/tril_indices_kernel.h"

#include <algorithm>
#include <tuple>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
__device__ inline int resolve_root_int(int b, int cX4, int x, int32_t sign) {
  int bXb_cX4 = b * b - cX4;
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
__device__ inline void get_coordinate_in_tril_trapezoid(int f,
                                                        int x,
                                                        T* row,
                                                        T* col) {
  f <<= 1;  // all statements use 2f, so only calculate it once here.
  auto b = f - 1;
  auto cX4 = -(x << 3);  // 4 * c = 4 * (-2x) = -8x;
  *row = resolve_root_int<T>(b, cX4, x, 1);
  *col = x - ((f + *row - 1) * *row >> 1);
}

template <typename T>
__global__ void tril_indices_kernel(T* out_data,
                                    int row_offset,
                                    int m_first_row,
                                    int col,
                                    int trapezoid_size,
                                    int tril_size) {
  int linear_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (linear_index < tril_size) {
    T r, c;
    if (linear_index < trapezoid_size) {
      // the coordinate is within the top trapezoid
      get_coordinate_in_tril_trapezoid<T>(m_first_row, linear_index, &r, &c);
    } else {
      // the coordinate falls in the bottom rectangle
      auto surplus = linear_index - trapezoid_size;
      // add the height of trapezoid: m_last_row (col) - m_first_row + 1
      r = surplus / col + col - m_first_row + 1;
      c = surplus % col;
    }
    r += row_offset;

    out_data[linear_index] = r;
    out_data[linear_index + tril_size] = c;
  }
}

template <typename T, typename Context>
void TrilIndicesKernel(const Context& dev_ctx,
                       int rows,
                       int cols,
                       int offset,
                       DataType dtype,
                       DenseTensor* out) {
  T* out_data = dev_ctx.template Alloc<T>(out);
  auto out_dims = out->dims();
  int tril_size = out_dims[1];

  if (tril_size > 0) {
    auto m_first_row = offset > 0
                           ? std::min<int>(cols, 1 + offset)
                           : rows + offset > 0;  // the number of first row
    auto trapezoid_row_offset =
        std::max<int>(0, -offset);  // index of the first row who has number
    auto rectangle_row_offset = trapezoid_row_offset + cols - m_first_row +
                                1;  // the length of the right-up rest matrix
    int rectangle_size = 0;
    if (rectangle_row_offset < rows) {
      rectangle_size = (rows - rectangle_row_offset) * cols;
    }  // the rectangle part of lowertriangle matrix

    auto GetBlockGridSize = [&dev_ctx](int size) {
      const int block_size =
          std::min(size, static_cast<int>(dev_ctx.GetMaxThreadsPerBlock()));
      int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
      const int max_blocks =
          std::max(((max_threads - 1) / block_size + 1), static_cast<int>(1));
      const int grid_size =
          std::min(max_blocks, (size + block_size - 1) / block_size);
      return std::tuple<int, int>{grid_size, block_size};
    };

    std::tuple<int, int> block_grid_size = GetBlockGridSize(tril_size);

    tril_indices_kernel<T><<<std::get<0>(block_grid_size),
                             std::get<1>(block_grid_size),
                             0,
                             dev_ctx.stream()>>>(out_data,
                                                 trapezoid_row_offset,
                                                 m_first_row,
                                                 cols,
                                                 tril_size - rectangle_size,
                                                 tril_size);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    tril_indices, GPU, ALL_LAYOUT, phi::TrilIndicesKernel, int, int64_t) {}
