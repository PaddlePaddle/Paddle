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

#include "paddle/phi/kernels/cum_maxmin_kernel.h"

#include <numeric>

#include "paddle/common/hostdevice.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <
    typename T1,
    typename T2,
    typename BinaryOperation,
    typename std::enable_if<std::is_floating_point<T1>::value, int>::type = 0>
__device__ void binary_op_update(const T1 lhs,
                                 T1* rhs,
                                 const T2 lhs_idx,
                                 T2* rhs_idx,
                                 BinaryOperation binary_op) {
  if (!isnan(*rhs) && (isnan(lhs) || !binary_op(*rhs, lhs))) {
    *rhs = lhs;
    *rhs_idx = lhs_idx;
  }
}

template <typename T1,
          typename T2,
          typename BinaryOperation,
          typename std::enable_if<std::is_integral<T1>::value, int>::type = 0>
__device__ void binary_op_update(const T1 lhs,
                                 T1* rhs,
                                 const T2 lhs_idx,
                                 T2* rhs_idx,
                                 BinaryOperation binary_op) {
  if (!binary_op(*rhs, lhs)) {
    *rhs = lhs;
    *rhs_idx = lhs_idx;
  }
}

template <
    typename T1,
    typename T2,
    typename BinaryOperation,
    typename std::enable_if<std::is_floating_point<T1>::value, int>::type = 0>
__device__ void binary_op_update_v(const T1 lhs,
                                   T1* rhs,
                                   const T2 lhs_idx,
                                   T2* rhs_idx,
                                   BinaryOperation binary_op) {
  if (isnan(lhs) || (!isnan(*rhs) && binary_op(lhs, *rhs))) {
    *rhs = lhs;
    *rhs_idx = lhs_idx;
  }
}

template <typename T1,
          typename T2,
          typename BinaryOperation,
          typename std::enable_if<std::is_integral<T1>::value, int>::type = 0>
__device__ void binary_op_update_v(const T1 lhs,
                                   T1* rhs,
                                   const T2 lhs_idx,
                                   T2* rhs_idx,
                                   BinaryOperation binary_op) {
  if (binary_op(lhs, *rhs)) {
    *rhs = lhs;
    *rhs_idx = lhs_idx;
  }
}

template <typename T1,
          typename T2,
          int num_threads_x,
          int num_threads_y,
          class BinaryFunction>
__global__ void KernelScanInnerWithIndices(const T1* x_data,
                                           T1* values_data,
                                           T2* indices_data,
                                           int num_rows,
                                           int row_size,
                                           T1 init,
                                           BinaryFunction binary_op) {
  __shared__ T1 vbuf[num_threads_y][2 * num_threads_x];
  __shared__ T2 ibuf[num_threads_y][2 * num_threads_x];
  T1* row_buf = vbuf[threadIdx.y];
  T2* row_idx_buf = ibuf[threadIdx.y];

  for (int block_row = blockIdx.x * blockDim.y; block_row < num_rows;
       block_row += blockDim.y * gridDim.x) {
    int row = block_row + threadIdx.y;
    const T1* row_self = x_data + row * row_size;
    T1* row_values = values_data + row * row_size;
    T2* row_indices = indices_data + row * row_size;
    T1 block_total = init;
    T2 block_idx_final = 0;
    // Perform scan on one block at a time, keeping track of the total value of
    // all blocks processed so far.
    for (int block_col = 0; block_col < row_size;
         block_col += 2 * num_threads_x) {
      // Load data into shared memory (two values per thread).
      int col1 = block_col + threadIdx.x;
      int col2 = block_col + num_threads_x + threadIdx.x;
      if (row < num_rows) {
        if (col1 < row_size) {
          row_buf[threadIdx.x] = *reinterpret_cast<const T1*>(&row_self[col1]);
          row_idx_buf[threadIdx.x] = col1;
        } else {
          row_buf[threadIdx.x] = init;
        }

        if (col2 < row_size) {
          row_buf[num_threads_x + threadIdx.x] =
              *reinterpret_cast<const T1*>(&row_self[col2]);
          row_idx_buf[num_threads_x + threadIdx.x] = col2;
        } else {
          row_buf[num_threads_x + threadIdx.x] = init;
        }

        if (threadIdx.x == 0) {
          binary_op_update(block_total,
                           &row_buf[0],
                           block_idx_final,
                           &row_idx_buf[0],
                           binary_op);
        }
      }
      __syncthreads();

      // Parallel reduction (up-sweep).
      for (int s = num_threads_x, d = 1; s >= 1; s >>= 1, d <<= 1) {
        if (row < num_rows && threadIdx.x < s) {
          int offset = (2 * threadIdx.x + 1) * d - 1;
          binary_op_update(row_buf[offset],
                           &row_buf[offset + d],
                           row_idx_buf[offset],
                           &row_idx_buf[offset + d],
                           binary_op);
        }
        __syncthreads();
      }

      // Down-sweep.
      for (int s = 2, d = num_threads_x / 2; d >= 1; s <<= 1, d >>= 1) {
        if (row < num_rows && threadIdx.x < s - 1) {
          int offset = 2 * (threadIdx.x + 1) * d - 1;
          binary_op_update(row_buf[offset],
                           &row_buf[offset + d],
                           row_idx_buf[offset],
                           &row_idx_buf[offset + d],
                           binary_op);
        }
        __syncthreads();
      }

      // Write back to output.
      if (row < num_rows) {
        if (col1 < row_size) {
          row_values[col1] = row_buf[threadIdx.x];
          row_indices[col1] = row_idx_buf[threadIdx.x];
        }
        if (col2 < row_size) {
          row_values[col2] = row_buf[num_threads_x + threadIdx.x];
          row_indices[col2] = row_idx_buf[num_threads_x + threadIdx.x];
        }
      }
      block_total = row_buf[2 * num_threads_x - 1];
      block_idx_final = row_idx_buf[2 * num_threads_x - 1];
      __syncthreads();
    }
  }
}

template <typename T1, typename T2, class BinaryFunction>
__global__ void KernelScanOuterWithIndices(const T1* x_data,
                                           T1* values_data,
                                           T2* indices_data,
                                           const uint32_t num_orows,
                                           const uint32_t num_irows,
                                           const uint32_t row_size,
                                           T1 init,
                                           BinaryFunction binary_op) {
  for (uint32_t orow = blockIdx.x; orow < num_orows; orow += gridDim.x) {
    for (uint32_t irow = blockIdx.y * blockDim.x + threadIdx.x;
         irow < num_irows;
         irow += gridDim.y * blockDim.x) {
      const T1* x = x_data + orow * row_size * num_irows + irow;
      T1* values = values_data + orow * row_size * num_irows + irow;
      T2* indices = indices_data + orow * row_size * num_irows + irow;
      T1 out = init;
      T2 out_idx = 0;

      for (T2 col = 0; col < row_size; ++col) {
        const auto val = *reinterpret_cast<const T1*>(x);
        binary_op_update_v(val, &out, col, &out_idx, binary_op);
        *values = out;
        *indices = out_idx;
        x += num_irows;
        values += num_irows;
        indices += num_irows;
      }
    }
  }
}

template <typename T1, typename T2, typename BinaryFunction, typename Context>
void ScanWithIndicesKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           int axis,
                           T1 init,
                           DenseTensor* out,
                           DenseTensor* indices) {
  dev_ctx.template Alloc<T1>(out);
  dev_ctx.template Alloc<T2>(indices);
  // For 0D Tensor
  if (out->numel() == 1) {
    auto raw_dims = out->dims();
    phi::Copy<Context>(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    phi::funcs::SetConstant<Context, T2> set_zero;
    set_zero(dev_ctx, indices, static_cast<T2>(0.0));
    out->Resize(raw_dims);
    indices->Resize(raw_dims);
    return;
  }

  BinaryFunction op;
  auto out_dims = out->dims();
  auto size = x.numel();

  PADDLE_ENFORCE_EQ(
      axis < out_dims.size() && axis >= (0 - out_dims.size()),
      true,
      common::errors::OutOfRange(
          "Attr(axis) is out of range, It's expected "
          "to be in range of [-%d, %d]. But received Attr(axis) = %d.",
          out_dims.size(),
          out_dims.size() - 1,
          axis));
  if (axis < 0) {
    axis += out_dims.size();
  }

  const T1* x_data = x.data<T1>();
  T1* values_data = out->data<T1>();
  T2* indices_data = indices->data<T2>();
  if (axis == out_dims.size() - 1) {
    int ndim = x.dims().size();
    int row_size = x.dims()[ndim - 1];
    int num_rows = x.numel() / row_size;

    dim3 threads(16, 32);
    dim3 grid(std::min(
        dev_ctx.GetCUDAMaxGridDimSize()[0],
        static_cast<unsigned int>(std::ceil(static_cast<float>(num_rows) /
                                            static_cast<float>(threads.y)))));

    KernelScanInnerWithIndices<T1, T2, 16, 32>
        <<<grid, threads, 0, dev_ctx.stream()>>>(
            x_data, values_data, indices_data, num_rows, row_size, init, op);
  } else {
    int64_t row_size = x.dims()[axis];
    auto sizes = common::vectorize(x.dims());

    const int64_t num_orows =
        std::accumulate(sizes.begin(),
                        sizes.begin() + axis,
                        int64_t(1),
                        [](int64_t& a, int64_t& b) { return a * b; });
    const int64_t num_irows =
        std::accumulate(sizes.begin() + axis + 1,
                        sizes.end(),
                        int64_t(1),
                        [](int64_t& a, int64_t& b) { return a * b; });

    dim3 threads(std::min(512, static_cast<int>(num_irows)));
    int64_t maxGridDim = dev_ctx.GetCUDAMaxGridDimSize()[1];
    dim3 grid(std::min(maxGridDim, num_orows),
              std::min(maxGridDim,
                       static_cast<int64_t>(
                           std::ceil(static_cast<double>(num_irows) /
                                     static_cast<double>(threads.x)))));

    KernelScanOuterWithIndices<T1, T2>
        <<<grid, threads, 0, dev_ctx.stream()>>>(x_data,
                                                 values_data,
                                                 indices_data,
                                                 num_orows,
                                                 num_irows,
                                                 row_size,
                                                 init,
                                                 op);
  }
}

template <typename T, typename Context>
void CummaxKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  int axis,
                  DataType dtype,
                  DenseTensor* out,
                  DenseTensor* indices) {
  T init = std::is_floating_point<T>::value
               ? (-1 * std::numeric_limits<T>::infinity())
               : std::numeric_limits<T>::lowest();
  if (dtype == DataType::INT32) {
    ScanWithIndicesKernel<T, int32_t, std::greater_equal<T>, Context>(
        dev_ctx, x, axis, init, out, indices);
  } else if (dtype == DataType::INT64) {
    ScanWithIndicesKernel<T, int64_t, std::greater_equal<T>, Context>(
        dev_ctx, x, axis, init, out, indices);
  }
}

template <typename T, typename Context>
void CumminKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  int axis,
                  DataType dtype,
                  DenseTensor* out,
                  DenseTensor* indices) {
  T init = std::is_floating_point<T>::value ? std::numeric_limits<T>::infinity()
                                            : std::numeric_limits<T>::max();
  if (dtype == DataType::INT32) {
    ScanWithIndicesKernel<T, int32_t, std::less_equal<T>, Context>(
        dev_ctx, x, axis, init, out, indices);
  } else if (dtype == DataType::INT64) {
    ScanWithIndicesKernel<T, int64_t, std::less_equal<T>, Context>(
        dev_ctx, x, axis, init, out, indices);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(cummax,
                   GPU,
                   ALL_LAYOUT,
                   phi::CummaxKernel,
                   float,
                   double,
                   int32_t,
                   int64_t) {}

PD_REGISTER_KERNEL(cummin,
                   GPU,
                   ALL_LAYOUT,
                   phi::CumminKernel,
                   float,
                   double,
                   int32_t,
                   int64_t) {}
