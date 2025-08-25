// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

#include "paddle/phi/kernels/funcs/conjugate_transpose.h"

#include "paddle/common/enforce.h"
#include "paddle/phi/backends/gpu/gpu_context.h"

namespace phi {
namespace funcs {

constexpr int TILE_DIM = 32;

template <typename T>
__global__ void ConjugateTransposeKernel(const T* __restrict__ input,
                                         T* __restrict__ output,
                                         int64_t m,
                                         int64_t n) {
  __shared__ T tile[TILE_DIM][TILE_DIM + 1];

  const int64_t matrix_size_in = m * n;
  const int64_t matrix_size_out = n * m;

  for (int64_t current_batch_id = blockIdx.z; current_batch_id < batch_size;
       current_batch_id += gridDim.z) {
    const T* current_input = input + current_batch_id * matrix_size_in;
    T* current_output = output + current_batch_id * matrix_size_out;

    int64_t x_in = blockIdx.x * TILE_DIM + threadIdx.x;
    int64_t y_in = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x_in < n && y_in < m) {
      tile[threadIdx.y][threadIdx.x] = current_input[y_in * n + x_in];
    }

    __syncthreads();

    int64_t x_out = blockIdx.y * TILE_DIM + threadIdx.x;
    int64_t y_out = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x_out < m && y_out < n) {
      T val = tile[threadIdx.x][threadIdx.y];
      if constexpr (std::is_same_v<T, phi::dtype::complex<float>> ||
                    std::is_same_v<T, phi::dtype::complex<double>>) {
        current_output[y_out * m + x_out] = phi::dtype::conj(val);
      } else {
        current_output[y_out * m + x_out] = val;
      }
    }
  }
}

template <typename T, typename Context>
void ConjugateTransposeFunctor<T, Context>::operator()(const Context& dev_ctx,
                                                       const DenseTensor& input,
                                                       DenseTensor* output) {
  const auto* d_input = input.data<T>();
  auto* d_output = dev_ctx.template Alloc<T>(output);

  const auto& dims = input.dims();
  const int rank = dims.size();
  PADDLE_ENFORCE_GE(rank,
                    2,
                    common::errors::InvalidArgument(
                        "The rank of input tensor must be at least 2."));

  const int64_t n = dims[rank - 1];  // width
  const int64_t m = dims[rank - 2];  // height
  const int64_t matrix_size = m * n;
  const int64_t batch_size = input.numel() / matrix_size;

  if (batch_size == 0) {
    return;
  }

  dim3 threads(TILE_DIM, TILE_DIM, 1);
  dim3 blocks;
  blocks.x = (n + threads.x - 1) / threads.x;
  blocks.y = (m + threads.y - 1) / threads.y;

  constexpr int64_t max_z_dim = 4096;
  blocks.z = std::min(batch_size, max_z_dim);

  ConjugateTransposeKernel<T>
      <<<blocks, threads, 0, dev_ctx.stream()>>>(d_input, d_output, m, n);
}

template class ConjugateTransposeFunctor<float, GPUContext>;
template class ConjugateTransposeFunctor<double, GPUContext>;
template class ConjugateTransposeFunctor<phi::dtype::complex<float>,
                                         GPUContext>;
template class ConjugateTransposeFunctor<phi::dtype::complex<double>,
                                         GPUContext>;

}  // namespace funcs
}  // namespace phi
