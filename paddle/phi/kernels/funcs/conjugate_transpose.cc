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
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {
namespace funcs {

constexpr int CPU_TILE_SIZE = 16;

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

  const int64_t n = dims[rank - 1];
  const int64_t m = dims[rank - 2];
  const int64_t matrix_size = m * n;
  const int64_t batch_size = input.numel() / matrix_size;

  if (batch_size == 0) {
    return;
  }

  auto transpose_task = [&](int64_t start_batch, int64_t end_batch) {
    for (int64_t b = start_batch; b < end_batch; ++b) {
      const T* src_matrix = d_input + b * matrix_size;
      T* dst_matrix = d_output + b * matrix_size;

      // Tiled transpose for better cache performance.
      for (int64_t i = 0; i < m; i += CPU_TILE_SIZE) {
        for (int64_t j = 0; j < n; j += CPU_TILE_SIZE) {
          // Transpose the tile.
          for (int64_t ti = i; ti < i + CPU_TILE_SIZE && ti < m; ++ti) {
            for (int64_t tj = j; tj < j + CPU_TILE_SIZE && tj < n; ++tj) {
              T val = src_matrix[ti * n + tj];
              if constexpr (std::is_same_v<T, phi::dtype::complex<float>> ||
                            std::is_same_v<T, phi::dtype::complex<double>>) {
                dst_matrix[tj * m + ti] = phi::dtype::conj(val);
              } else {
                dst_matrix[tj * m + ti] = val;
              }
            }
          }
        }
      }
    }
  };

  phi::funcs::ForRange<CPUContext> for_range(dev_ctx, batch_size);
  for_range(transpose_task);
}

template class ConjugateTransposeFunctor<float, CPUContext>;
template class ConjugateTransposeFunctor<double, CPUContext>;
template class ConjugateTransposeFunctor<phi::dtype::complex<float>,
                                         CPUContext>;
template class ConjugateTransposeFunctor<phi::dtype::complex<double>,
                                         CPUContext>;

}  // namespace funcs
}  // namespace phi
