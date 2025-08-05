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

#include "paddle/phi/kernels/funcs/inverse_from_lu.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/kernels/funcs/lapack/lapack_function.h"

namespace phi {
namespace funcs {

template <typename T>
void InverseFromLUFunctor<T, CPUContext>::operator()(const CPUContext& dev_ctx,
                                                     const DenseTensor& lu_data,
                                                     const DenseTensor& pivots,
                                                     DenseTensor* inverse_out) {
  const auto& dims = lu_data.dims();
  const int rank = dims.size();
  const int64_t n = dims[rank - 1];
  const int64_t n_square = n * n;
  const int64_t batch_size = lu_data.numel() / n_square;

  // getri is an in-place operation, copy lu_data to inverse_out first.
  dev_ctx.template Alloc<T>(inverse_out);
  phi::Copy(dev_ctx, lu_data, dev_ctx.GetPlace(), false, inverse_out);

  auto* pivots_data = pivots.data<int>();
  auto* inverse_data = inverse_out->data<T>();

  for (int64_t i = 0; i < batch_size; ++i) {
    int info = 0;
    phi::funcs::lapackGETRI<T>(
        n, inverse_data + i * n_square, pivots_data + i * n, &info);
  }
}

template class InverseFromLUFunctor<float, CPUContext>;
template class InverseFromLUFunctor<double, CPUContext>;
template class InverseFromLUFunctor<phi::dtype::complex<float>, CPUContext>;
template class InverseFromLUFunctor<phi::dtype::complex<double>, CPUContext>;

}  // namespace funcs
}  // namespace phi
