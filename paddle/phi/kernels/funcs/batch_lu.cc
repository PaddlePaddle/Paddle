/// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/funcs/batch_lu.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/lapack/lapack_function.h"

namespace phi {
namespace funcs {

template <typename T, typename Context>
void BatchLUFunctor<T, Context>::operator()(const Context& dev_ctx,
                                            const DenseTensor& x,
                                            DenseTensor* lu_out,
                                            DenseTensor* pivots,
                                            DenseTensor* infos) {
  const auto& x_dims = x.dims();
  const int rank = x_dims.size();
  const int n = static_cast<int>(x_dims[rank - 1]);
  const int64_t matrix_size = static_cast<int64_t>(n) * n;
  const int64_t batch_size = x.numel() / matrix_size;

  if (batch_size == 0) {
    return;
  }

  // getrf is an in-place operation, so we copy input 'x' to 'lu_out'.
  dev_ctx.template Alloc<T>(lu_out);
  phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, lu_out);

  dev_ctx.template Alloc<int>(pivots);
  dev_ctx.template Alloc<int>(infos);

  auto* lu_data = lu_out->data<T>();
  auto* pivots_data = pivots->data<int>();
  auto* infos_data = infos->data<int>();

  for (int64_t i = 0; i < batch_size; ++i) {
    infos_data[i] = 0;

    T* current_lu = lu_data + i * matrix_size;
    int* current_pivots = pivots_data + i * n;
    int* current_info = infos_data + i;

    phi::funcs::lapackGETRF<T>(
        n, n, current_lu, n, current_pivots, current_info);
  }
}

template class BatchLUFunctor<float, CPUContext>;
template class BatchLUFunctor<double, CPUContext>;
template class BatchLUFunctor<phi::dtype::complex<float>, CPUContext>;
template class BatchLUFunctor<phi::dtype::complex<double>, CPUContext>;

}  // namespace funcs
}  // namespace phi
