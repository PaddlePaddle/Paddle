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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/lapack/lapack_function.h"
#include "paddle/phi/kernels/transpose_kernel.h"
#include "paddle/phi/kernels/tril_triu_kernel.h"

namespace phi {

template <typename T, typename Context>
void CholeskyInverseKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           bool upper,
                           DenseTensor* out) {
  auto& dims = x.dims();
  int batch_count = 1;
  for (int i = 0; i < dims.size() - 2; i++) {
    batch_count *= static_cast<int>(dims[i]);
  }
  int m = dims[dims.size() - 1];
  out->Resize(x.dims());
  dev_ctx.template Alloc<T>(out);

  DenseTensor trans_result = phi::TransposeLast2Dim<T>(dev_ctx, x);
  phi::Copy<Context>(dev_ctx, trans_result, dev_ctx.GetPlace(), false, out);
  auto* out_data = out->data<T>();

  DenseTensor info = phi::Empty<int, Context>(dev_ctx, IntArray({batch_count}));
  int* info_data = info.data<int>();

  char uplo = upper ? 'U' : 'L';
  for (int i = 0; i < batch_count; i++) {
    funcs::lapackCholeskyInverse<T>(
        uplo, m, out_data + i * m * m, std::max(1, m), info_data + i);
    PADDLE_ENFORCE_EQ(
        *(info_data + i),
        0,
        errors::InvalidArgument("Cholesky inverse was not successful. The "
                                "%d-th input matrice "
                                "might not be invertible.",
                                i));
  }
  DenseTensor* tri = new DenseTensor();
  tri->Resize(x.dims());
  dev_ctx.template Alloc<T>(tri);
  phi::Copy<Context>(dev_ctx, *out, dev_ctx.GetPlace(), false, tri);

  if (upper) {
    phi::TrilTriuKernel<T>(dev_ctx, *out, -1, true, tri);
  } else {
    phi::TrilTriuKernel<T>(dev_ctx, *out, 1, false, tri);
  }

  trans_result = phi::TransposeLast2Dim<T>(dev_ctx, *tri);
  phi::Copy(dev_ctx, trans_result, dev_ctx.GetPlace(), false, tri);
  phi::AddKernel<T>(dev_ctx, *out, *tri, out);
}
}  // namespace phi

PD_REGISTER_KERNEL(cholesky_inverse,
                   CPU,
                   ALL_LAYOUT,
                   phi::CholeskyInverseKernel,
                   float,
                   double) {}
