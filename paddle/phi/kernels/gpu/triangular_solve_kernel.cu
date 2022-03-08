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

#include "paddle/phi/kernels/triangular_solve_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memory.h"

namespace phi {

template <typename T, typename Context>
void TriangularSolveKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           bool upper,
                           bool transpose,
                           bool unitriangular,
                           DenseTensor* out) {
  // get broadcast dim
  std::vector<int64_t> x_bst_dims_vec;
  std::vector<int64_t> y_bst_dims_vec;
  std::tie(x_bst_dims_vec, y_bst_dims_vec) =
      funcs::MatrixGetBroadcastDims(x, y);
  int x_bst_ndim = x_bst_dims_vec.size();
  int y_bst_ndim = y_bst_dims_vec.size();

  // Tensor broadcast to 'out' and temp 'x_bst'
  ScalarArray x_bst_dims(x_bst_dims_vec);
  DenseTensor x_bst = phi::Empty<T, Context>(dev_ctx, x_bst_dims);
  const T* x_bst_data = x_bst.data<T>();
  ExpandKernel<T, Context>(dev_ctx, x, x_bst_dims, &x_bst);

  out->Resize(phi::make_ddim(y_bst_dims_vec));
  T* out_data = dev_ctx.template Alloc<T>(out);
  ScalarArray y_bst_dims(y_bst_dims_vec);
  ExpandKernel<T, Context>(dev_ctx, y, y_bst_dims, out);

  // calculate use cublas library
  CBLAS_UPLO uplo = upper ? CblasUpper : CblasLower;
  CBLAS_TRANSPOSE transA = transpose ? CblasTrans : CblasNoTrans;
  CBLAS_DIAG diag = unitriangular ? CblasUnit : CblasNonUnit;

  int M = static_cast<int>(y_bst_dims_vec[y_bst_ndim - 2]);
  int N = static_cast<int>(y_bst_dims_vec[y_bst_ndim - 1]);
  auto lda = std::max(1, M);
  auto ldb = std::max(1, N);

  int batch_size = 1;
  for (int i = 0; i < x_bst_ndim - 2; i++) {
    batch_size *= x_bst_dims_vec[i];
  }

  auto blas = phi::funcs::GetBlas<GPUContext, T>(dev_ctx);
  if (batch_size <= 8 && M >= 64) {
    for (auto i = 0; i < batch_size; i++) {
      blas.TRSM(CblasLeft,
                uplo,
                transA,
                diag,
                M,
                N,
                T(1),
                x_bst_data + i * M * M,
                lda,
                out_data + i * N * M,
                ldb);
    }
  } else {
    std::vector<const T*> cpu_ptrs(batch_size * 2);
    for (int i = 0; i < batch_size; ++i) {
      cpu_ptrs[i] = x_bst_data + i * M * M;
      cpu_ptrs[i + batch_size] = out_data + i * M * N;
    }

    // Copy the addresses of A and tmp_b from host to device.
    paddle::memory::allocation::AllocationPtr tmp_gpu_ptrs_data =
        paddle::memory::Alloc(dev_ctx, cpu_ptrs.size() * sizeof(T*));

    paddle::memory::Copy(dev_ctx.GetPlace(),
                         tmp_gpu_ptrs_data->ptr(),
                         paddle::platform::CPUPlace(),
                         static_cast<void*>(cpu_ptrs.data()),
                         cpu_ptrs.size() * sizeof(T*),
                         dev_ctx.stream());

    const T** gpu_a_ptrs =
        reinterpret_cast<const T**>(tmp_gpu_ptrs_data->ptr());
    T** gpu_b_ptrs =
        reinterpret_cast<T**>(tmp_gpu_ptrs_data->ptr()) + batch_size;
    blas.BatchedTRSM(CblasLeft,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     static_cast<T>(1.0),
                     gpu_a_ptrs,
                     lda,
                     gpu_b_ptrs,
                     ldb,
                     batch_size);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(triangular_solve,
                   GPU,
                   ALL_LAYOUT,
                   phi::TriangularSolveKernel,
                   float,
                   double) {}
