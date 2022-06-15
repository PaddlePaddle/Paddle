/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/softmax_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename IntT = int>
__global__ void SoftmaxGpuKernel(const IntT* x_crows,
                                 const T* x_values,
                                 T* out_values,
                                 int row_number) {
  // out = exp(x-x_max) / sum(exp(x-x_max))
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  int non_zero_idx = threadIdx.x;
  if (row >= row_number) return;
  int row_first = static_cast<int>(x_crows[row]);
  int row_nnz = static_cast<int>(x_crows[row + 1] - x_crows[row]);
  if (row_nnz == 0) return;

  int kIteration = (row_nnz + warpSize - 1) / warpSize;

  T max_val = -std::numeric_limits<T>::infinity();
  for (int i = 0; i < kIteration; ++i) {
    int idx = non_zero_idx + i * warpSize;
    if (idx >= row_nnz) break;

    if (max_val < x_values[row_first + idx]) {
      max_val = x_values[row_first + idx];
    }
  }
  T row_max_val = phi::funcs::warpReduceMax<T>(max_val, 0xFFFFFFFF);

  T exp_sum = 0;
  for (int i = 0; i < kIteration; ++i) {
    int idx = non_zero_idx + i * warpSize;
    if (idx >= row_nnz) break;

    auto functor = phi::funcs::CudaExpFunctor<T>();
    out_values[row_first + idx] =
        functor(x_values[row_first + idx] - row_max_val);
    exp_sum += functor(x_values[row_first + idx] - row_max_val);
  }
  T row_exp_sum = phi::funcs::warpReduceSum<T>(exp_sum, 0xFFFFFFFF);

  for (int i = 0; i < kIteration; ++i) {
    int idx = non_zero_idx + i * warpSize;
    if (idx >= row_nnz) break;

    out_values[row_first + idx] = out_values[row_first + idx] / row_exp_sum;
  }
}

template <typename T, typename Context>
void SoftmaxCsrKernel(const Context& dev_ctx,
                      const SparseCsrTensor& x,
                      int axis,
                      SparseCsrTensor* out) {
  PADDLE_ENFORCE_EQ(axis,
                    -1,
                    phi::errors::Unimplemented(
                        "SparseCsrTensor only support axis=-1 for softmax, "
                        "which is faster when reading data by row (axis=-1)"));
  EmptyLikeCsrKernel<T, Context>(dev_ctx, x, out);

  auto x_dim = x.dims();
  int row_number = 1;
  for (int i = 0; i < x_dim.size() - 1; ++i) {
    row_number *= x_dim[i];
  }
  dim3 grid((row_number + 3) / 4);
  dim3 block(32, 4);

  DenseTensor tmp_tensor =
      phi::EmptyLike<T, Context>(dev_ctx, x.non_zero_elements());

  PD_VISIT_INTEGRAL_TYPES(x.non_zero_crows().dtype(), "CsrSoftmaxKernel", ([&] {
                            SoftmaxGpuKernel<T, data_t>
                                <<<grid, block, 0, dev_ctx.stream()>>>(
                                    x.non_zero_crows().data<data_t>(),
                                    x.non_zero_elements().data<T>(),
                                    out->mutable_non_zero_elements()->data<T>(),
                                    row_number);
                          }));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(softmax_csr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SoftmaxCsrKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
