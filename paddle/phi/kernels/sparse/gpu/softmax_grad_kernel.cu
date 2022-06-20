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
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/softmax_grad_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename IntT = int>
__global__ void SoftmaxGradGpuKernel(const IntT* out_crows,
                                     const T* out_values,
                                     const T* dout_values,
                                     T* dx_values,
                                     int row_number) {
  // dx = (dout - sum(dout * out)) * out
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  int non_zero_idx = threadIdx.x;
  if (row >= row_number) return;
  int row_first = static_cast<int>(out_crows[row]);
  int row_nnz = static_cast<int>(out_crows[row + 1] - out_crows[row]);
  if (row_nnz == 0) return;

  int kIteration = (row_nnz + warpSize - 1) / warpSize;

  T mul_result = 0;
  for (int i = 0; i < kIteration; ++i) {
    int idx = non_zero_idx + i * warpSize;
    if (idx >= row_nnz) break;

    mul_result += out_values[row_first + idx] * dout_values[row_first + idx];
  }
  T sum = phi::funcs::warpReduceSum<T>(mul_result, 0xFFFFFFFF);

  for (int i = 0; i < kIteration; ++i) {
    int idx = non_zero_idx + i * warpSize;
    if (idx >= row_nnz) break;

    dx_values[row_first + idx] =
        (dout_values[row_first + idx] - sum) * out_values[row_first + idx];
  }
}

template <typename T, typename Context>
void SoftmaxCsrGradKernel(const Context& dev_ctx,
                          const SparseCsrTensor& out,
                          const SparseCsrTensor& dout,
                          int axis,
                          SparseCsrTensor* dx) {
  PADDLE_ENFORCE_EQ(axis,
                    -1,
                    phi::errors::Unimplemented(
                        "SparseCsrTensor only support axis=-1 for softmax, "
                        "which is faster when reading data by row (axis=-1)"));
  EmptyLikeCsrKernel<T, Context>(dev_ctx, dout, dx);

  auto out_dim = out.dims();
  int row_number = 1;
  for (int i = 0; i < out_dim.size() - 1; ++i) {
    row_number *= out_dim[i];
  }

  dim3 grid((row_number + 3) / 4);
  dim3 block(32, 4);

  PD_VISIT_INTEGRAL_TYPES(
      out.non_zero_crows().dtype(), "SoftmaxCsrGradKernel", ([&] {
        SoftmaxGradGpuKernel<T, data_t><<<grid, block, 0, dev_ctx.stream()>>>(
            out.non_zero_crows().data<data_t>(),
            out.non_zero_elements().data<T>(),
            dout.non_zero_elements().data<T>(),
            dx->mutable_non_zero_elements()->data<T>(),
            row_number);
      }));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(softmax_csr_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SoftmaxCsrGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
