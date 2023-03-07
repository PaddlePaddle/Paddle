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

#include "paddle/phi/kernels/sparse/unary_grad_kernel.h"
#include "paddle/phi/kernels/sparse/unary_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/impl/unary_grad_kernel_impl.h"

namespace phi {
namespace sparse {

// std::vector<int> get_gpu_grad_perm(std::vector<int> perm) {
//   std::vector<int> grad_perm(perm.size());
//   for (unsigned int i = 0; i < perm.size(); ++i) {
//     grad_perm[perm[i]] = i;
//   }
//   return grad_perm;
// }
template <typename T>
__global__ void SetValueCudaKernel(const T* value,
                                   const int64_t length,
                                   T* data) {
  CUDA_KERNEL_LOOP_TYPE(index, length, int64_t) { data[index] = value[0]; }
}

template <typename T, typename Context>
void SumCooGradKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      const SparseCooTensor& dout,
                      const IntArray& axis,
                      bool keep_dim,
                      SparseCooTensor* dx) {}

template <typename T, typename Context>
void SumCsrGradKernel(const Context& dev_ctx,
                      const SparseCsrTensor& x,
                      const SparseCsrTensor& dout,
                      const IntArray& axis,
                      bool keep_dim,
                      SparseCsrTensor* dx) {
  EmptyLikeCsrKernel<T, Context>(dev_ctx, x, dx);
  unsigned int n_dim = axis.size();

  const DenseTensor& x_crows = x.crows();
  const DenseTensor& x_cols = x.cols();
  const DenseTensor& dout_values = dout.values();
  const auto* x_crows_data = x_crows.data<int64_t>();

  DenseTensor* dx_crows = dx->mutable_crows();
  DenseTensor* dx_cols = dx->mutable_cols();
  DenseTensor* dx_values = dx->mutable_values();

  *dx_crows = x_crows;
  *dx_cols = x_cols;

  if (n_dim == 0) {
    auto config =
        phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, dx->nnz(), 1);
    SetValueCudaKernel<int64_t><<<config.block_per_grid.x,
                                  config.thread_per_block.x,
                                  0,
                                  dev_ctx.stream()>>>(
        dout_values.data<T>(), dx->nnz(), dx_values->data<T>());
  } else {
    PADDLE_ENFORCE_EQ(axis[0],
                      -1,
                      phi::errors::Unimplemented(
                          "`axis` of SumCsrKernel only support None or -1 now."
                          "More number will be supported in the future."));

    if (x.dims().size() == 2) {
      int value_index = 0;
      for (int k = 0; k < x.dims()[0]; ++k) {
        if (x_crows_data[k] != x_crows_data[k + 1]) {
          T value = dout_values.data<T>()[value_index];
          for (auto i = x_crows_data[k]; i < x_crows_data[k + 1]; ++i) {
            dx_values->data<T>()[i] = value;
          }
          value_index += 1;
        }
      }
    } else {
      int dout_value_index = 0;
      int dx_value_index = 0;
      for (auto batch = 0; batch < x.dims()[0]; ++batch) {
        for (auto k = batch * (x.dims()[1] + 1);
             k < batch * (x.dims()[1] + 1) + x.dims()[1];
             ++k) {
          if (x_crows_data[k] != x_crows_data[k + 1]) {
            T value = dout_values.data<T>()[dout_value_index];
            for (auto i = x_crows_data[k]; i < x_crows_data[k + 1]; ++i) {
              dx_values->data<T>()[dx_value_index] = value;
              dx_value_index++;
            }
            dout_value_index++;
          }
        }
      }
    }
  }
}
}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(sum_coo_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SumCooGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(sum_csr_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SumCsrGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}
