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

#include "paddle/phi/kernels/sparse/sparse_pool_grad_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/pooling.h"
#include "paddle/phi/kernels/funcs/sparse/convolution.h"

namespace phi {
namespace sparse {

template <typename T, typename IntT = int>
void MaxPoolGradCPUKernel(const CPUContext& dev_ctx,
                          const SparseCooTensor& x,
                          const DenseTensor& rulebook,
                          const SparseCooTensor& out,
                          const SparseCooTensor& out_grad,
                          const std::vector<int>& kernel_sizes,
                          SparseCooTensor* x_grad) {
  int kernel_size = kernel_sizes[0] * kernel_sizes[1] * kernel_sizes[2];
  const int channels = x.dims()[4];
  int rulebook_len = rulebook.dims()[1];
  const IntT* rulebook_ptr = rulebook.data<IntT>();
  std::vector<int> offsets(kernel_size + 1), counter(kernel_size, 0);
  for (int i = 0; i < rulebook_len; i++) {
    counter[rulebook_ptr[i]] += 1;
  }
  phi::funcs::sparse::PrefixSum(&counter[0], &offsets[0], kernel_size);

  const T* in_features_ptr = x.non_zero_elements().data<T>();
  const T* out_features_ptr = out.non_zero_elements().data<T>();
  const T* out_grad_ptr = out_grad.non_zero_elements().data<T>();
  // TODO(zhangkaihuo): call phi::sparse::EmptyLike
  DenseTensor x_grad_indices =
      phi::EmptyLike<IntT>(dev_ctx, x.non_zero_indices());
  DenseTensor x_grad_values = phi::EmptyLike<T>(dev_ctx, x.non_zero_elements());
  x_grad->SetMember(x_grad_indices, x_grad_values, x.dims(), true);
  T* x_grad_ptr = x_grad_values.data<T>();
  memset(x_grad_ptr, 0, sizeof(T) * x_grad->numel());
  phi::Copy<CPUContext>(dev_ctx,
                        x.non_zero_indices(),
                        dev_ctx.GetPlace(),
                        false,
                        &x_grad_indices);

  phi::funcs::MaxPoolGrad<T> grad_functor;
  for (int i = 0; i < kernel_size; i++) {
    for (int j = 0; j < counter[i]; j++) {
      IntT in_i = rulebook_ptr[rulebook_len + offsets[i] + j];
      IntT out_i = rulebook_ptr[rulebook_len * 2 + offsets[i] + j];
      for (int c = 0; c < channels; c++) {
        grad_functor.compute(in_features_ptr[in_i * channels + c],
                             out_features_ptr[out_i * channels + c],
                             out_grad_ptr[out_i * channels + c],
                             1,
                             &x_grad_ptr[in_i * channels + c]);
      }
    }
  }
}

template <typename T, typename Context>
void MaxPoolGradKernel(const Context& dev_ctx,
                       const SparseCooTensor& x,
                       const DenseTensor& rulebook,
                       const SparseCooTensor& out,
                       const SparseCooTensor& out_grad,
                       const std::vector<int>& kernel_sizes,
                       SparseCooTensor* x_grad) {
  PD_VISIT_INTEGRAL_TYPES(
      x.non_zero_indices().dtype(), "MaxPoolGradCPUKernel", ([&] {
        MaxPoolGradCPUKernel<T, data_t>(
            dev_ctx, x, rulebook, out, out_grad, kernel_sizes, x_grad);
      }));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(sparse_maxpool_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::MaxPoolGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
