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

#pragma once

#include "paddle/common/ddim.h"
#include "paddle/phi/core/kmap_cache.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace phi {
namespace funcs {
namespace sparse {

template <typename T, typename Context>
inline void SubmPreProcess(const Context& dev_ctx,
                           const SparseCooTensor& x,
                           const DenseTensor& kernel,
                           const DenseTensor& out_grad,
                           const int in_channels,
                           const int out_channels,
                           const int half_kernel_size,
                           DenseTensor* kernel_grad,
                           DenseTensor* x_grad) {
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  const bool is_params_freezing = kernel_grad == nullptr;
  if (!is_params_freezing) {
    T* d_kernel_ptr = kernel_grad->data<T>();
    blas.GEMM(CblasTrans,
              CblasNoTrans,
              x.non_zero_elements().dims()[1],
              out_grad.dims()[1],
              x.non_zero_elements().dims()[0],
              static_cast<T>(1),
              x.non_zero_elements().data<T>(),
              out_grad.data<T>(),
              static_cast<T>(0),
              d_kernel_ptr + half_kernel_size * in_channels * out_channels);
  }

  // call gemm: d_x = out_grad * transpose(kernel)
  // (n, out_channels) * (out_channels, in_channels)
  T* x_grad_ptr = x_grad->data<T>();
  blas.GEMM(CblasNoTrans,
            CblasTrans,
            out_grad.dims()[0],
            in_channels,
            out_grad.dims()[1],
            static_cast<T>(1),
            out_grad.data<T>(),
            kernel.data<T>() + half_kernel_size * in_channels * out_channels,
            static_cast<T>(0),
            x_grad_ptr);
}

}  // namespace sparse
}  // namespace funcs
}  // namespace phi
