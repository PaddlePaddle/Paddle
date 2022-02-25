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

#include "paddle/phi/kernels/sparse/cpu/convolution.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace phi {
namespace sparse {

// rulebook:
//[
//  [counter],
//  [in_i],
//  [out_i],
//]
template <typename T, typename Context>
void Conv3dGradKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      const DenseTensor& rulebook,
                      const DenseTensor& kernel,
                      const SparseCooTensor& out_grad,
                      const std::vector<int>& paddings,
                      const std::vector<int>& dilations,
                      const std::vector<int>& strides,
                      const int groups,
                      DenseTensor* x_grad,
                      DenseTensor* kernel_grad) {
  const auto& kernel_dims = kernel.dims();
  const int kernel_size = kernel_dims[0] * kernel_dims[1] * kernel_dims[2];
  const int in_channels = kernel_dims[3];
  const int out_channels = kernel_dims[4];
  const int* rulebook_ptr = rulebook.data<int>();

  DenseTensorMeta in_features_meta(
      x.dtype(), {rulebook_len, in_channels}, DataLayout::NCHW);
  DenseTensorMeta d_x_features_meta(
      x.dtype(), {rulebook_len, in_channels}, DataLayout::NCHW);
  DenseTensorMeta out_grad_features_meta(
      x.dtype(), {rulebook_len, out_channels}, DataLayout::NCHW);
  phi::DenseTensor in_features =
      phi::Empty(dev_ctx, std::move(in_features_meta));
  phi::DenseTensor d_x_features =
      phi::Empty(dev_ctx, std::move(d_x_features_meta));
  phi::DenseTensor out_grad_features =
      phi::Empty(dev_ctx, std::move(out_features_meta));

  T* in_features_ptr = in_features.mutable_data<T>(place);
  T* d_x_features_ptr = d_x_features.mutable_data<T>(place);
  T* out_grad_features_ptr = out_grad_features.mutable_data<T>(place);
  kernel_grad.Resize(kernel_dims);
  T* d_kernel_ptr = kernel_grad->mutable_data<T>(place);

  const int rulebook_len = rulebook.dims()[1];
  Gather<T>(x.non_zero_elements().data<T>(),
            rulebook_ptr + rulebook_len,
            rulebook_len,
            in_channels,
            in_features_ptr);
  Gather<T>(out_grad.non_zero_elements().data<T>(),
            rulebook_ptr + rulebook_len * 2,
            rulebook_len,
            out_channels,
            out_grad_features_ptr);

  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  std::vector<int> offsets(kernel_size + 1);
  const int* counter_ptr = rulebook_ptr;
  int offset = 0;
  for (int i = 0; i < kernel_size; i++) {
    offsets[i] = offset;
    offset += counter_ptr[i];
  }
  offsets[kernel_size] = offset;

  const T* kernel_ptr = kernel.data<T>();
  for (int i = 0; i < kernel_size; i++) {
    if (counter_ptr[i] <= 0) {
      continue;
    }

    const int M = counter_ptr[i];
    const int K = in_channels;   // in_channels
    const int N = out_channels;  // out_channels
    T* tmp_in_ptr = in_features_ptr + offsets[i] * in_channels;
    T* tmp_out_grad_ptr = out_grad_features_ptr + offsets[i] * out_channels;
    const T* tmp_kernel_ptr = kernel_ptr + i * in_channels * out_channels;
    T* tmp_d_x_ptr = d_x_features_ptr + offsets[i] * out_channels;
    T* tmp_d_kernel_ptr = d_kernel_ptr + i * in_channels * out_channels;

    // call gemm: d_kernel = transpose(x) * out_grad
    // (in_channels, n) * (n, out_channels)
    blas.GEMM(CblasTrans,
              CblasNoTrans,
              M,
              N,
              K,
              static_cast<T>(1),
              tmp_in_ptr,
              tmp_out_grad_ptr,
              static_cast<T>(0),
              tmp_d_kernel_ptr);

    // call gemm: d_x = out_grad * transpose(kernel)
    // (n, out_channels) * (out_channels, in_channels)
    blas.GEMM(CblasNoTrans,
              CblasTrans,
              M,
              K,
              N,
              static_cast<T>(1),
              tmp_out_grad_ptr,
              tmp_kernel_ptr,
              static_cast<T>(0),
              tmp_d_x_ptr);
  }

  // 4. scatter
  x_grad.Resize(x.dims());
  T* x_grad_values_ptr = x_grad->mutable_data<T>(place);
  memset(x_grad_values_ptr, 0, sizeof(T) * x_grad->nnz() * in_channels);
  Scatter<T>(d_x_features_ptr,
             rulebook.data<int>() + rulebook_len,
             rulebook_len,
             in_channels,
             x_grad_values_ptr);
}

}  // namespace sparse
}  // namespace phi
