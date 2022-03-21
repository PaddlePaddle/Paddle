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

#include "paddle/phi/kernels/sparse/convolution_grad_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/sparse/cpu/convolution.h"

namespace phi {
namespace sparse {

// rulebook:
//[
//  [kernel_index],
//  [in_i],
//  [out_i],
//]
// x_grad = out_grad * transpose(kenrel)
// kernel_grad = transpose(x) * out_grad
template <typename T, typename Context>
void Conv3dGradKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      const DenseTensor& rulebook,
                      const DenseTensor& kernel,
                      const DenseTensor& out_grad,
                      const std::vector<int>& paddings,
                      const std::vector<int>& dilations,
                      const std::vector<int>& strides,
                      const int groups,
                      const bool subm,
                      DenseTensor* x_grad,
                      DenseTensor* kernel_grad) {
  const auto& kernel_dims = kernel.dims();
  const int kernel_size = kernel_dims[0] * kernel_dims[1] * kernel_dims[2];
  const int in_channels = kernel_dims[3];
  const int out_channels = kernel_dims[4];
  const int* rulebook_ptr = rulebook.data<int>();

  const int rulebook_len = rulebook.dims()[1];

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
      phi::Empty(dev_ctx, std::move(out_grad_features_meta));

  T* in_features_ptr = in_features.data<T>();
  T* d_x_features_ptr = d_x_features.data<T>();
  T* out_grad_features_ptr = out_grad_features.data<T>();
  kernel_grad->Resize(kernel_dims);
  dev_ctx.Alloc(
      kernel_grad, kernel_grad->dtype(), kernel_grad->numel() * sizeof(T));
  T* d_kernel_ptr = kernel_grad->data<T>();
  memset(d_kernel_ptr, 0, sizeof(T) * kernel_grad->numel());

  int half_kernel_size = kernel_size / 2;
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  x_grad->Resize(x.non_zero_elements().dims());
  dev_ctx.Alloc(x_grad, x_grad->dtype(), sizeof(T) * x_grad->numel());
  T* x_grad_values_ptr = x_grad->data<T>();
  memset(x_grad_values_ptr, 0, sizeof(T) * x_grad->numel());
  memset(d_x_features_ptr, 0, sizeof(T) * d_x_features.numel());

  std::vector<int> offsets(kernel_size + 1), counter(kernel_size, 0);
  for (int i = 0; i < rulebook_len; i++) {
    counter[rulebook_ptr[i]] += 1;
  }
  int offset = 0, max_count = 0;
  for (int i = 0; i < kernel_size; i++) {
    offsets[i] = offset;
    offset += counter[i];
    if (i < half_kernel_size) {
      max_count = std::max(max_count, counter[i]);
    }
  }
  offsets[kernel_size] = offset;

  if (subm) {
    phi::funcs::sparse::SubmPreProcess<T, Context>(dev_ctx,
                                                   x,
                                                   kernel,
                                                   out_grad,
                                                   in_channels,
                                                   out_channels,
                                                   half_kernel_size,
                                                   kernel_grad,
                                                   x_grad);
    if (max_count == 0) {
      return;
    }
  }

  Gather<T>(x.non_zero_elements().data<T>(),
            rulebook_ptr + rulebook_len,
            rulebook_len,
            in_channels,
            in_features_ptr);
  Gather<T>(out_grad.data<T>(),
            rulebook_ptr + rulebook_len * 2,
            rulebook_len,
            out_channels,
            out_grad_features_ptr);

  const T* kernel_ptr = kernel.data<T>();
  for (int i = 0; i < kernel_size; i++) {
    if (counter[i] <= 0 || (subm && i == half_kernel_size)) {
      continue;
    }

    const int M = counter[i];
    const int K = in_channels;
    const int N = out_channels;
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
  Scatter<T>(d_x_features_ptr,
             rulebook.data<int>() + rulebook_len,
             rulebook_len,
             in_channels,
             x_grad_values_ptr);
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(sparse_conv3d_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::Conv3dGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
