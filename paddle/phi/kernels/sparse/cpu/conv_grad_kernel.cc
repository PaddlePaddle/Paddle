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

#include "paddle/phi/kernels/sparse/conv_grad_kernel.h"

#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/sparse/cpu/conv.h"

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
template <typename T, typename IntT = int>
void Conv3dCooGradCPUKernel(const CPUContext& dev_ctx,
                            const SparseCooTensor& x,
                            const DenseTensor& kernel,
                            const SparseCooTensor& out,
                            const DenseTensor& rulebook,
                            const DenseTensor& counter,
                            const SparseCooTensor& out_grad,
                            const std::vector<int>& paddings,
                            const std::vector<int>& dilations,
                            const std::vector<int>& strides,
                            const int groups,
                            const bool subm,
                            const std::string& key,
                            SparseCooTensor* x_grad,
                            DenseTensor* kernel_grad) {
  const auto& kernel_dims = kernel.dims();
  const int kernel_size = kernel_dims[0] * kernel_dims[1] * kernel_dims[2];
  const int in_channels = kernel_dims[3];
  const int out_channels = kernel_dims[4];

  int rulebook_len = 0;
  const IntT* rulebook_ptr = phi::funcs::sparse::GetRulebookPtr<IntT>(
      out, rulebook, key, &rulebook_len);
  const int* counter_ptr = phi::funcs::sparse::GetCounterPtr(out, counter, key);

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
  *kernel_grad = phi::EmptyLike<T>(dev_ctx, kernel);
  T* d_kernel_ptr = kernel_grad->data<T>();
  memset(d_kernel_ptr, 0, sizeof(T) * kernel_grad->numel());

  int half_kernel_size = kernel_size / 2;
  auto blas = phi::funcs::GetBlas<CPUContext, T>(dev_ctx);
  DenseTensor x_grad_indices = phi::EmptyLike<IntT>(dev_ctx, x.indices());
  DenseTensor x_grad_values = phi::EmptyLike<T>(dev_ctx, x.values());
  T* x_grad_values_ptr = x_grad_values.data<T>();
  memset(x_grad_values_ptr, 0, sizeof(T) * x_grad_values.numel());
  memset(d_x_features_ptr, 0, sizeof(T) * d_x_features.numel());
  phi::Copy<CPUContext>(
      dev_ctx, x.indices(), dev_ctx.GetPlace(), false, &x_grad_indices);
  x_grad->SetMember(x_grad_indices, x_grad_values, x.dims(), true);

  std::vector<IntT> offsets(kernel_size + 1);
  IntT offset = 0;
  int max_count = 0;
  for (int i = 0; i < kernel_size; i++) {
    offsets[i] = offset;
    offset += counter_ptr[i];
    if (i < half_kernel_size) {
      max_count = std::max(max_count, counter_ptr[i]);
    }
  }
  offsets[kernel_size] = offset;

  if (subm) {
    phi::funcs::sparse::SubmPreProcess<T, CPUContext>(dev_ctx,
                                                      x,
                                                      kernel,
                                                      out_grad.values(),
                                                      in_channels,
                                                      out_channels,
                                                      half_kernel_size,
                                                      kernel_grad,
                                                      &x_grad_values);
    if (max_count == 0) {
      return;
    }
  }

  Gather<T, IntT>(x.values().data<T>(),
                  rulebook_ptr + rulebook_len,
                  rulebook_len,
                  in_channels,
                  in_features_ptr);
  Gather<T, IntT>(out_grad.values().data<T>(),
                  rulebook_ptr + rulebook_len * 2,
                  rulebook_len,
                  out_channels,
                  out_grad_features_ptr);

  const T* kernel_ptr = kernel.data<T>();
  for (int i = 0; i < kernel_size; i++) {
    if (counter_ptr[i] <= 0 || (subm && i == half_kernel_size)) {
      continue;
    }

    const int M = counter_ptr[i];
    const int K = in_channels;
    const int N = out_channels;
    T* tmp_in_ptr = in_features_ptr + offsets[i] * in_channels;
    T* tmp_out_grad_ptr = out_grad_features_ptr + offsets[i] * out_channels;
    const T* tmp_kernel_ptr = kernel_ptr + i * in_channels * out_channels;
    T* tmp_d_x_ptr = d_x_features_ptr + offsets[i] * in_channels;
    T* tmp_d_kernel_ptr = d_kernel_ptr + i * in_channels * out_channels;

    // call gemm: d_kernel = transpose(x) * out_grad
    // (in_channels, n) * (n, out_channels)
    blas.GEMM(CblasTrans,
              CblasNoTrans,
              K,
              N,
              M,
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
  Scatter<T, IntT>(d_x_features_ptr,
                   rulebook_ptr + rulebook_len,
                   rulebook_len,
                   in_channels,
                   x_grad_values_ptr);
}

template <typename T, typename Context>
void Conv3dCooGradKernel(const Context& dev_ctx,
                         const SparseCooTensor& x,
                         const DenseTensor& kernel,
                         const SparseCooTensor& out,
                         const DenseTensor& rulebook,
                         const DenseTensor& counter,
                         const SparseCooTensor& out_grad,
                         const std::vector<int>& paddings,
                         const std::vector<int>& dilations,
                         const std::vector<int>& strides,
                         const int groups,
                         const bool subm,
                         const std::string& key,
                         SparseCooTensor* x_grad,
                         DenseTensor* kernel_grad) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      x.indices().dtype(), "Conv3dCooGradCPUKernel", ([&] {
        Conv3dCooGradCPUKernel<T, data_t>(dev_ctx,
                                          x,
                                          kernel,
                                          out,
                                          rulebook,
                                          counter,
                                          out_grad,
                                          paddings,
                                          dilations,
                                          strides,
                                          groups,
                                          subm,
                                          key,
                                          x_grad,
                                          kernel_grad);
      }));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(conv3d_coo_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::Conv3dCooGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
