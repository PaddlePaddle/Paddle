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

#include "paddle/phi/kernels/sparse/convolution_kernel.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/sparse/cpu/convolution.h"

namespace phi {
namespace sparse {

/**
 * x: (N, D, H, W, C)
 * kernel: (D, H, W, C, OC)
 * out: (N, D, H, W, OC)
**/
template <typename T, typename Context>
void Conv3dKernel(const Context& dev_ctx,
                  const SparseCooTensor& x,
                  const DenseTensor& kernel,
                  const std::vector<int>& paddings,
                  const std::vector<int>& dilations,
                  const std::vector<int>& strides,
                  const int groups,
                  SparseCooTensor* out,
                  DenseTensor* rulebook) {
  // update padding and dilation
  // Currently, only support x.layout is NDHWC, groups = 1
  // if x.layout != NDHWC then transpose(x), transpose(weight)

  const auto& x_dims = x.dims();
  const auto& kernel_dims = kernel.dims();
  int kernel_size = kernel_dims[0] * kernel_dims[1] * kernel_dims[2];
  DDim out_dims = {1, 1, 1, 1, 1};
  GetOutShape(x_dims, kernel_dims, paddings, dilations, strides, &out_dims);
  const int in_channels = kernel_dims[3];
  const int out_channels = kernel_dims[4];

  // Second algorithm:
  // https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf
  // 1. product rulebook
  DenseTensorMeta counter_meta(
      DataType::INT32, {kernel_size}, DataLayout::NCHW);
  // DenseTensor rulebook = phi::Empty<int, Context>(dev_ctx);
  DenseTensor counter_per_kernel = phi::Empty(dev_ctx, std::move(counter_meta));

  ProductRuleBook<T, Context>(dev_ctx,
                              x,
                              kernel,
                              paddings,
                              dilations,
                              strides,
                              out_dims,
                              rulebook,
                              &counter_per_kernel);

  UpdateRulebookAndOutIndex<T>(
      dev_ctx, x, kernel_size, out_channels, out_dims, rulebook, out);

  int n = rulebook->dims()[1];
  const int* counter_ptr = counter_per_kernel.data<int>();

  // 2. gather
  DenseTensorMeta in_features_meta(
      x.dtype(), {n, in_channels}, DataLayout::NHWC);
  DenseTensorMeta out_features_meta(
      x.dtype(), {n, out_channels}, DataLayout::NHWC);
  phi::DenseTensor in_features =
      phi::Empty(dev_ctx, std::move(in_features_meta));
  phi::DenseTensor out_features =
      phi::Empty(dev_ctx, std::move(out_features_meta));
  dev_ctx.Alloc(&in_features, x.dtype(), sizeof(T) * in_features.numel());
  dev_ctx.Alloc(&out_features, x.dtype(), sizeof(T) * out_features.numel());
  T* in_features_ptr = in_features.data<T>();
  T* out_features_ptr = out_features.data<T>();

  Gather<T>(x.non_zero_elements().data<T>(),
            rulebook->data<int>() + n,
            n,
            in_channels,
            in_features_ptr);

  // 3. call gemm for every werght
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  std::vector<int> offsets(kernel_size + 1);
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

    // call gemm: (n, in_channels) * (in_channels, out_channels)
    const int M = counter_ptr[i];
    const int K = in_channels;   // in_channels
    const int N = out_channels;  // out_channels
    T* tmp_in_ptr = in_features_ptr + offsets[i] * in_channels;
    const T* tmp_kernel_ptr = kernel_ptr + i * K * N;
    T* tmp_out_ptr = out_features_ptr + offsets[i] * out_channels;
    blas.GEMM(CblasNoTrans,
              CblasNoTrans,
              M,
              N,
              K,
              static_cast<T>(1),
              tmp_in_ptr,
              tmp_kernel_ptr,
              static_cast<T>(0),
              tmp_out_ptr);
  }

  // 4. scatter
  dev_ctx.Alloc(out->mutable_non_zero_elements(),
                out->mutable_non_zero_elements()->dtype(),
                sizeof(T) * in_features.numel());
  T* out_values_ptr = out->mutable_non_zero_elements()->data<T>();
  memset(out_values_ptr, 0, sizeof(T) * out->nnz() * out_channels);
  Scatter<T>(out_features_ptr,
             rulebook->data<int>() + n * 2,
             n,
             out_channels,
             out_values_ptr);
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(
    sparse_conv3d, CPU, ALL_LAYOUT, phi::sparse::Conv3dKernel, float, double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
