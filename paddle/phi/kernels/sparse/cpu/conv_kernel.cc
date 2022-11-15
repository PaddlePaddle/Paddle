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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/sparse/cpu/conv.h"

namespace phi {
namespace sparse {

/**
 * x: (N, D, H, W, C)
 * kernel: (D, H, W, C, OC)
 * out: (N, D, H, W, OC)
 **/
template <typename T, typename IntT = int>
void Conv3dCooCPUKernel(const CPUContext& dev_ctx,
                        const SparseCooTensor& x,
                        const DenseTensor& kernel,
                        const std::vector<int>& paddings,
                        const std::vector<int>& dilations,
                        const std::vector<int>& strides,
                        const int groups,
                        const bool subm,
                        const std::string& key,
                        SparseCooTensor* out,
                        DenseTensor* rulebook,
                        DenseTensor* counter) {
  // update padding and dilation
  // Currently, only support x.layout is NDHWC, groups = 1
  // if x.layout != NDHWC then transpose(x), transpose(weight)

  const auto& x_dims = x.dims();
  const auto& kernel_dims = kernel.dims();
  int kernel_size = kernel_dims[0] * kernel_dims[1] * kernel_dims[2];
  DDim out_dims = {1, 1, 1, 1, 1};
  std::vector<int> kernel_sizes(kernel_dims.size());
  for (int i = 0; i < kernel_dims.size(); i++) {
    kernel_sizes[i] = kernel_dims[i];
  }

  std::vector<int> subm_paddings(paddings), subm_strides(strides);
  if (subm) {
    // the out shape of subm_conv is same as input shape
    // reset the padding=kernel_size/2 and strides=1
    phi::funcs::sparse::ResetSubmKernelSizeAndStrides(
        kernel.dims(), &subm_paddings, &subm_strides);
  }

  phi::funcs::sparse::GetOutShape(
      x_dims, kernel_sizes, subm_paddings, dilations, subm_strides, &out_dims);
  const int in_channels = kernel_dims[3];
  const int out_channels = kernel_dims[4];

  // Second algorithm:
  // https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf
  // 1. product rulebook
  DenseTensor h_counter, h_offsets;
  h_counter.Resize({kernel_size});
  h_offsets.Resize({kernel_size + 1});
  int* h_counter_ptr = dev_ctx.template HostAlloc<int>(&h_counter);
  int* h_offsets_ptr = dev_ctx.template HostAlloc<int>(&h_offsets);

  // DenseTensor* rulebook = nullptr;
  const IntT* rulebook_ptr = nullptr;
  int n = 0;
  bool need_product_rulebook = true;
  if (subm && !key.empty()) {
    rulebook_ptr = phi::funcs::sparse::PrepareSubm<T, IntT, CPUContext>(
        dev_ctx,
        x,
        key,
        out_dims,
        out,
        h_counter_ptr,
        h_offsets_ptr,
        &n,
        &need_product_rulebook);
  }
  if (need_product_rulebook) {
    DenseTensor tmp_rulebook;
    ProductRuleBook<T, CPUContext, IntT>(dev_ctx,
                                         x,
                                         kernel_sizes,
                                         subm_paddings,
                                         dilations,
                                         subm_strides,
                                         out_dims,
                                         subm,
                                         &tmp_rulebook,
                                         h_counter_ptr);

    UpdateRulebookAndOutIndex<T, CPUContext, IntT>(
        dev_ctx, x, kernel_size, out_channels, out_dims, &tmp_rulebook, out);
    n = tmp_rulebook.dims()[1];
    rulebook_ptr = tmp_rulebook.data<IntT>();

    phi::funcs::sparse::SaveToTable(
        dev_ctx, x, key, tmp_rulebook, h_counter, out, rulebook, counter);
  }
  // int n = rulebook->dims()[1];

  // 2. gather
  DenseTensorMeta in_features_meta(
      x.dtype(), {n, in_channels}, DataLayout::NHWC);
  DenseTensorMeta out_features_meta(
      x.dtype(), {n, out_channels}, DataLayout::NHWC);
  phi::DenseTensor in_features =
      phi::Empty(dev_ctx, std::move(in_features_meta));
  phi::DenseTensor out_features =
      phi::Empty(dev_ctx, std::move(out_features_meta));
  T* in_features_ptr = in_features.data<T>();
  T* out_features_ptr = out_features.data<T>();

  Gather<T, IntT>(
      x.values().data<T>(), rulebook_ptr + n, n, in_channels, in_features_ptr);

  // 3. call gemm for every werght
  auto blas = phi::funcs::GetBlas<CPUContext, T>(dev_ctx);
  int offset = 0;
  for (int i = 0; i < kernel_size; i++) {
    h_offsets_ptr[i] = offset;
    offset += h_counter_ptr[i];
  }
  h_offsets_ptr[kernel_size] = offset;

  const T* kernel_ptr = kernel.data<T>();
  for (int i = 0; i < kernel_size; i++) {
    if (h_counter_ptr[i] <= 0) {
      continue;
    }

    // call gemm: (n, in_channels) * (in_channels, out_channels)
    const int M = h_counter_ptr[i];
    const int K = in_channels;   // in_channels
    const int N = out_channels;  // out_channels
    T* tmp_in_ptr = in_features_ptr + h_offsets_ptr[i] * in_channels;
    const T* tmp_kernel_ptr = kernel_ptr + i * K * N;
    T* tmp_out_ptr = out_features_ptr + h_offsets_ptr[i] * out_channels;
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
  T* out_values_ptr = out->mutable_values()->data<T>();
  memset(out_values_ptr, 0, sizeof(T) * out->nnz() * out_channels);
  Scatter<T, IntT>(
      out_features_ptr, rulebook_ptr + n * 2, n, out_channels, out_values_ptr);
}

template <typename T, typename Context>
void Conv3dCooKernel(const Context& dev_ctx,
                     const SparseCooTensor& x,
                     const DenseTensor& kernel,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     const std::vector<int>& strides,
                     const int groups,
                     const bool subm,
                     const std::string& key,
                     SparseCooTensor* out,
                     DenseTensor* rulebook,
                     DenseTensor* counter) {
  PD_VISIT_BASE_INTEGRAL_TYPES(x.indices().dtype(), "Conv3dCooCPUKernel", ([&] {
                                 Conv3dCooCPUKernel<T, data_t>(dev_ctx,
                                                               x,
                                                               kernel,
                                                               paddings,
                                                               dilations,
                                                               strides,
                                                               groups,
                                                               subm,
                                                               key,
                                                               out,
                                                               rulebook,
                                                               counter);
                               }));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(
    conv3d_coo, CPU, ALL_LAYOUT, phi::sparse::Conv3dCooKernel, float, double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
