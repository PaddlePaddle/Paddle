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

struct Dims4D {
  int dims[4];
  Dims4D(const int batch, const int x, const int y, const int z) {
    dims[0] = batch;
    dims[1] = z;
    dims[2] = y;
    dims[3] = x;
  }
  HOSTDEVICE const int& operator[](int i) const { return dims[i]; }
};

// Judge whether the current position x is in (lower, upper)
template <typename IntT = int>
inline HOSTDEVICE bool Check(const IntT& x,
                             const int& kx,
                             const int& pad,
                             const int& stride,
                             const int dilation,
                             const int kdim,
                             const int xdim) {
  const IntT lower = x - dilation * kx + pad;
  const IntT upper = x + (kdim - kx - 1) * dilation - pad;
  return (lower >= 0 && lower % stride == 0 && upper < xdim);
}

// Check whether the current position(x, y, z) is legal:
// Judge the minimum and maximum values at each latitude
template <typename IntT = int>
inline HOSTDEVICE bool Check(const Dims4D& dims,
                             const Dims4D& kernel_dims,
                             const Dims4D& paddings,
                             const Dims4D& dilations,
                             const Dims4D& strides,
                             const IntT x,
                             const IntT y,
                             const IntT z,
                             const int kx,
                             const int ky,
                             const int kz) {
  bool x_valid = Check(
      x, kx, paddings[3], strides[3], dilations[3], kernel_dims[3], dims[3]);
  bool y_valid = Check(
      y, ky, paddings[2], strides[2], dilations[2], kernel_dims[2], dims[2]);
  bool z_valid = Check(
      z, kz, paddings[1], strides[1], dilations[1], kernel_dims[1], dims[1]);
  return (x_valid && y_valid && z_valid);
}

template <typename Dim, typename IntT = int>
inline HOSTDEVICE IntT PointToIndex(const IntT& batch,
                                    const IntT& x,
                                    const IntT& y,
                                    const IntT& z,
                                    const Dim& dims) {
  return batch * dims[1] * dims[2] * dims[3] + z * dims[2] * dims[3] +
         y * dims[3] + x;
}

// TODO(zhangkaihuo): use division and multiply to optimize
// modulo operation
template <typename Dim, typename IntT = int>
inline HOSTDEVICE void IndexToPoint(
    const IntT index, const Dim& dims, IntT* batch, IntT* x, IntT* y, IntT* z) {
  IntT n = index;
  *x = n % dims[3];
  n /= dims[3];
  *y = n % dims[2];
  n /= dims[2];
  *z = n % dims[1];
  n /= dims[1];
  *batch = n;
}

inline void GetOutShape(const DDim& x_dims,
                        const std::vector<int>& kernel_sizes,
                        const std::vector<int>& paddings,
                        const std::vector<int>& dilations,
                        const std::vector<int>& strides,
                        DDim* out_dims) {
  const bool is2D = out_dims->size() == 4 ? true : false;
  if (is2D) {
    PADDLE_ENFORCE_EQ(x_dims.size(),
                      4,
                      common::errors::InvalidArgument(
                          "the shape of x should be (N, H, W, C)"));
    PADDLE_ENFORCE_EQ(kernel_sizes.size(),
                      4,
                      common::errors::InvalidArgument(
                          "the shape of kernel should be (H, W, C, OC)"));

    // infer out shape
    (*out_dims)[0] = x_dims[0];
    (*out_dims)[3] = kernel_sizes[3];
    for (int i = 1; i < 3; i++) {
      (*out_dims)[i] = (x_dims[i] + 2 * paddings[i - 1] -
                        dilations[i - 1] * (kernel_sizes[i - 1] - 1) - 1) /
                           strides[i - 1] +
                       1;
    }
  } else {
    PADDLE_ENFORCE_EQ(x_dims.size(),
                      5,
                      common::errors::InvalidArgument(
                          "the shape of x should be (N, D, H, W, C)"));
    PADDLE_ENFORCE_EQ(kernel_sizes.size(),
                      5,
                      common::errors::InvalidArgument(
                          "the shape of kernel should be (D, H, W, C, OC)"));

    // infer out shape
    (*out_dims)[0] = x_dims[0];
    (*out_dims)[4] = kernel_sizes[4];
    for (int i = 1; i < 4; i++) {
      (*out_dims)[i] = (x_dims[i] + 2 * paddings[i - 1] -
                        dilations[i - 1] * (kernel_sizes[i - 1] - 1) - 1) /
                           strides[i - 1] +
                       1;
    }
  }
}

inline void ResetSubmKernelSizeAndStrides(const DDim& kernel_dims,
                                          std::vector<int>* paddings,
                                          std::vector<int>* strides) {
  for (uint64_t i = 0; i < paddings->size(); i++) {
    (*paddings)[i] = kernel_dims[i] / 2;
    (*strides)[i] = 1;
  }
}

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

inline const std::vector<int> PoolResetKernel(
    const std::vector<int>& kernel_sizes,
    const int in_channels,
    const int out_channels) {
  std::vector<int> res(kernel_sizes);
  res.resize(5);
  res[3] = in_channels;
  res[4] = out_channels;
  return res;
}

template <typename T>
inline void PrefixSum(const T* counter, T* offsets, const int n) {
  T offset = 0;
  for (int i = 0; i < n; i++) {
    offsets[i] = offset;
    offset += counter[i];
  }
  offsets[n] = offset;
}

template <typename IntT>
inline const IntT* GetRulebookPtr(const SparseCooTensor& coo,
                                  const DenseTensor& rulebook,
                                  const std::string& key,
                                  int* rulebook_len) {
  if (!key.empty()) {
    const auto* indices_pairs = coo.IndicesPairs(key);
    if (indices_pairs != nullptr) {
      const DenseTensor& tmp_rulebook = indices_pairs->first;
      *rulebook_len = tmp_rulebook.dims()[1];
      return tmp_rulebook.data<IntT>();
    }
  }
  *rulebook_len = rulebook.dims()[1];
  return rulebook.data<IntT>();
}

inline const int* GetCounterPtr(const SparseCooTensor& coo,
                                const DenseTensor& counter,
                                const std::string& key) {
  if (!key.empty()) {
    const auto* indices_pairs = coo.IndicesPairs(key);
    if (indices_pairs != nullptr) {
      return indices_pairs->second.data<int>();
    }
  }
  return counter.data<int>();
}

template <typename T, typename IntT, typename Context>
inline const IntT* PrepareSubm(const Context& dev_ctx,
                               const SparseCooTensor& x,
                               const std::string& key,
                               const DDim& out_dims,
                               SparseCooTensor* out,
                               int* counter,
                               int* offsets,
                               int* rulebook_len,
                               bool* need_product_rulebook) {
  const auto* indices_pairs = x.IndicesPairs(key);
  if (indices_pairs != nullptr) {
    *need_product_rulebook = false;
    const DenseTensor& rulebook = indices_pairs->first;
    const int counter_size = indices_pairs->second.numel();
    memcpy(
        counter, indices_pairs->second.data<int>(), counter_size * sizeof(int));
    out->SetIndicesDict(x.GetIndicesDict());

    *rulebook_len = rulebook.dims()[1];

    DenseTensor out_indices =
        phi::EmptyLike<IntT>(dev_ctx, x.non_zero_indices());
    DenseTensor out_values = phi::EmptyLike<T>(dev_ctx, x.non_zero_elements());
    phi::Copy(
        dev_ctx, x.non_zero_indices(), dev_ctx.GetPlace(), false, &out_indices);
    out->SetMember(out_indices, out_values, out_dims, false);
    PrefixSum<int>(counter, offsets, counter_size);
    return rulebook.data<IntT>();
  }
  return nullptr;
}

template <typename Context>
inline void SaveToTable(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        const std::string& key,
                        const DenseTensor& in_rulebook,
                        const DenseTensor& h_counter,
                        SparseCooTensor* out,
                        DenseTensor* out_rulebook,
                        DenseTensor* counter) {
  out->SetIndicesDict(x.GetIndicesDict());
  if (!key.empty()) {
    out->SaveIndicesPairs(key, std::make_pair(in_rulebook, h_counter));
  } else {
    *out_rulebook = in_rulebook;
    counter->Resize({h_counter.numel()});
    int* counter_ptr = dev_ctx.template HostAlloc<int>(counter);
    memcpy(counter_ptr, h_counter.data<int>(), h_counter.numel() * sizeof(int));
  }
}

}  // namespace sparse
}  // namespace funcs
}  // namespace phi
