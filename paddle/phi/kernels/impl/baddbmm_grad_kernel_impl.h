/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

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

#include <type_traits>

#include "glog/logging.h"

#include "paddle/common/flags.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/kernels/baddbmm_grad_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

COMMON_DECLARE_bool(use_accuracy_compatible_kernel);

namespace phi {

template <typename T>
struct BCopyOrScaleFunctor {
  BCopyOrScaleFunctor(const double scale, const T* x, T* output, int64_t numel)
      : scale_(scale), x_(x), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    using MPType = typename dtype::MPTypeTrait<T>::Type;
    if (x_ == nullptr) {
      output_[idx] = static_cast<T>(0);
      return;
    }
    const MPType mp_scale = static_cast<MPType>(scale_);
    const MPType mp_x = static_cast<MPType>(x_[idx]);
    output_[idx] = static_cast<T>(mp_scale * mp_x);
  }

 private:
  const double scale_;
  const T* x_;
  T* output_;
  int64_t numel_;
};

template <typename T, typename Context>
void BaddbmmGradKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       const DenseTensor& out_grad,
                       double alpha,
                       double beta,
                       DenseTensor* input_grad,
                       DenseTensor* x_grad,
                       DenseTensor* y_grad) {
  using MPType = typename dtype::MPTypeTrait<T>::Type;

  auto input_dims = input.dims();
  auto in_dims = input_dims;
  if (input.dims().size() == 2) {
    in_dims = {input.dims()[0], 1, input.dims()[1]};
    if (input_grad) {
      input_grad->Resize(in_dims);
    }
  }
  int64_t total_elems = 0;

  VLOG(3) << "alpha: " << alpha << " beta: " << beta;

  if (input_grad != nullptr) {
    input_grad->set_lod(out_grad.lod());
  }
  if (x_grad != nullptr) {
    x_grad->set_lod(x.lod());
  }
  if (y_grad != nullptr) {
    y_grad->set_lod(y.lod());
  }

  auto blas = funcs::GetBlas<Context, T>(dev_ctx);
  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);
    total_elems = in_dims[0] * in_dims[1] * in_dims[2];
    bool batch_compress = in_dims[0] != out_grad.dims()[0];
    bool row_compress = in_dims[1] != out_grad.dims()[1];
    bool col_compress = in_dims[2] != out_grad.dims()[2];
    std::vector<int64_t> reduce_dims;
    if (batch_compress) {
      reduce_dims.push_back(0);
    }
    if (row_compress) {
      reduce_dims.push_back(1);
    }
    if (col_compress) {
      reduce_dims.push_back(2);
    }

    if (out_grad.numel() == 0) {
      funcs::ForRange<Context> for_range(dev_ctx, total_elems);
      BCopyOrScaleFunctor<T> functor(
          0, nullptr, input_grad->data<T>(), total_elems);
      for_range(functor);
    } else if (!reduce_dims.empty()) {
      SumKernel<T, Context>(dev_ctx,
                            out_grad,
                            IntArray(reduce_dims),
                            out_grad.dtype(),
                            true,
                            input_grad);
    } else {
      funcs::ForRange<Context> for_range(dev_ctx, total_elems);
      BCopyOrScaleFunctor<T> functor(
          1, out_grad.data<T>(), input_grad->data<T>(), total_elems);
      for_range(functor);
    }

    funcs::ForRange<Context> for_range(dev_ctx, total_elems);
    BCopyOrScaleFunctor<T> functor(
        beta, input_grad->data<T>(), input_grad->data<T>(), total_elems);
    for_range(functor);
    if (input.dims().size() == 2) {
      input_grad->Resize(input_dims);
    }
  }
  if (x_grad) {
    dev_ctx.template Alloc<T>(x_grad);
    total_elems = x.dims()[0] * x.dims()[1] * x.dims()[2];
    // x_grad = alpha * out_grad @ y^T
    // out_grad: [B, M, N], y: [B, K, N], x_grad: [B, M, K]
    int64_t B_dim = x.dims()[0];
    int64_t M_dim = x.dims()[1];
    int64_t K_dim = x.dims()[2];
    int64_t N_dim = y.dims()[2];
    if (x_grad->numel() == 0 || N_dim == 0) {
      funcs::ForRange<Context> for_range(dev_ctx, total_elems);
      BCopyOrScaleFunctor<T> functor(
          0, nullptr, x_grad->data<T>(), total_elems);
      for_range(functor);
    } else if constexpr (std::is_same_v<MPType, float>) {
      float gemm_alpha = FLAGS_use_accuracy_compatible_kernel
                             ? 1.0f
                             : static_cast<float>(alpha);
      float zero = 0.0f;
      blas.BatchedGEMM(CblasNoTrans,
                       CblasTrans,
                       M_dim,
                       K_dim,
                       N_dim,
                       gemm_alpha,
                       out_grad.data<T>(),
                       y.data<T>(),
                       zero,
                       x_grad->data<T>(),
                       B_dim,
                       M_dim * N_dim,
                       K_dim * N_dim);
    } else {
      T gemm_alpha = FLAGS_use_accuracy_compatible_kernel
                         ? static_cast<T>(1)
                         : static_cast<T>(alpha);
      T zero = static_cast<T>(0);
      blas.BatchedGEMM(CblasNoTrans,
                       CblasTrans,
                       M_dim,
                       K_dim,
                       N_dim,
                       gemm_alpha,
                       out_grad.data<T>(),
                       y.data<T>(),
                       zero,
                       x_grad->data<T>(),
                       B_dim,
                       M_dim * N_dim,
                       K_dim * N_dim);
    }
    if (FLAGS_use_accuracy_compatible_kernel) {
      funcs::ForRange<Context> for_range(dev_ctx, total_elems);
      BCopyOrScaleFunctor<T> functor(
          alpha, x_grad->data<T>(), x_grad->data<T>(), total_elems);
      for_range(functor);
    }
  }
  if (y_grad) {
    dev_ctx.template Alloc<T>(y_grad);
    total_elems = y.dims()[0] * y.dims()[1] * y.dims()[2];
    // y_grad = alpha * x^T @ out_grad
    // x: [B, M, K], out_grad: [B, M, N], y_grad: [B, K, N]
    int64_t B_dim = x.dims()[0];
    int64_t M_dim = x.dims()[1];
    int64_t K_dim = x.dims()[2];
    int64_t N_dim = y.dims()[2];
    if (y_grad->numel() == 0 || M_dim == 0) {
      funcs::ForRange<Context> for_range(dev_ctx, total_elems);
      BCopyOrScaleFunctor<T> functor(
          0, nullptr, y_grad->data<T>(), total_elems);
      for_range(functor);
    } else if constexpr (std::is_same_v<MPType, float>) {
      float gemm_alpha = FLAGS_use_accuracy_compatible_kernel
                             ? 1.0f
                             : static_cast<float>(alpha);
      float zero = 0.0f;
      blas.BatchedGEMM(CblasTrans,
                       CblasNoTrans,
                       K_dim,
                       N_dim,
                       M_dim,
                       gemm_alpha,
                       x.data<T>(),
                       out_grad.data<T>(),
                       zero,
                       y_grad->data<T>(),
                       B_dim,
                       M_dim * K_dim,
                       M_dim * N_dim);
    } else {
      T gemm_alpha = FLAGS_use_accuracy_compatible_kernel
                         ? static_cast<T>(1)
                         : static_cast<T>(alpha);
      T zero = static_cast<T>(0);
      blas.BatchedGEMM(CblasTrans,
                       CblasNoTrans,
                       K_dim,
                       N_dim,
                       M_dim,
                       gemm_alpha,
                       x.data<T>(),
                       out_grad.data<T>(),
                       zero,
                       y_grad->data<T>(),
                       B_dim,
                       M_dim * K_dim,
                       M_dim * N_dim);
    }
    if (FLAGS_use_accuracy_compatible_kernel) {
      funcs::ForRange<Context> for_range(dev_ctx, total_elems);
      BCopyOrScaleFunctor<T> functor(
          alpha, y_grad->data<T>(), y_grad->data<T>(), total_elems);
      for_range(functor);
    }
  }
}

}  // namespace phi
