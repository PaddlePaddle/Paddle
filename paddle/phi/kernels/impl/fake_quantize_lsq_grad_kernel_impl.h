// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <cmath>
#include "paddle/phi/kernels/fake_quantize_kernel.h"
#include "paddle/phi/kernels/funcs/fake_quantize_functor.h"
// for lsqplus
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/extensions.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

#include "glog/logging.h"

namespace phi {
template <typename T>
struct QuantizeDataType {
  using type = T;
};

template <>
struct QuantizeDataType<phi::dtype::float16> {
  using type = float;
};

// for lsqplus
template <typename T>
struct GetIntermediateParams {
 public:
  GetIntermediateParams(const T* x,
                        const T* alpha,
                        const T* beta,
                        const T* gradient,
                        const int Qn,
                        const int Qp,
                        const int round_type,
                        T* alpha_matrix,
                        T* beta_matrix,
                        T* mask)
      : x_(x),
        alpha_(alpha),
        beta_(beta),
        gradient_(gradient),
        Qn_(Qn),
        Qp_(Qp),
        round_type_(round_type),
        alpha_matrix_(alpha_matrix),
        beta_matrix_(beta_matrix),
        mask_(mask) {}

  HOSTDEVICE void operator()(size_t i) {
    using ComputeDataType = typename QuantizeDataType<T>::type;
    ComputeDataType trans_out;
    ComputeDataType round_out;

    ComputeDataType x = static_cast<ComputeDataType>(x_[i]);
    ComputeDataType alpha = static_cast<ComputeDataType>(alpha_[0]);
    ComputeDataType beta = static_cast<ComputeDataType>(beta_[0]);
    ComputeDataType g = static_cast<ComputeDataType>(gradient_[i]);

    ComputeDataType alpha_mx;
    ComputeDataType beta_mx;

    ComputeDataType inv_alpha = phi::funcs::inverse(alpha);
    trans_out = (x - beta) * inv_alpha;

    if (round_type_ == 0) {
      round_out = phi::funcs::roundWithTiesToEven(trans_out);
    } else {
      round_out = std::round(trans_out);
    }
    int mask_n = 0;
    int mask_m = 0;
    int mask_p = 0;

    if (trans_out < Qn_) {
      mask_n = 1;
      mask_m = 0;
      mask_p = 0;
    } else if (trans_out > Qp_) {
      mask_n = 0;
      mask_m = 0;
      mask_p = 1;
    } else {
      mask_n = 0;
      mask_m = 1;
      mask_p = 0;
    }
    alpha_mx = static_cast<T>((mask_n * Qn_ + mask_m * round_out +
                               mask_p * Qp_ - mask_m * trans_out) *
                              g);
    alpha_matrix_[i] = static_cast<T>(alpha_mx);

    beta_mx = static_cast<T>((mask_n + mask_p) * g);
    beta_matrix_[i] = static_cast<T>(beta_mx);

    mask_[i] = static_cast<T>(mask_m);
  }

 private:
  const T* x_;
  const T* alpha_;
  const T* beta_;
  const T* gradient_;
  const int Qn_;
  const int Qp_;
  const int round_type_;
  T* alpha_matrix_;
  T* beta_matrix_;
  T* mask_;
};

// for lsqplus
template <typename T, typename Context>
void FakeQuantizeDequantizeLsqplusGradKernel(const Context& dev_ctx,
                                             const DenseTensor& x,
                                             const DenseTensor& alpha,
                                             const DenseTensor& beta,
                                             const DenseTensor& g,
                                             const DenseTensor& out_grad,
                                             int bit_length,
                                             bool is_sign,
                                             int round_type,
                                             DenseTensor* x_grad,
                                             DenseTensor* alpha_grad,
                                             DenseTensor* beta_grad) {
  // space for x_grad, alpha_grad and beta_grad
  dev_ctx.template Alloc<T>(x_grad);
  dev_ctx.template Alloc<T>(alpha_grad);
  dev_ctx.template Alloc<T>(beta_grad);

  // intermediate params initialization
  DenseTensor alpha_matrix = FullLike<T, Context>(dev_ctx, x, 0);
  DenseTensor beta_matrix = FullLike<T, Context>(dev_ctx, x, 0);
  DenseTensor mask = FullLike<T, Context>(dev_ctx, x, 0);

  // gradient scaling
  DenseTensor gradient_scale = FullLike<T, Context>(dev_ctx, x, 0);
  MultiplyKernel<T, Context>(dev_ctx, out_grad, g, &gradient_scale);

  // get intermedate params
  int Qn = 0;
  int Qp = 255;
  if (is_sign) {
    Qn = -std::pow(2, bit_length - 1);
    Qp = std::pow(2, bit_length - 1) - 1;
  } else {
    Qn = 0;
    Qp = std::pow(2, bit_length) - 1;
  }
  size_t numel = x.numel();
  phi::funcs::ForRange<Context> for_range(dev_ctx, numel);

  // note: cannot use alpha.data<T>()[0] and beta.data<T>()[0] in GPU version
  // otherwise, the kernel will be killed
  phi::GetIntermediateParams<T> get_intermediate_params(
      x.data<T>(),
      alpha.data<T>(),
      beta.data<T>(),
      gradient_scale.data<T>(),
      Qn,
      Qp,
      round_type,
      alpha_matrix.data<T>(),
      beta_matrix.data<T>(),
      mask.data<T>());

  for_range(get_intermediate_params);

  std::vector<int> v_dims(x.dims().size());
  std::iota(v_dims.begin(), v_dims.end(), 0);
  IntArray v_axes(v_dims);
  // get alpha_grad
  SumKernel<T, Context>(
      dev_ctx, alpha_matrix, v_axes, x.dtype(), 0, alpha_grad);
  alpha_grad->Resize(alpha.dims());

  // get beta_grad
  SumKernel<T, Context>(dev_ctx, beta_matrix, v_axes, x.dtype(), 0, beta_grad);
  beta_grad->Resize(beta.dims());

  // get x_grad
  MultiplyKernel<T, Context>(dev_ctx, mask, out_grad, x_grad);
}

}  // namespace phi
