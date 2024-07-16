
#pragma once

#include "paddle/phi/core/dense_tensor.h"

#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/common/transform.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T>
struct GetIntermediateParams {
  HOSTDEVICE void operator()(size_t i) {
    T trans_out;
    T round_out;
    trans_out = (x_[i] - beta_[i]) / alpha_[i];
    round_out = round(trans_out);
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
    alpha_matrix[i] = mask_n * Qn_ + mask_m * round_out + mask_p * Qp_
                     - mask_m * trans_out;
    alpha_matrix[i] = alpha_matrix[i] * gradient[i];

    beta_matrix[i] = mask_n + mask_p;
    beta_matrix[i] = beta_matrix[i] * gradient[i];

    mask[i] = mask_m;
  }

  const T* x_;
  const T* alpha_;
  const T* beta_;
  const T* gradient;
  const int Qn_;
  const int Qp_;
  T* alpha_matrix;
  T* beta_matrix;
  T* mask;
};

template <typename T, typename Context>
void LsqplusGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& alpha,
                       const DenseTensor& beta,
                       const DenseTensor& g,
                       // const DenseTensor& out,
                       const DenseTensor& out_grad,
                       int Qn,
                       int Qp,
                       DenseTensor* x_grad,
                       DenseTensor* alpha_grad,
                       DenseTensor* beta_grad);

}