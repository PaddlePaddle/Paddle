// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/adam_kernel.h"

#include <vector>

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/funcs/adam_functors.h"

DECLARE_int32(inner_op_parallelism);

namespace phi {

template <typename T, typename Context>
void AdamDenseKernel(const Context& dev_ctx,
                     const DenseTensor& param,
                     const DenseTensor& grad,
                     const DenseTensor& learning_rate,
                     const DenseTensor& moment1,
                     const DenseTensor& moment2,
                     const DenseTensor& beta1_pow,
                     const DenseTensor& beta2_pow,
                     paddle::optional<const DenseTensor&> master_param,
                     paddle::optional<const DenseTensor&> skip_update,
                     const Scalar& beta1,
                     const Scalar& beta2,
                     const Scalar& epsilon,
                     bool lazy_mode,
                     int64_t min_row_size_to_use_multithread,
                     bool multi_precision,
                     bool use_global_beta_pow,
                     DenseTensor* param_out,
                     DenseTensor* moment1_out,
                     DenseTensor* moment2_out,
                     DenseTensor* beta1_pow_out,
                     DenseTensor* beta2_pow_out,
                     DenseTensor* master_param_outs) {
  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;

  bool skip_update_ = false;
  if (skip_update.is_initialized()) {
    PADDLE_ENFORCE_EQ(
        skip_update->numel(),
        1,
        errors::InvalidArgument("Input(SkipUpdate) size must be 1, but get %d",
                                skip_update->numel()));
    std::vector<bool> skip_update_vec;
    paddle::framework::TensorToVector(*skip_update, dev_ctx, &skip_update_vec);
    skip_update_ = skip_update_vec[0];
  }
  // skip_update=true, just copy input to output, and TensorCopy will call
  // mutable_data
  if (skip_update_) {
    VLOG(4) << "Adam skip update";
    phi::Copy(dev_ctx, param, dev_ctx.GetPlace(), false, param_out);
    phi::Copy(dev_ctx, moment1, dev_ctx.GetPlace(), false, moment1_out);
    phi::Copy(dev_ctx, moment2, dev_ctx.GetPlace(), false, moment2_out);
    phi::Copy(dev_ctx, beta1_pow, beta1_pow.place(), false, beta1_pow_out);
    phi::Copy(dev_ctx, beta2_pow, beta2_pow.place(), false, beta2_pow_out);

    return;
  }

  T beta1_ = beta1.to<T>();
  T beta2_ = beta2.to<T>();
  T epsilon_ = epsilon.to<T>();

  VLOG(3) << "beta1_pow.numel() : " << beta1_pow.numel();
  VLOG(3) << "beta2_pow.numel() : " << beta2_pow.numel();
  VLOG(3) << "param.numel(): " << param.numel();

  PADDLE_ENFORCE_EQ(
      beta1_pow_out->numel(),
      1,
      errors::InvalidArgument("beta1 pow output size should be 1, but received "
                              "value is:%d.",
                              beta1_pow_out->numel()));

  PADDLE_ENFORCE_EQ(
      beta2_pow_out->numel(),
      1,
      errors::InvalidArgument("beta2 pow output size should be 1, but received "
                              "value is:%d.",
                              beta2_pow_out->numel()));

  T beta1_p = beta1_pow.data<T>()[0];
  T beta2_p = beta2_pow.data<T>()[0];

  if (!use_global_beta_pow) {
    dev_ctx.template Alloc<T>(beta1_pow_out)[0] = beta1_ * beta1_p;
    dev_ctx.template Alloc<T>(beta2_pow_out)[0] = beta2_ * beta2_p;
  }

  T* param_out_ptr = dev_ctx.template Alloc<T>(param_out);
  T* mom1_out_ptr = dev_ctx.template Alloc<T>(moment1_out);
  T* mom2_out_ptr = dev_ctx.template Alloc<T>(moment2_out);

  T learning_rate_ =
      learning_rate.data<T>()[0] * (sqrt(1 - beta2_p) / (1 - beta1_p));
  T eps = epsilon_ * sqrt(1 - beta2_p);

  paddle::operators::jit::adam_attr_t attr(beta1_, beta2_);
  int64_t numel = param.numel();

  const T* param_ptr = param.data<T>();
  const T* mom1_ptr = moment1.data<T>();
  const T* mom2_ptr = moment2.data<T>();
  const T* grad_ptr = grad.data<T>();

  auto adam =
      paddle::operators::jit::KernelFuncs<paddle::operators::jit::AdamTuple<T>,
                                          phi::CPUPlace>::Cache()
          .At(attr);

  static constexpr int64_t chunk_size = 512;

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < numel / chunk_size; ++i) {
    const int64_t offset = i * chunk_size;
    adam(beta1_,
         beta2_,
         -learning_rate_,
         eps,
         chunk_size,
         grad_ptr + offset,
         mom1_ptr + offset,
         mom2_ptr + offset,
         param_ptr + offset,
         mom1_out_ptr + offset,
         mom2_out_ptr + offset,
         param_out_ptr + offset);
  }

  if (numel % chunk_size != 0) {
    const int64_t offset = (numel / chunk_size) * chunk_size;
    const int64_t tail_numel = numel % chunk_size;
    adam(beta1_,
         beta2_,
         -learning_rate_,
         eps,
         tail_numel,
         grad_ptr + offset,
         mom1_ptr + offset,
         mom2_ptr + offset,
         param_ptr + offset,
         mom1_out_ptr + offset,
         mom2_out_ptr + offset,
         param_out_ptr + offset);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(adam, CPU, ALL_LAYOUT, phi::AdamDenseKernel, float, double) {
}
