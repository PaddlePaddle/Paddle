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

#include <math.h>  // for sqrt in CPU and CUDA
#include <vector>

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/funcs/adam_functors.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

template <typename T, typename MT>
__global__ void AdamKernelREG(MT beta1,
                              MT beta2,
                              MT epsilon,
                              MT beta1_pow_,
                              MT beta2_pow_,
                              const MT* moment1,
                              MT* moment1_out,
                              const MT* moment2,
                              MT* moment2_out,
                              const MT* lr_,
                              const T* grad,
                              const T* param,
                              T* param_out,
                              const MT* master_param,
                              MT* master_param_out,
                              int ndim) {
  MT lr = *lr_;
  MT beta1_pow = beta1_pow_;
  MT beta2_pow = beta2_pow_;

  int id = blockIdx.x * blockDim.x + threadIdx.x;

  for (; id < ndim; id += gridDim.x * blockDim.x) {
    MT p = master_param ? master_param[id] : static_cast<MT>(param[id]);
    MT g = static_cast<MT>(grad[id]);
    MT mom1 = static_cast<MT>(moment1[id]);
    MT mom2 = static_cast<MT>(moment2[id]);
    mom1 = beta1 * mom1 + (static_cast<MT>(1.0) - beta1) * g;
    mom2 = beta2 * mom2 + (static_cast<MT>(1.0) - beta2) * g * g;

    MT denom = (sqrt(mom2) / sqrt(static_cast<MT>(1.0) - beta2_pow)) + epsilon;
    p += (mom1 / denom) * (-(lr / (static_cast<MT>(1.0) - beta1_pow)));

    moment1_out[id] = mom1;
    moment2_out[id] = mom2;
    param_out[id] = static_cast<T>(p);
    if (master_param_out) {
      master_param_out[id] = p;
    }
  }
}

template <typename T, typename MT>
__global__ void AdamKernelMEM(MT beta1,
                              MT beta2,
                              MT epsilon,
                              const MT* beta1_pow_,
                              const MT* beta2_pow_,
                              const MT* moment1,
                              MT* moment1_out,
                              const MT* moment2,
                              MT* moment2_out,
                              const MT* lr_,
                              const T* grad,
                              const T* param,
                              T* param_out,
                              const MT* master_param,
                              MT* master_param_out,
                              int ndim) {
  MT lr = *lr_;
  MT beta1_pow = *beta1_pow_;
  MT beta2_pow = *beta2_pow_;

  int id = blockIdx.x * blockDim.x + threadIdx.x;

  for (; id < ndim; id += gridDim.x * blockDim.x) {
    MT p = master_param ? master_param[id] : static_cast<MT>(param[id]);
    MT g = static_cast<MT>(grad[id]);
    MT mom1 = static_cast<MT>(moment1[id]);
    MT mom2 = static_cast<MT>(moment2[id]);
    mom1 = beta1 * mom1 + (static_cast<MT>(1.0) - beta1) * g;
    mom2 = beta2 * mom2 + (static_cast<MT>(1.0) - beta2) * g * g;

    MT denom = (sqrt(mom2) / sqrt(static_cast<MT>(1.0) - beta2_pow)) + epsilon;
    p += (mom1 / denom) * (-(lr / (static_cast<MT>(1.0) - beta1_pow)));

    moment1_out[id] = mom1;
    moment2_out[id] = mom2;
    param_out[id] = static_cast<T>(p);
    if (master_param_out) {
      master_param_out[id] = p;
    }
  }
}

template <typename T>
__global__ void UpdateBetaPow(T beta1,
                              T beta2,
                              const T* beta1_pow_,
                              const T* beta2_pow_,
                              T* beta1_pow_out,
                              T* beta2_pow_out) {
  *beta1_pow_out = beta1 * beta1_pow_[0];
  *beta2_pow_out = beta2 * beta2_pow_[0];
}

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
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;

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

  MPDType beta1_ = beta1.to<MPDType>();
  MPDType beta2_ = beta2.to<MPDType>();
  MPDType epsilon_ = epsilon.to<MPDType>();
  VLOG(3) << "beta1_pow.numel() : " << beta1_pow.numel()
          << "beta2_pow.numel() : " << beta2_pow.numel();
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

  const MPDType* master_in_data =
      multi_precision ? master_param->data<MPDType>() : nullptr;
  MPDType* master_out_data =
      multi_precision ? dev_ctx.template Alloc<MPDType>(master_param_outs)
                      : nullptr;

  // update param and moment
  int threads = 512;
  int blocks = (param.numel() + threads - 1) / threads;

  if (beta1_pow.place() == CPUPlace() && beta2_pow.place() == CPUPlace()) {
    // Compute with betapow in REG
    AdamKernelREG<T, MPDType><<<blocks, threads, 0, dev_ctx.stream()>>>(
        beta1_,
        beta2_,
        epsilon_,
        *beta1_pow.data<MPDType>(),
        *beta2_pow.data<MPDType>(),
        moment1.data<MPDType>(),
        dev_ctx.template Alloc<MPDType>(moment1_out),
        moment2.data<MPDType>(),
        dev_ctx.template Alloc<MPDType>(moment2_out),
        learning_rate.data<MPDType>(),
        grad.data<T>(),
        param.data<T>(),
        dev_ctx.template Alloc<T>(param_out),
        master_in_data,
        master_out_data,
        param.numel());
    if (!use_global_beta_pow) {
      // Cpu update
      dev_ctx.template HostAlloc<MPDType>(beta1_pow_out)[0] =
          beta1_ * beta1_pow.data<MPDType>()[0];
      dev_ctx.template HostAlloc<MPDType>(beta2_pow_out)[0] =
          beta2_ * beta2_pow.data<MPDType>()[0];
    }
  } else {
    AdamKernelMEM<T, MPDType><<<blocks, threads, 0, dev_ctx.stream()>>>(
        beta1_,
        beta2_,
        epsilon_,
        beta1_pow.data<MPDType>(),
        beta2_pow.data<MPDType>(),
        moment1.data<MPDType>(),
        dev_ctx.template Alloc<MPDType>(moment1_out),
        moment2.data<MPDType>(),
        dev_ctx.template Alloc<MPDType>(moment2_out),
        learning_rate.data<MPDType>(),
        grad.data<T>(),
        param.data<T>(),
        dev_ctx.template Alloc<T>(param_out),
        master_in_data,
        master_out_data,
        param.numel());
    if (!use_global_beta_pow) {
      // Update with gpu
      UpdateBetaPow<MPDType><<<1, 32, 0, dev_ctx.stream()>>>(
          beta1_,
          beta2_,
          beta1_pow.data<MPDType>(),
          beta2_pow.data<MPDType>(),
          dev_ctx.template Alloc<MPDType>(beta1_pow_out),
          dev_ctx.template Alloc<MPDType>(beta2_pow_out));
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(adam,
                   GPU,
                   ALL_LAYOUT,
                   phi::AdamDenseKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  // Skip beta1_pow, beta2_pow, skip_update data transform
  kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(8).SetBackend(phi::Backend::ALL_BACKEND);
}
