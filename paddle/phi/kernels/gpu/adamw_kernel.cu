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

#include "paddle/phi/kernels/adamw_kernel.h"

#include <math.h>  // for sqrt in CPU and CUDA

#include <vector>

#include "glog/logging.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/adam_functors.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/selected_rows_functor.h"

namespace phi {
template <typename T, typename TG, typename MT>
__global__ void AdamWKernelREG(MT beta1,
                               MT beta2,
                               MT epsilon,
                               MT coeff,
                               MT lr_ratio,
                               MT beta1_pow_,
                               MT beta2_pow_,
                               const MT* moment1,
                               MT* moment1_out,
                               const MT* moment2,
                               MT* moment2_out,
                               const MT* moment2_max,
                               MT* moment2_max_out,
                               const MT* lr_,
                               const TG* grad,
                               const T* param,
                               T* param_out,
                               const MT* master_param,
                               MT* master_param_out,
                               int64_t ndim,
                               bool amsgrad) {
  MT lr = *lr_ * lr_ratio;
  MT beta1_pow = beta1_pow_;
  MT beta2_pow = beta2_pow_;

  int64_t id = blockIdx.x * blockDim.x + threadIdx.x;

  for (; id < ndim; id += gridDim.x * blockDim.x) {
    MT p = master_param ? master_param[id] : static_cast<MT>(param[id]);
    MT g = static_cast<MT>(grad[id]);
    MT mom1 = static_cast<MT>(moment1[id]);
    MT mom2 = static_cast<MT>(moment2[id]);

    p *= (static_cast<MT>(1.0) - lr * coeff);

    mom1 = beta1 * mom1 + (static_cast<MT>(1.0) - beta1) * g;
    mom2 = beta2 * mom2 + (static_cast<MT>(1.0) - beta2) * g * g;

    MT denom;
    if (amsgrad) {
      MT mom2_max = static_cast<MT>(moment2_max[id]);
      MT mom2_max_ = std::max(mom2, mom2_max);
      moment2_max_out[id] = mom2_max_;

      denom =
          (sqrt(mom2_max_) / sqrt(static_cast<MT>(1.0) - beta2_pow)) + epsilon;
    } else {
      denom = (sqrt(mom2) / sqrt(static_cast<MT>(1.0) - beta2_pow)) + epsilon;
    }

    p += (mom1 / denom) * (-(lr / (static_cast<MT>(1.0) - beta1_pow)));

    moment1_out[id] = mom1;
    moment2_out[id] = mom2;
    param_out[id] = static_cast<T>(p);
    if (master_param_out) {
      master_param_out[id] = p;
    }
  }
}

template <typename T, typename TG, typename MT>
__global__ void AdamWKernelMEM(MT beta1,
                               MT beta2,
                               MT epsilon,
                               MT coeff,
                               MT lr_ratio,
                               const MT* beta1_pow_,
                               const MT* beta2_pow_,
                               const MT* moment1,
                               MT* moment1_out,
                               const MT* moment2,
                               MT* moment2_out,
                               const MT* moment2_max,
                               MT* moment2_max_out,
                               const MT* lr_,
                               const TG* grad,
                               const T* param,
                               T* param_out,
                               const MT* master_param,
                               MT* master_param_out,
                               int64_t ndim,
                               bool amsgrad) {
  MT lr = *lr_ * lr_ratio;
  MT beta1_pow = *beta1_pow_;
  MT beta2_pow = *beta2_pow_;

  int64_t id = blockIdx.x * blockDim.x + threadIdx.x;

  for (; id < ndim; id += gridDim.x * blockDim.x) {
    MT p = master_param ? master_param[id] : static_cast<MT>(param[id]);
    MT g = static_cast<MT>(grad[id]);
    MT mom1 = static_cast<MT>(moment1[id]);
    MT mom2 = static_cast<MT>(moment2[id]);

    p *= (static_cast<MT>(1.0) - lr * coeff);

    mom1 = beta1 * mom1 + (static_cast<MT>(1.0) - beta1) * g;
    mom2 = beta2 * mom2 + (static_cast<MT>(1.0) - beta2) * g * g;

    MT denom;
    if (amsgrad) {
      MT mom2_max = static_cast<MT>(moment2_max[id]);
      MT mom2_max_ = std::max(mom2, mom2_max);
      moment2_max_out[id] = mom2_max_;

      denom =
          (sqrt(mom2_max_) / sqrt(static_cast<MT>(1.0) - beta2_pow)) + epsilon;
    } else {
      denom = (sqrt(mom2) / sqrt(static_cast<MT>(1.0) - beta2_pow)) + epsilon;
    }

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
__global__ void UpdateAdamWBetaPow(T beta1,
                                   T beta2,
                                   const T* beta1_pow_,
                                   const T* beta2_pow_,
                                   T* beta1_pow_out,
                                   T* beta2_pow_out) {
  *beta1_pow_out = beta1 * beta1_pow_[0];
  *beta2_pow_out = beta2 * beta2_pow_[0];
}

template <typename T, typename Context>
void AdamwDenseKernel(const Context& dev_ctx,
                      const DenseTensor& param,
                      const DenseTensor& grad,
                      const DenseTensor& learning_rate,
                      const DenseTensor& moment1,
                      const DenseTensor& moment2,
                      const paddle::optional<DenseTensor>& moment2_max,
                      const DenseTensor& beta1_pow,
                      const DenseTensor& beta2_pow,
                      const paddle::optional<DenseTensor>& master_param,
                      const paddle::optional<DenseTensor>& skip_update,
                      const Scalar& beta1,
                      const Scalar& beta2,
                      const Scalar& epsilon,
                      float lr_ratio,
                      float coeff,
                      bool with_decay,
                      bool lazy_mode,
                      int64_t min_row_size_to_use_multithread,
                      bool multi_precision,
                      bool use_global_beta_pow,
                      bool amsgrad,
                      DenseTensor* param_out,
                      DenseTensor* moment1_out,
                      DenseTensor* moment2_out,
                      DenseTensor* moment2_max_out,
                      DenseTensor* beta1_pow_out,
                      DenseTensor* beta2_pow_out,
                      DenseTensor* master_param_outs) {
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;

  const auto grad_type = grad.dtype();

  VLOG(4) << "multi_precision: " << multi_precision;
  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;
  VLOG(4) << "amsgrad:" << amsgrad;

  MPDType coeff_ = static_cast<MPDType>(coeff);
  MPDType lr_ratio_ = static_cast<MPDType>(lr_ratio);

  bool skip_update_ = false;
  if (skip_update.is_initialized()) {
    PADDLE_ENFORCE_EQ(
        skip_update->numel(),
        1,
        errors::InvalidArgument("Input(SkipUpdate) size must be 1, but get %d",
                                skip_update->numel()));
    std::vector<bool> skip_update_vec;
    phi::TensorToVector(*skip_update, dev_ctx, &skip_update_vec);
    skip_update_ = skip_update_vec[0];
  }

  // skip_update=true, just copy input to output, and TensorCopy will call
  // mutable_data
  if (skip_update_) {
    VLOG(4) << "Adamw skip update";
    phi::Copy(dev_ctx, param, dev_ctx.GetPlace(), false, param_out);
    phi::Copy(dev_ctx, moment1, dev_ctx.GetPlace(), false, moment1_out);
    phi::Copy(dev_ctx, moment2, dev_ctx.GetPlace(), false, moment2_out);
    if (amsgrad) {
      phi::Copy(dev_ctx,
                moment2_max.get(),
                dev_ctx.GetPlace(),
                false,
                moment2_max_out);
    }
    if (!use_global_beta_pow) {
      phi::Copy(dev_ctx, beta1_pow, beta1_pow.place(), false, beta1_pow_out);
      phi::Copy(dev_ctx, beta2_pow, beta2_pow.place(), false, beta2_pow_out);
    }
    return;
  }

  // if with_decay = false, coeff = 0
  if (!with_decay) {
    coeff_ = static_cast<MPDType>(0.0);
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

  const MPDType* moment2_max_in_data =
      amsgrad ? moment2_max.get().data<MPDType>() : nullptr;
  MPDType* moment2_max_out_data =
      amsgrad ? dev_ctx.template Alloc<MPDType>(moment2_max_out) : nullptr;

  // update param and moment
  int threads = 512;
  int blocks = (param.numel() + threads - 1) / threads;

  if (beta1_pow.place() == CPUPlace() && beta2_pow.place() == CPUPlace()) {
    // Compute with betapow in REG
    if (grad_type == phi::DataType::FLOAT32) {
      AdamWKernelREG<T, float, MPDType>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(
              beta1_,
              beta2_,
              epsilon_,
              coeff_,
              lr_ratio_,
              *beta1_pow.data<MPDType>(),
              *beta2_pow.data<MPDType>(),
              moment1.data<MPDType>(),
              dev_ctx.template Alloc<MPDType>(moment1_out),
              moment2.data<MPDType>(),
              dev_ctx.template Alloc<MPDType>(moment2_out),
              moment2_max_in_data,
              moment2_max_out_data,
              learning_rate.data<MPDType>(),
              grad.data<float>(),
              param.data<T>(),
              dev_ctx.template Alloc<T>(param_out),
              master_in_data,
              master_out_data,
              param.numel(),
              amsgrad);
    } else {
      AdamWKernelREG<T, T, MPDType><<<blocks, threads, 0, dev_ctx.stream()>>>(
          beta1_,
          beta2_,
          epsilon_,
          coeff_,
          lr_ratio_,
          *beta1_pow.data<MPDType>(),
          *beta2_pow.data<MPDType>(),
          moment1.data<MPDType>(),
          dev_ctx.template Alloc<MPDType>(moment1_out),
          moment2.data<MPDType>(),
          dev_ctx.template Alloc<MPDType>(moment2_out),
          moment2_max_in_data,
          moment2_max_out_data,
          learning_rate.data<MPDType>(),
          grad.data<T>(),
          param.data<T>(),
          dev_ctx.template Alloc<T>(param_out),
          master_in_data,
          master_out_data,
          param.numel(),
          amsgrad);
    }
    if (!use_global_beta_pow) {
      // Cpu update
      dev_ctx.template HostAlloc<MPDType>(beta1_pow_out)[0] =
          beta1_ * beta1_pow.data<MPDType>()[0];
      dev_ctx.template HostAlloc<MPDType>(beta2_pow_out)[0] =
          beta2_ * beta2_pow.data<MPDType>()[0];
    }
  } else {
    if (grad_type == phi::DataType::FLOAT32) {
      AdamWKernelMEM<T, float, MPDType>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(
              beta1_,
              beta2_,
              epsilon_,
              coeff_,
              lr_ratio_,
              beta1_pow.data<MPDType>(),
              beta2_pow.data<MPDType>(),
              moment1.data<MPDType>(),
              dev_ctx.template Alloc<MPDType>(moment1_out),
              moment2.data<MPDType>(),
              dev_ctx.template Alloc<MPDType>(moment2_out),
              moment2_max_in_data,
              moment2_max_out_data,
              learning_rate.data<MPDType>(),
              grad.data<float>(),
              param.data<T>(),
              dev_ctx.template Alloc<T>(param_out),
              master_in_data,
              master_out_data,
              param.numel(),
              amsgrad);
    } else {
      AdamWKernelMEM<T, T, MPDType><<<blocks, threads, 0, dev_ctx.stream()>>>(
          beta1_,
          beta2_,
          epsilon_,
          coeff_,
          lr_ratio_,
          beta1_pow.data<MPDType>(),
          beta2_pow.data<MPDType>(),
          moment1.data<MPDType>(),
          dev_ctx.template Alloc<MPDType>(moment1_out),
          moment2.data<MPDType>(),
          dev_ctx.template Alloc<MPDType>(moment2_out),
          moment2_max_in_data,
          moment2_max_out_data,
          learning_rate.data<MPDType>(),
          grad.data<T>(),
          param.data<T>(),
          dev_ctx.template Alloc<T>(param_out),
          master_in_data,
          master_out_data,
          param.numel(),
          amsgrad);
    }
    if (!use_global_beta_pow) {
      // Update with gpu
      UpdateAdamWBetaPow<MPDType><<<1, 1, 0, dev_ctx.stream()>>>(
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

PD_REGISTER_KERNEL(adamw,
                   GPU,
                   ALL_LAYOUT,
                   phi::AdamwDenseKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  // Skip beta1_pow, beta2_pow, skip_update data transform
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(7).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(9).SetBackend(phi::Backend::ALL_BACKEND);

  if (kernel_key.dtype() == phi::DataType::FLOAT16 ||
      kernel_key.dtype() == phi::DataType::BFLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(5).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(6).SetDataType(phi::DataType::FLOAT32);
  }
  kernel->OutputAt(4).SetBackend(phi::Backend::UNDEFINED);
  kernel->OutputAt(5).SetBackend(phi::Backend::UNDEFINED);
}
