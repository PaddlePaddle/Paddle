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

#include "paddle/phi/kernels/adan_kernel.h"

#include <math.h>  // for sqrt in CPU and CUDA

#include <vector>

#include "glog/logging.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

template <typename T, typename TG, typename MT>
// __global__ void AdanKernelMEM(MT beta1,
//                               MT beta2,
//                               MT beta3,
//                               MT epsilon,
//                               MT weight_decay,
//                               MT beta1_pow_,
//                               MT beta2_pow_,
//                               MT beta3_pow_,
//                               bool no_prox,
//                               const MT* moment1,
//                               MT* moment1_out,
//                               const MT* moment2,
//                               MT* moment2_out,
//                               const MT* lr_,
//                               const TG* grad,
//                               const TG* pre_grad,
//                               TG* pre_grad_out,
//                               const T* param,
//                               T* param_out,
//                               const MT* master_param,
//                               MT* master_param_out,
//                               int ndim) {
//   // printf("I am In");
//   // MT lr = *lr_;
//   // MT beta1_pow = beta1_pow_;
//   // MT beta2_pow = beta2_pow_;
//   // MT beta3_pow = beta3_pow_;

//   // int id = blockIdx.x * blockDim.x + threadIdx.x;
//   // for (; id < ndim; id += gridDim.x * blockDim.x) {
//   //   MT p = master_param ? master_param[id] : static_cast<MT>(param[id]);
//   //   MT g = static_cast<MT>(grad[id]);
//   //   MT pre_g = static_cast<MT>(pre_grad[id]);
//   //   MT g_diff = g - pre_g;

//   //   MT mom1 = static_cast<MT>(moment1[id]);
//   //   MT mom2 = static_cast<MT>(moment2[id]);

//   //   mom1 = beta1 * mom1 + (static_cast<MT>(1.0) - beta1) * g;
//   //   mom2 = beta2 * mom2 + (static_cast<MT>(1.0) - beta2) * g_diff;

//   //   MT denom = (sqrt(mom2) / sqrt(static_cast<MT>(1.0) - beta3_pow_)) + epsilon;
//   //   MT update = (mom1 / (1.0 - beta1_pow_) + beta2 * mom2 / (1.0 - beta2_pow_)) / (denom);

//   //   if (no_prox){
//   //     p = p - update * (1 - lr * weight_decay);
//   //     p = p - (update * lr);
//   //   }else
//   //   {
//   //     p = p - (update * lr);
//   //     p = p - update * (1 - lr * weight_decay);
//   //   }
   
//   //   moment1_out[id] = mom1;
//   //   moment2_out[id] = mom2;
//   //   pre_grad_out[id] = g;
//   //   param_out[id] = static_cast<T>(p);
//   //   if (master_param_out) {
//   //     master_param_out[id] = p;
//   //   }
//   // }
// }

__global__ void AdanKernelMEM(int ndim) {
  printf("I am In");
  // MT lr = *lr_;
  // MT beta1_pow = beta1_pow_;
  // MT beta2_pow = beta2_pow_;
  // MT beta3_pow = beta3_pow_;

  // int id = blockIdx.x * blockDim.x + threadIdx.x;
  // for (; id < ndim; id += gridDim.x * blockDim.x) {
  //   MT p = master_param ? master_param[id] : static_cast<MT>(param[id]);
  //   MT g = static_cast<MT>(grad[id]);
  //   MT pre_g = static_cast<MT>(pre_grad[id]);
  //   MT g_diff = g - pre_g;

  //   MT mom1 = static_cast<MT>(moment1[id]);
  //   MT mom2 = static_cast<MT>(moment2[id]);

  //   mom1 = beta1 * mom1 + (static_cast<MT>(1.0) - beta1) * g;
  //   mom2 = beta2 * mom2 + (static_cast<MT>(1.0) - beta2) * g_diff;

  //   MT denom = (sqrt(mom2) / sqrt(static_cast<MT>(1.0) - beta3_pow_)) + epsilon;
  //   MT update = (mom1 / (1.0 - beta1_pow_) + beta2 * mom2 / (1.0 - beta2_pow_)) / (denom);

  //   if (no_prox){
  //     p = p - update * (1 - lr * weight_decay);
  //     p = p - (update * lr);
  //   }else
  //   {
  //     p = p - (update * lr);
  //     p = p - update * (1 - lr * weight_decay);
  //   }
   
  //   moment1_out[id] = mom1;
  //   moment2_out[id] = mom2;
  //   pre_grad_out[id] = g;
  //   param_out[id] = static_cast<T>(p);
  //   if (master_param_out) {
  //     master_param_out[id] = p;
  //   }
  // }
}

template <typename T>
__global__ void UpdateBetaPow(T beta1,
                              T beta2,
                              T beta3,
                              const T* beta1_pow_,
                              const T* beta2_pow_,
                              const T* beta3_pow_,
                              T* beta1_pow_out,
                              T* beta2_pow_out,
                              T* beta3_pow_out
                              ) {
  *beta1_pow_out = beta1 * beta1_pow_[0];
  *beta2_pow_out = beta2 * beta2_pow_[0];
  *beta3_pow_out = beta2 * beta3_pow_[0];
}

template <typename T, typename Context>
void AdanDenseKernel(const Context& dev_ctx,
                     const DenseTensor& param,
                     const DenseTensor& grad,
                     const DenseTensor& learning_rate,
                     const DenseTensor& pre_grad,
                     const DenseTensor& moment1,
                     const DenseTensor& moment2,
                     const DenseTensor& beta1_pow,
                     const DenseTensor& beta2_pow,
                     const DenseTensor& beta3_pow,
                     const paddle::optional<DenseTensor>& master_param,
                     const Scalar& beta1,
                     const Scalar& beta2,
                     const Scalar& beta3,
                     const Scalar& epsilon,
                     const Scalar& weight_decay,
                     bool no_prox,
                     bool multi_precision,
                     bool use_global_beta_pow,
                     DenseTensor* param_out,
                     DenseTensor* pre_grad_out,
                     DenseTensor* moment1_out,
                     DenseTensor* moment2_out,
                     DenseTensor* beta1_pow_out,
                     DenseTensor* beta2_pow_out,
                     DenseTensor* beta3_pow_out,
                     DenseTensor* master_param_outs) {
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;
  const auto grad_type = grad.dtype();

  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;

  MPDType beta1_ = beta1.to<MPDType>();
  MPDType beta2_ = beta2.to<MPDType>();
  MPDType beta3_ = beta3.to<MPDType>();
  MPDType epsilon_ = epsilon.to<MPDType>();
  MPDType weight_decay_ = weight_decay.to<MPDType>();
  VLOG(3) << "beta1_pow.numel() : " << beta1_pow.numel()
          << "beta2_pow.numel() : " << beta2_pow.numel()
          << "beta3_pow.numel() : " << beta3_pow.numel();
  VLOG(3) << "param.numel(): " << param.numel();

  PADDLE_ENFORCE_EQ(
      beta1_pow_out->numel(),
      1,
      errors::InvalidArgument("beta1 pow output size should be 1, but received "
                              "value is:%d.",
                              beta1_pow_out->numel()));
  VLOG(3) << beta1_pow_out->numel();

  PADDLE_ENFORCE_EQ(
      beta2_pow_out->numel(),
      1,
      errors::InvalidArgument("beta2 pow output size should be 1, but received "
                              "value is:%d.",
                              beta2_pow_out->numel()));
  PADDLE_ENFORCE_EQ(
      beta3_pow_out->numel(),
      1,
      errors::InvalidArgument("beta3 pow output size should be 1, but received "
                              "value is:%d.",
                              beta3_pow_out->numel()));

  const MPDType* master_in_data =
      multi_precision ? master_param->data<MPDType>() : nullptr;
  MPDType* master_out_data =
      multi_precision ? dev_ctx.template Alloc<MPDType>(master_param_outs)
                      : nullptr;

  // update param and moment
  int threads = 512;
  int blocks = (param.numel() + threads - 1) / threads;


  VLOG(3) << grad_type <<"-----------------";
  // AdanKernelMEM<T, float, MPDType>
  //     <<<blocks, threads, 0, dev_ctx.stream()>>>(
  //         beta1_,
  //         beta2_,
  //         beta3_,
  //         epsilon_,
  //         weight_decay_,
  //         *beta1_pow.data<MPDType>(),
  //         *beta2_pow.data<MPDType>(),
  //         *beta3_pow.data<MPDType>(),
  //         no_prox,
  //         moment1.data<MPDType>(),
  //         dev_ctx.template Alloc<MPDType>(moment1_out),
  //         moment2.data<MPDType>(),
  //         dev_ctx.template Alloc<MPDType>(moment2_out),
  //         learning_rate.data<MPDType>(),
  //         grad.data<float>(),
  //         pre_grad.data<float>(),
  //         dev_ctx.template Alloc<float>(pre_grad_out),
  //         param.data<T>(),
  //         dev_ctx.template Alloc<T>(param_out),
  //         master_in_data,
  //         master_out_data,
  //         param.numel());

  AdanKernelMEM<T, float, MPDType><<<blocks, threads, 0, dev_ctx.stream()>>>(param.numel());
  
  VLOG(3) << "-----------------";
  if (!use_global_beta_pow) {
    // Update with gpu
    UpdateBetaPow<MPDType><<<1, 1, 0, dev_ctx.stream()>>>(
        beta1_,
        beta2_,
        beta3_,
        beta1_pow.data<MPDType>(),
        beta2_pow.data<MPDType>(),
        beta3_pow.data<MPDType>(),
        dev_ctx.template Alloc<MPDType>(beta1_pow_out),
        dev_ctx.template Alloc<MPDType>(beta2_pow_out),
        dev_ctx.template Alloc<MPDType>(beta3_pow_out));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(adan,
                   GPU,
                   ALL_LAYOUT,
                   phi::AdanDenseKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  // Skip beta1_pow, beta2_pow, skip_update data transform
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(7).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(8).SetBackend(phi::Backend::ALL_BACKEND);

  if (kernel_key.dtype() == phi::DataType::FLOAT16 ||
      kernel_key.dtype() == phi::DataType::BFLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(5).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(6).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(7).SetDataType(phi::DataType::FLOAT32);
  }
  kernel->OutputAt(4).SetBackend(phi::Backend::UNDEFINED);
  kernel->OutputAt(5).SetBackend(phi::Backend::UNDEFINED);
  kernel->OutputAt(6).SetBackend(phi::Backend::UNDEFINED);
}

