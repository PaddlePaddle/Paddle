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

#include "paddle/phi/kernels/nadam_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {
template <typename T, typename MT>
__global__ void NAdamGPUKernel(const T* param,
                               const T* grad,
                               const MT* learning_rate,
                               const MT* moment1,
                               const MT* moment2,
                               const MT* master_param,
                               MT beta1,
                               MT beta2,
                               MT epsilon,
                               MT beta2_pow_scalar,
                               MT mu_t_scalar,
                               MT mu_t_1_scalar,
                               MT mu_product_scalar,
                               MT mu_product_t_1_scalar,
                               int num,
                               T* param_out,
                               MT* moment1_out,
                               MT* moment2_out,
                               MT* master_param_out) {
  MT lr_scalar = static_cast<MT>(learning_rate[0]);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int index = idx; index < num; index += gridDim.x * blockDim.x) {
    // load and cast input to MT
    MT d_param =
        master_param ? master_param[index] : static_cast<MT>(param[index]);
    MT d_grad = static_cast<MT>(grad[index]);
    MT d_moment1 = static_cast<MT>(moment1[index]);
    MT d_moment2 = static_cast<MT>(moment2[index]);
    // compute
    MT m1_out = beta1 * d_moment1 + (static_cast<MT>(1) - beta1) * d_grad;
    MT m2_out =
        beta2 * d_moment2 + (static_cast<MT>(1) - beta2) * d_grad * d_grad;

    MT m1_hat =
        mu_t_1_scalar * m1_out / (static_cast<MT>(1) - mu_product_t_1_scalar) +
        (static_cast<MT>(1) - mu_t_scalar) * d_grad /
            (static_cast<MT>(1) - mu_product_scalar);
    MT m2_hat = m2_out / (static_cast<MT>(1) - beta2_pow_scalar);

    MT p_out = d_param - lr_scalar * m1_hat / (std::sqrt(m2_hat) + epsilon);
    // store
    param_out[index] = static_cast<T>(p_out);
    moment1_out[index] = static_cast<MT>(m1_out);
    moment2_out[index] = static_cast<MT>(m2_out);

    if (master_param_out) {
      master_param_out[index] = p_out;
    }
  }
}

template <typename T, typename Context>
void NAdamKernel(const Context& dev_ctx,
                 const DenseTensor& param,
                 const DenseTensor& grad,
                 const DenseTensor& learning_rate,
                 const DenseTensor& momentum_decay_pow,
                 const DenseTensor& beta2_pow,
                 const DenseTensor& mu_product,
                 const DenseTensor& moment1,
                 const DenseTensor& moment2,
                 const paddle::optional<DenseTensor>& master_param,
                 float beta1,
                 float beta2,
                 float epsilon,
                 float momentum_decay,
                 bool multi_precision,
                 DenseTensor* param_out,
                 DenseTensor* momentum_decay_pow_out,
                 DenseTensor* beta2_pow_out,
                 DenseTensor* mu_product_out,
                 DenseTensor* moment1_out,
                 DenseTensor* moment2_out,
                 DenseTensor* master_param_out) {
  using MPDType = typename phi::dtype::template MPTypeTrait<T>::Type;
  T* param_out_data = dev_ctx.template Alloc<T>(param_out);

  MPDType* momentum_decay_pow_out_data =
      dev_ctx.template HostAlloc<MPDType>(momentum_decay_pow_out);
  MPDType* beta2_pow_out_data =
      dev_ctx.template HostAlloc<MPDType>(beta2_pow_out);
  MPDType* mu_product_out_data =
      dev_ctx.template HostAlloc<MPDType>(mu_product_out);

  MPDType* moment1_out_data = dev_ctx.template Alloc<MPDType>(moment1_out);
  MPDType* moment2_out_data = dev_ctx.template Alloc<MPDType>(moment2_out);

  const MPDType* master_in_data =
      multi_precision ? master_param->data<MPDType>() : nullptr;
  MPDType* master_out_data =
      multi_precision ? dev_ctx.template Alloc<MPDType>(master_param_out)
                      : nullptr;

  MPDType beta1_ = static_cast<MPDType>(beta1);
  MPDType beta2_ = static_cast<MPDType>(beta2);
  MPDType epsilon_ = static_cast<MPDType>(epsilon);
  MPDType momentum_decay_ = static_cast<MPDType>(momentum_decay);

  // make cpu accumulator to tensor
  DenseTensor momentum_decay_pow_data;
  phi::Copy(dev_ctx,
            momentum_decay_pow,
            phi::CPUPlace(),
            false,
            &momentum_decay_pow_data);
  MPDType momentum_decay_pow_scalar =
      momentum_decay_pow_data.data<MPDType>()[0] * static_cast<MPDType>(0.96);
  momentum_decay_pow_out_data[0] = momentum_decay_pow_scalar;

  DenseTensor beta2_pow_data;
  phi::Copy(dev_ctx, beta2_pow, phi::CPUPlace(), false, &beta2_pow_data);
  MPDType beta2_pow_scalar = beta2_pow_data.data<MPDType>()[0] * beta2_;
  beta2_pow_out_data[0] = beta2_pow_scalar;

  MPDType mu_t_scalar =
      beta1_ * (static_cast<MPDType>(1) -
                static_cast<MPDType>(0.5) *
                    std::pow(momentum_decay_pow_scalar, momentum_decay_));
  MPDType mu_t_1_scalar =
      beta1_ * (static_cast<MPDType>(1) -
                static_cast<MPDType>(0.5) *
                    std::pow(momentum_decay_pow_scalar, momentum_decay_) *
                    std::pow(static_cast<MPDType>(0.96), momentum_decay_));

  DenseTensor mu_product_data;
  phi::Copy(dev_ctx, mu_product, phi::CPUPlace(), false, &mu_product_data);
  MPDType mu_product_scalar = mu_product_data.data<MPDType>()[0] * mu_t_scalar;
  MPDType mu_product_t_1_scalar = mu_product_scalar * mu_t_1_scalar;
  mu_product_out_data[0] = mu_product_scalar;

  int numel = param.numel();
  int block = 512;
  int grid = (param.numel() + block - 1) / block;
  auto stream = dev_ctx.stream();

  NAdamGPUKernel<T, MPDType>
      <<<block, grid, 0, stream>>>(param.data<T>(),
                                   grad.data<T>(),
                                   learning_rate.data<MPDType>(),
                                   moment1.data<MPDType>(),
                                   moment2.data<MPDType>(),
                                   master_in_data,
                                   beta1_,
                                   beta2_,
                                   epsilon_,
                                   beta2_pow_scalar,
                                   mu_t_scalar,
                                   mu_t_1_scalar,
                                   mu_product_scalar,
                                   mu_product_t_1_scalar,
                                   numel,
                                   param_out_data,
                                   moment1_out_data,
                                   moment2_out_data,
                                   master_out_data);
}
}  // namespace phi

PD_REGISTER_KERNEL(nadam,
                   GPU,
                   ALL_LAYOUT,
                   phi::NAdamKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
