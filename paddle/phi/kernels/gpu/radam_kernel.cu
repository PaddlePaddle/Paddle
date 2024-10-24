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

#include "paddle/phi/kernels/radam_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {
template <typename T, typename MT>
__global__ void RAdamGPUKernel(const T* param,
                               const T* grad,
                               const MT* learning_rate,
                               const MT* beta1_pow,
                               const MT* beta2_pow,
                               const MT* rho,
                               const MT* moment1,
                               const MT* moment2,
                               const MT* master_param,
                               MT beta1,
                               MT beta2,
                               MT epsilon,
                               MT rho_inf,
                               int num,
                               T* param_out,
                               MT* beta1_pow_out,
                               MT* beta2_pow_out,
                               MT* rho_out,
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
    MT d_beta1_pow = static_cast<MT>(beta1_pow[index]);
    MT d_beta2_pow = static_cast<MT>(beta2_pow[index]);
    MT d_rho = static_cast<MT>(rho[index]);
    MT d_moment1 = static_cast<MT>(moment1[index]);
    MT d_moment2 = static_cast<MT>(moment2[index]);

    // compute
    MT beta1_pow_scalar = d_beta1_pow * beta1;
    MT beta2_pow_scalar = d_beta2_pow * beta2;
    MT rho_scalar = (d_rho * (beta2 - beta2_pow_scalar) + beta2_pow_scalar) /
                    (static_cast<MT>(1) - beta2_pow_scalar);

    MT m1_out = beta1 * d_moment1 + (static_cast<MT>(1) - beta1) * d_grad;
    MT m2_out =
        beta2 * d_moment2 + (static_cast<MT>(1) - beta2) * d_grad * d_grad;

    MT m1_hat = m1_out / (static_cast<MT>(1) - beta1_pow_scalar);

    MT rho_t = rho_inf - static_cast<MT>(2) * rho_scalar;
    MT p_out = static_cast<MT>(0);

    if (rho_t > static_cast<MT>(5)) {
      MT l_t = std::sqrt((static_cast<MT>(1) - beta2_pow_scalar)) /
               (std::sqrt(m2_out) + epsilon);
      MT r_t = std::sqrt(((rho_t - static_cast<MT>(4)) *
                          (rho_t - static_cast<MT>(2)) * rho_inf) /
                         ((rho_inf - static_cast<MT>(4)) *
                          (rho_inf - static_cast<MT>(2)) * rho_t));

      p_out = d_param - lr_scalar * m1_hat * r_t * l_t;
    } else {
      p_out = d_param - lr_scalar * m1_hat;
    }

    // store
    param_out[index] = static_cast<T>(p_out);
    beta1_pow_out[index] = static_cast<T>(beta1_pow_scalar);
    beta2_pow_out[index] = static_cast<T>(beta2_pow_scalar);
    rho_out[index] = static_cast<T>(rho_scalar);
    moment1_out[index] = static_cast<MT>(m1_out);
    moment2_out[index] = static_cast<MT>(m2_out);

    if (master_param_out) {
      master_param_out[index] = p_out;
    }
  }
}

template <typename T, typename Context>
void RAdamKernel(const Context& dev_ctx,
                 const DenseTensor& param,
                 const DenseTensor& grad,
                 const DenseTensor& learning_rate,
                 const DenseTensor& beta1_pow,
                 const DenseTensor& beta2_pow,
                 const DenseTensor& rho,
                 const DenseTensor& moment1,
                 const DenseTensor& moment2,
                 const paddle::optional<DenseTensor>& master_param,
                 float beta1,
                 float beta2,
                 float epsilon,
                 bool multi_precision,
                 DenseTensor* param_out,
                 DenseTensor* beta1_pow_out,
                 DenseTensor* beta2_pow_out,
                 DenseTensor* rho_out,
                 DenseTensor* moment1_out,
                 DenseTensor* moment2_out,
                 DenseTensor* master_param_out) {
  using MPDType = typename phi::dtype::template MPTypeTrait<T>::Type;
  T* param_out_data = dev_ctx.template Alloc<T>(param_out);

  MPDType* beta1_pow_out_data = dev_ctx.template Alloc<MPDType>(beta1_pow_out);
  MPDType* beta2_pow_out_data = dev_ctx.template Alloc<MPDType>(beta2_pow_out);
  MPDType* rho_out_data = dev_ctx.template Alloc<MPDType>(rho_out);

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

  MPDType rho_inf =
      static_cast<MPDType>(2) / (static_cast<MPDType>(1) - beta2_) -
      static_cast<MPDType>(1);

  int numel = param.numel();
  int block = 512;
  int grid = (param.numel() + block - 1) / block;
  auto stream = dev_ctx.stream();

  RAdamGPUKernel<T, MPDType>
      <<<block, grid, 0, stream>>>(param.data<T>(),
                                   grad.data<T>(),
                                   learning_rate.data<MPDType>(),
                                   beta1_pow.data<MPDType>(),
                                   beta2_pow.data<MPDType>(),
                                   rho.data<MPDType>(),
                                   moment1.data<MPDType>(),
                                   moment2.data<MPDType>(),
                                   master_in_data,
                                   beta1_,
                                   beta2_,
                                   epsilon_,
                                   rho_inf,
                                   numel,
                                   param_out_data,
                                   beta1_pow_out_data,
                                   beta2_pow_out_data,
                                   rho_out_data,
                                   moment1_out_data,
                                   moment2_out_data,
                                   master_out_data);
}
}  // namespace phi

PD_REGISTER_KERNEL(radam,
                   GPU,
                   ALL_LAYOUT,
                   phi::RAdamKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
