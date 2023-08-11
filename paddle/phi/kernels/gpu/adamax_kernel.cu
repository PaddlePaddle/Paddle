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

#include "paddle/phi/kernels/adamax_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
namespace phi {
template <typename T, typename MT>
__global__ void AdamaxGPUKernel(const T* param,
                                const T* grad,
                                const MT* learning_rate,
                                const MT* moment,
                                const MT* inf_norm,
                                const MT* beta1_pow,
                                const MT* master_param,
                                MT d_beta1,
                                MT d_beta2,
                                MT d_epsilon,
                                int num,
                                T* param_out,
                                MT* moment_out,
                                MT* inf_norm_out,
                                MT* master_param_out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  MT lr = static_cast<MT>(learning_rate[0]);
  MT d_pow = static_cast<MT>(beta1_pow[0]);
  MT one = static_cast<MT>(1.0f);
  auto l_r = lr / (one - d_pow);

  for (int index = idx; index < num; index += gridDim.x * blockDim.x) {
    // load and cast input to MT
    MT d_param =
        master_param ? master_param[index] : static_cast<MT>(param[index]);
    MT d_grad = static_cast<MT>(grad[index]);
    MT d_moment = static_cast<MT>(moment[index]);
    MT d_inf = static_cast<MT>(inf_norm[index]);
    // compute
    auto mom_out = d_beta1 * d_moment + (one - d_beta1) * d_grad;
    auto norm_out = std::max(std::abs(d_grad), d_beta2 * d_inf + d_epsilon);
    auto out_data = d_param - l_r * (mom_out / norm_out);
    // store
    param_out[index] = static_cast<T>(out_data);
    moment_out[index] = static_cast<T>(mom_out);
    inf_norm_out[index] = static_cast<T>(norm_out);

    if (master_param_out) {
      master_param_out[index] = out_data;
    }
  }
}

template <typename T, typename Context>
void AdamaxKernel(const Context& dev_ctx,
                  const DenseTensor& param,
                  const DenseTensor& grad,
                  const DenseTensor& learning_rate,
                  const DenseTensor& moment,
                  const DenseTensor& inf_norm,
                  const DenseTensor& beta1_pow,
                  const paddle::optional<DenseTensor>& master_param,
                  float beta1,
                  float beta2,
                  float epsilon,
                  bool multi_precision,
                  DenseTensor* param_out,
                  DenseTensor* moment_out,
                  DenseTensor* inf_norm_out,
                  DenseTensor* master_param_outs) {
  using MPDType = typename phi::dtype::template MPTypeTrait<T>::Type;
  T* param_out_data = dev_ctx.template Alloc<T>(param_out);
  MPDType* moment_out_data = dev_ctx.template Alloc<MPDType>(moment_out);
  MPDType* inf_norm_out_data = dev_ctx.template Alloc<MPDType>(inf_norm_out);
  const MPDType* master_in_data =
      multi_precision ? master_param->data<MPDType>() : nullptr;
  MPDType* master_out_data =
      multi_precision ? dev_ctx.template Alloc<MPDType>(master_param_outs)
                      : nullptr;
  PADDLE_ENFORCE_EQ(
      beta1_pow.numel(),
      1,
      errors::InvalidArgument("beta1 pow's size should be 1, but received "
                              "value is:%d.",
                              beta1_pow.numel()));

  MPDType beta1_ = static_cast<MPDType>(beta1);
  MPDType beta2_ = static_cast<MPDType>(beta2);
  MPDType epsilon_ = static_cast<MPDType>(epsilon);

  int numel = param.numel();
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel, 1);
  int grid = config.block_per_grid.x;
  int block = config.thread_per_block.x;
  auto stream = dev_ctx.stream();

  AdamaxGPUKernel<T, MPDType>
      <<<block, grid, 0, stream>>>(param.data<T>(),
                                   grad.data<T>(),
                                   learning_rate.data<MPDType>(),
                                   moment.data<MPDType>(),
                                   inf_norm.data<MPDType>(),
                                   beta1_pow.data<MPDType>(),
                                   master_in_data,
                                   beta1_,
                                   beta2_,
                                   epsilon_,
                                   numel,
                                   param_out_data,
                                   moment_out_data,
                                   inf_norm_out_data,
                                   master_out_data);
}
}  // namespace phi
PD_REGISTER_KERNEL(adamax,
                   GPU,
                   ALL_LAYOUT,
                   phi::AdamaxKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
  }
}
