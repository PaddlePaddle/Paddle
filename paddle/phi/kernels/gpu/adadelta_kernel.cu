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

#include "paddle/phi/kernels/adadelta_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
namespace phi {

template <typename T, typename MT>
__global__ void AdadeltaGPUKernel(const T* param,
                                  const T* grad,
                                  const MT* avg_squared_grad,
                                  const MT* avg_squared_update,
                                  const MT* master_param,
                                  MT rho,
                                  MT epsilon,
                                  T* param_out,
                                  MT* avg_squared_grad_out,
                                  MT* avg_squared_update_out,
                                  MT* master_param_outs,
                                  int num) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  auto l_rho = static_cast<MT>(1) - rho;

  for (int idx = index; idx < num; idx += gridDim.x * blockDim.x) {
    auto grad_data = static_cast<MT>(grad[idx]);
    avg_squared_grad_out[idx] = static_cast<MT>(avg_squared_grad[idx]) * rho +
                                l_rho * grad_data * grad_data;

    auto update_data = static_cast<MT>(avg_squared_update[idx]);
    auto update =
        -sqrt((update_data + epsilon) / (avg_squared_grad_out[idx] + epsilon)) *
        grad_data;
    avg_squared_update_out[idx] = rho * update_data + l_rho * update * update;

    auto input =
        master_param_outs ? master_param[idx] : static_cast<MT>(param[idx]);

    auto param_out_data = input + update;
    param_out[idx] = static_cast<T>(param_out_data);

    if (master_param_outs) {
      master_param_outs[idx] = param_out_data;
    }
  }
}

template <typename T, typename Context>
void AdadeltaKernel(const Context& dev_ctx,
                    const DenseTensor& param,
                    const DenseTensor& grad,
                    const DenseTensor& avg_squared_grad,
                    const DenseTensor& avg_squared_update,
                    const paddle::optional<DenseTensor>& master_param,
                    float rho,
                    float epsilon,
                    bool multi_precision,
                    DenseTensor* param_out,
                    DenseTensor* avg_squared_grad_out,
                    DenseTensor* avg_squared_update_out,
                    DenseTensor* master_param_outs) {
  using MPDType = typename phi::dtype::template MPTypeTrait<T>::Type;

  T* param_out_data = dev_ctx.template Alloc<T>(param_out);
  MPDType* avg_squared_grad_out_data =
      dev_ctx.template Alloc<MPDType>(avg_squared_grad_out);
  MPDType* avg_squared_update_out_data =
      dev_ctx.template Alloc<MPDType>(avg_squared_update_out);

  const MPDType* master_in_data =
      multi_precision ? master_param->data<MPDType>() : nullptr;
  MPDType* master_out_data =
      multi_precision ? dev_ctx.template Alloc<MPDType>(master_param_outs)
                      : nullptr;

  MPDType rho_ = static_cast<MPDType>(rho);
  MPDType epsilon_ = static_cast<MPDType>(epsilon);

  int numel = param.numel();
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel, 1);
  int grid = config.block_per_grid.x;
  int block = config.thread_per_block.x;
  auto stream = dev_ctx.stream();
  AdadeltaGPUKernel<T, MPDType>
      <<<block, grid, 0, stream>>>(param.data<T>(),
                                   grad.data<T>(),
                                   avg_squared_grad.data<MPDType>(),
                                   avg_squared_update.data<MPDType>(),
                                   master_in_data,
                                   rho,
                                   epsilon,
                                   param_out_data,
                                   avg_squared_grad_out_data,
                                   avg_squared_update_out_data,
                                   master_out_data,
                                   numel);
}
}  // namespace phi

PD_REGISTER_KERNEL(adadelta,
                   GPU,
                   ALL_LAYOUT,
                   phi::AdadeltaKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
