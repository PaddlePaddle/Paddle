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

#include "paddle/phi/kernels/asgd_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_helper.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/mixed_vector.h"

namespace phi {

template <typename T, typename MT>
__global__ void ASGDKernelGPUImpl(const T* param,
                                  const T* grad,
                                  const T* learning_rate,
                                  const T* d,
                                  const T* y,
                                  const T* n,
                                  const MT* master_param,
                                  int num,
                                  T* param_out,
                                  T* d_out,
                                  T* y_out,
                                  MT* master_param_out) {
  MT learning_rate_MT = static_cast<MT>(learning_rate[0]);
  MT n_MT = static_cast<MT>(n[0]);
  CUDA_KERNEL_LOOP(i, num) {
    MT param_data = master_param ? master_param[i] : static_cast<MT>(param[i]);
    MT grad_data = static_cast<MT>(grad[i]);
    MT d_data = static_cast<MT>(d[i]);
    MT y_data = static_cast<MT>(y[i]);
    d_data = d_data - y_data + grad_data;
    y_data = grad_data;
    param_data = param_data - (learning_rate_MT / n_MT) * d_data;
    param_out[i] = static_cast<T>(param_data);
    d_out[i] = static_cast<T>(d_data);
    y_out[i] = static_cast<T>(y_data);
    if (master_param_out) {
      master_param_out[i] = param_data;
    }
  }
}

template <typename T, typename Context>
void ASGDKernel(const Context& dev_ctx,
                const DenseTensor& param,
                const DenseTensor& grad,
                const DenseTensor& learning_rate,
                const DenseTensor& d,
                const DenseTensor& y,
                const DenseTensor& n,
                const paddle::optional<DenseTensor>& master_param,
                bool multi_precision,
                DenseTensor* param_out,
                DenseTensor* d_out,
                DenseTensor* y_out,
                DenseTensor* master_param_out) {
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;
  const MPDType* master_in_data =
      multi_precision ? master_param->data<MPDType>() : nullptr;
  MPDType* master_out_data =
      multi_precision ? dev_ctx.template Alloc<MPDType>(master_param_out)
                      : nullptr;

  int block = 512;
  int grid = (param.numel() + block - 1) / block;

  ASGDKernelGPUImpl<T, MPDType><<<grid, block, 0, dev_ctx.stream()>>>(
      param.data<T>(),
      grad.data<T>(),
      learning_rate.data<T>(),
      d.data<T>(),
      y.data<T>(),
      n.data<T>(),
      master_in_data,
      param.numel(),
      dev_ctx.template Alloc<T>(param_out),
      dev_ctx.template Alloc<T>(d_out),
      dev_ctx.template Alloc<T>(y_out),
      master_out_data);
}

}  // namespace phi

PD_REGISTER_KERNEL(asgd,
                   GPU,
                   ALL_LAYOUT,
                   phi::ASGDKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   float,
                   double) {}
