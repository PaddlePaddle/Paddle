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

#include "paddle/phi/kernels/rprop_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_helper.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/mixed_vector.h"

namespace phi {

template <typename T, typename MT>
__global__ void RpropKernelGPUImpl(const T* param,
                                   const T* grad,
                                   const T* prev,
                                   const T* learning_rate,
                                   const MT* master_param,
                                   const T* learning_rate_range,
                                   const T* etas,
                                   int num,
                                   T* param_out,
                                   T* prev_out,
                                   T* learning_rate_out,
                                   MT* master_param_out) {
  MT learning_rate_min_data = static_cast<MT>(learning_rate_range[0]);
  MT learning_rate_max_data = static_cast<MT>(learning_rate_range[1]);
  MT eta_negative_data = static_cast<MT>(etas[0]);
  MT eta_positive_data = static_cast<MT>(etas[1]);
  MT zero_data = static_cast<MT>(0);
  MT one_data = static_cast<MT>(1);
  MT negative_one_data = static_cast<MT>(-1);

  CUDA_KERNEL_LOOP(i, num) {
    MT param_data = master_param ? master_param[i] : static_cast<MT>(param[i]);
    MT grad_data = static_cast<MT>(grad[i]);
    MT prev_data = static_cast<MT>(prev[i]);
    MT learning_rate_data = static_cast<MT>(learning_rate[i]);
    MT product_data = grad_data * prev_data;

    MT eta_data = one_data;
    if (product_data > zero_data) {
      eta_data = eta_positive_data;
    } else if (product_data < zero_data) {
      grad_data = zero_data;
      eta_data = eta_negative_data;
    }

    learning_rate_data = learning_rate_data * eta_data;
    if (learning_rate_data > learning_rate_max_data) {
      learning_rate_data = learning_rate_max_data;
    } else if (learning_rate_data < learning_rate_min_data) {
      learning_rate_data = learning_rate_min_data;
    }

    MT grad_sign_data = zero_data;
    if (grad_data > zero_data) {
      grad_sign_data = one_data;
    } else if (grad_data < zero_data) {
      grad_sign_data = negative_one_data;
    }

    param_data = param_data - grad_sign_data * learning_rate_data;
    prev_data = grad_data;

    param_out[i] = static_cast<T>(param_data);
    prev_out[i] = static_cast<T>(prev_data);
    learning_rate_out[i] = static_cast<T>(learning_rate_data);
    if (master_param_out) {
      master_param_out[i] = param_data;
    }
  }
}

template <typename T, typename Context>
void RpropKernel(const Context& dev_ctx,
                 const DenseTensor& param,
                 const DenseTensor& grad,
                 const DenseTensor& prev,
                 const DenseTensor& learning_rate,
                 const paddle::optional<DenseTensor>& master_param,
                 const DenseTensor& learning_rate_range,
                 const DenseTensor& etas,
                 bool multi_precision,
                 DenseTensor* param_out,
                 DenseTensor* prev_out,
                 DenseTensor* learning_rate_out,
                 DenseTensor* master_param_out) {
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;
  const MPDType* master_in_data =
      multi_precision ? master_param->data<MPDType>() : nullptr;
  MPDType* master_out_data =
      multi_precision ? dev_ctx.template Alloc<MPDType>(master_param_out)
                      : nullptr;

  int block = 512;
  int grid = (param.numel() + block - 1) / block;

  RpropKernelGPUImpl<T, MPDType><<<grid, block, 0, dev_ctx.stream()>>>(
      param.data<T>(),
      grad.data<T>(),
      prev.data<T>(),
      learning_rate.data<T>(),
      master_in_data,
      learning_rate_range.data<T>(),
      etas.data<T>(),
      param.numel(),
      dev_ctx.template Alloc<T>(param_out),
      dev_ctx.template Alloc<T>(prev_out),
      dev_ctx.template Alloc<T>(learning_rate_out),
      master_out_data);
}

}  // namespace phi

#ifdef PADDLE_WITH_CUDA
PD_REGISTER_KERNEL(rprop,
                   GPU,
                   ALL_LAYOUT,
                   phi::RpropKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   float,
                   double) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16 ||
      kernel_key.dtype() == phi::DataType::BFLOAT16) {
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
  }
}
#endif

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(rprop,
                   GPU,
                   ALL_LAYOUT,
                   phi::RpropKernel,
                   phi::dtype::float16,
                   float,
                   double) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
  }
}
#endif
