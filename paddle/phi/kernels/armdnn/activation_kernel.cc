// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/backends/armdnn/armdnn_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"

namespace phi {

template <typename T, typename Context>
void ReluKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DenseTensor* out) {
  using ArmDNNType = typename ArmDNNTypeTrait<T>::Type;
  PADDLE_ENFORCE_NOT_NULL(out,
                          errors::NotFound("Output Out should not be nullptr"));
  dev_ctx.template Alloc<ArmDNNType>(out);
  armdnnlibrary::relu<ArmDNNType>(dev_ctx.context(),
                                  x.data<ArmDNNType>(),
                                  out->data<ArmDNNType>(),
                                  x.numel());
}

}  // namespace phi

#define PD_REGISTER_ACTIVATION_KERNEL(name, func) \
  PD_REGISTER_KERNEL(                             \
      name, ArmDNN, ALL_LAYOUT, phi::func, float, phi::dtype::bfloat16) {}

PD_REGISTER_ACTIVATION_KERNEL(relu, ReluKernel)
