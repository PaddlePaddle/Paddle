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

#include "paddle/phi/kernels/merged_momentum_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/merged_momentum_impl.h"

namespace phi {

template <typename T, typename Context>
void MergedMomentumKernel(const Context& dev_ctx,
                          const std::vector<const DenseTensor*>& param,
                          const std::vector<const DenseTensor*>& grad,
                          const std::vector<const DenseTensor*>& velocity,
                          const std::vector<const DenseTensor*>& learning_rate,
                          const std::vector<const DenseTensor*>& master_param,
                          float mu,
                          bool use_nesterov,
                          const std::vector<std::string>& regularization_method,
                          const std::vector<float>& regularization_coeff,
                          bool multi_precision,
                          float rescale_grad,
                          std::vector<DenseTensor*> param_out,
                          std::vector<DenseTensor*> velocity_out,
                          std::vector<DenseTensor*> master_param_out) {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  if (multi_precision) {
    InnerCompute<MPType, Context, MPType, T>(dev_ctx,
                                             param,
                                             grad,
                                             velocity,
                                             learning_rate,
                                             master_param,
                                             mu,
                                             use_nesterov,
                                             regularization_method,
                                             regularization_coeff,
                                             rescale_grad,
                                             multi_precision,
                                             param_out,
                                             velocity_out,
                                             master_param_out);
  } else {
    InnerCompute<T, Context, MPType, T>(dev_ctx,
                                        param,
                                        grad,
                                        velocity,
                                        learning_rate,
                                        master_param,
                                        mu,
                                        use_nesterov,
                                        regularization_method,
                                        regularization_coeff,
                                        rescale_grad,
                                        multi_precision,
                                        param_out,
                                        velocity_out,
                                        master_param_out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(merged_momentum,
                   GPU,
                   ALL_LAYOUT,
                   phi::MergedMomentumKernel,
                   phi::dtype::float16,
                   float,
                   double) {}
