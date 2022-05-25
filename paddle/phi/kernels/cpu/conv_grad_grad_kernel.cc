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

#include "paddle/phi/kernels/conv_grad_grad_kernel.h"
#include "paddle/phi/kernels/impl/conv_grad_grad_kernel_impl.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void Conv3DGradGradKernel(const Context& ctx,
                          paddle::optional<const DenseTensor&> input_grad_grad,
                          paddle::optional<const DenseTensor&> filter_grad_grad,
                          const DenseTensor& out_grad,
                          const DenseTensor& input,
                          const DenseTensor& filter,
                          const std::vector<int>& strides,
                          const std::vector<int>& paddings_t,
                          const std::string& padding_algorithm,
                          int groups,
                          const std::vector<int>& dilations_t,
                          const std::string& data_format,
                          bool use_addto,
                          int workspace_size_MB,
                          bool exhaustive_search_t,
                          DenseTensor* out_grad_grad,
                          DenseTensor* input_grad,
                          DenseTensor* filter_grad) {
  ConvGradGradKernel<T>(ctx,
                        input,
                        filter,
                        out_grad,
                        input_grad_grad,
                        filter_grad_grad,
                        strides,
                        paddings_t,
                        padding_algorithm,
                        groups,
                        dilations_t,
                        data_format,
                        use_addto,
                        workspace_size_MB,
                        exhaustive_search_t,
                        input_grad,
                        filter_grad,
                        out_grad_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    conv2d_grad_grad, CPU, ALL_LAYOUT, phi::ConvGradGradKernel, float, double) {
}

PD_REGISTER_KERNEL(conv3d_grad_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::Conv3DGradGradKernel,
                   float,
                   double) {}
