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

#include "paddle/phi/kernels/conv_grad_kernel.h"
#include "paddle/phi/kernels/impl/conv_grad_kernel_impl.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void Conv3DGradKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      const DenseTensor& filter,
                      const DenseTensor& out_grad,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings,
                      const std::string& paddding_algorithm,
                      int groups,
                      const std::vector<int>& dilations,
                      const std::string& data_format,
                      bool use_addto,
                      int workspace_size_MB,
                      bool exhaustive_search,
                      DenseTensor* input_grad,
                      DenseTensor* filter_grad) {
  ConvGradKernel<T>(dev_ctx,
                    input,
                    filter,
                    out_grad,
                    strides,
                    paddings,
                    paddding_algorithm,
                    groups,
                    dilations,
                    data_format,
                    use_addto,
                    workspace_size_MB,
                    exhaustive_search,
                    input_grad,
                    filter_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    conv2d_grad, GPU, ALL_LAYOUT, phi::ConvGradKernel, float, double) {}

PD_REGISTER_KERNEL(
    conv3d_grad, GPU, ALL_LAYOUT, phi::Conv3DGradKernel, float, double) {}
