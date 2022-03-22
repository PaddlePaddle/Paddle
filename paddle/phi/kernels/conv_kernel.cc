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

#include "paddle/phi/kernels/conv_kernel.h"

#include "paddle/phi/core/kernel_registry.h"

#include "paddle/fluid/platform/cudnn_workspace_helper.h"

namespace phi {

template <typename T, typename Context>
void ConvInferKernel(const Context& dev_ctx,
                     const DenseTensor& input,
                     const DenseTensor& filter,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     const std::string& paddding_algorithm,
                     int groups,
                     const std::vector<int>& dilations,
                     const std::string& data_format,
                     DenseTensor* out) {
  ConvKernel<T, Context>(dev_ctx,
                         input,
                         filter,
                         strides,
                         paddings,
                         paddding_algorithm,
                         groups,
                         dilations,
                         data_format,
                         /*use_addto=*/false,
                         /*workspace_size_MB=*/paddle::platform::
                             GetDefaultConvWorkspaceSizeLimitMB(),
                         /*exhaustive_search=*/false,
                         out);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    conv2d_infer, CPU, ALL_LAYOUT, phi::ConvInferKernel, float, double) {}
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(
    conv2d_infer, GPU, ALL_LAYOUT, phi::ConvInferKernel, float, double) {}
#endif
