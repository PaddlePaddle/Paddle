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

#include "paddle/phi/backends/xpu/enforce_xpu.h"

#include "glog/logging.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void SpatialTransformerResblockXPUKernel(const Context& ctx,
                           const DenseTensor& x,
                        const DenseTensor& conv_bias,
                        const DenseTensor& conv_filter,
                        const DenseTensor& gn_bias,
                        const DenseTensor& gn_scale,
                        const std::vector<std::string>& conv_filter_max,
                        const std::vector<int>& dilations,
                        const std::vector<int>& paddings,
                        const std::vector<int>& strides, 
                        const std::vector<float>& gn_eps, 
                        const std::vector<int>& gn_groups, 
                        const std::vector<int>& groups,
                        DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(spatial_transformer_resblock_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::SpatialTransformerResblockXPUKernel,
                   float,
                   phi::dtype::float16) {}
