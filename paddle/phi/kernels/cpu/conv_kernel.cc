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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/conv_kernel_impl.h"

namespace phi {
<<<<<<< HEAD

template <typename T, typename Context>
void ConvKernel(const Context& dev_ctx,
                const DenseTensor& input,
                const DenseTensor& filter,
                const std::vector<int>& strides,
                const std::vector<int>& paddings,
                const std::string& padding_algorithm,
                const std::vector<int>& dilations,
                int groups,
                const std::string& data_format,
                DenseTensor* out) {
  ConvKernelImpl<T>(dev_ctx,
                    input,
                    filter,
                    strides,
                    paddings,
                    padding_algorithm,
                    groups,
                    dilations,
                    data_format,
                    out);
}

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
template <typename T, typename Context>
void DepthwiseConvKernel(const Context& dev_ctx,
                         const DenseTensor& input,
                         const DenseTensor& filter,
                         const std::vector<int>& strides,
                         const std::vector<int>& paddings,
                         const std::string& padding_algorithm,
                         int groups,
                         const std::vector<int>& dilations,
                         const std::string& data_format,
<<<<<<< HEAD
                         DenseTensor* out) {
  ConvKernelImpl<T>(dev_ctx,
                    input,
                    filter,
                    strides,
                    paddings,
                    padding_algorithm,
                    groups,
                    dilations,
                    data_format,
                    out);
=======
                         bool use_addto,
                         int workspace_size_MB,
                         bool exhaustive_search,
                         bool fuse_relu,
                         DenseTensor* out) {
  ConvKernel<T>(dev_ctx,
                input,
                filter,
                strides,
                paddings,
                padding_algorithm,
                groups,
                dilations,
                data_format,
                use_addto,
                workspace_size_MB,
                exhaustive_search,
                out);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
}

template <typename T, typename Context>
void Conv3DKernel(const Context& dev_ctx,
                  const DenseTensor& input,
                  const DenseTensor& filter,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string& padding_algorithm,
                  int groups,
                  const std::vector<int>& dilations,
                  const std::string& data_format,
<<<<<<< HEAD
                  DenseTensor* out) {
  ConvKernelImpl<T>(dev_ctx,
                    input,
                    filter,
                    strides,
                    paddings,
                    padding_algorithm,
                    groups,
                    dilations,
                    data_format,
                    out);
=======
                  bool use_addto,
                  int workspace_size_MB,
                  bool exhaustive_search,
                  DenseTensor* out) {
  ConvKernel<T>(dev_ctx,
                input,
                filter,
                strides,
                paddings,
                padding_algorithm,
                groups,
                dilations,
                data_format,
                use_addto,
                workspace_size_MB,
                exhaustive_search,
                out);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
}

}  // namespace phi

PD_REGISTER_KERNEL(conv2d, CPU, ALL_LAYOUT, phi::ConvKernel, float, double) {}

PD_REGISTER_KERNEL(depthwise_conv2d,
                   CPU,
                   ALL_LAYOUT,
                   phi::DepthwiseConvKernel,
                   float,
                   double) {}

PD_REGISTER_KERNEL(conv3d, CPU, ALL_LAYOUT, phi::Conv3DKernel, float, double) {}
