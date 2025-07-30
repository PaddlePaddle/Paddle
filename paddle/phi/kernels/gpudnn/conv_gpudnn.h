// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/phi/kernels/conv_kernel.h"

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

#ifdef PADDLE_WITH_HIP
#include "paddle/phi/kernels/gpudnn/conv_miopen_helper.h"
#else
#include "paddle/phi/kernels/gpudnn/conv_cudnn_v7.h"
#endif

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"

#ifdef PADDLE_WITH_CUDNN_FRONTEND
// clang-format off
#include "paddle/phi/backends/dynload/cudnn_frontend.h"
#include "paddle/phi/kernels/autotune/cache.h"
#include "paddle/phi/kernels/gpudnn/conv_cudnn_frontend.h"
// clang-format on
#endif

namespace phi {

template <typename T, typename Context>
void ConvCudnnKernel(const Context& dev_ctx,
                     const DenseTensor& input,
                     const DenseTensor& filter,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings_t,
                     const std::string& padding_algorithm,
                     const std::vector<int>& dilations_t,
                     int groups,
                     const std::string& data_format,
                     DenseTensor* output);

template <typename T, typename Context>
void DepthwiseConvCudnnKernel(const Context& dev_ctx,
                              const DenseTensor& input,
                              const DenseTensor& filter,
                              const std::vector<int>& strides,
                              const std::vector<int>& paddings,
                              const std::string& padding_algorithm,
                              int groups,
                              const std::vector<int>& dilations,
                              const std::string& data_format,
                              DenseTensor* out) {
  ConvCudnnKernel<T>(dev_ctx,
                     input,
                     filter,
                     strides,
                     paddings,
                     padding_algorithm,
                     dilations,
                     groups,
                     data_format,
                     out);
}

template <typename T, typename Context>
void ConvCudnnGradKernel(const Context& dev_ctx,
                         const DenseTensor& input,
                         const DenseTensor& filter,
                         const DenseTensor& output_grad,
                         const std::vector<int>& strides_t,
                         const std::vector<int>& paddings_t,
                         const std::string& padding_algorithm,
                         const std::vector<int>& dilations_t,
                         int groups,
                         const std::string& data_format,
                         DenseTensor* input_grad,
                         DenseTensor* filter_grad);

template <typename T, typename Context>
void DepthwiseConvCudnnGradKernel(const Context& dev_ctx,
                                  const DenseTensor& input,
                                  const DenseTensor& filter,
                                  const DenseTensor& out_grad,
                                  const std::vector<int>& strides,
                                  const std::vector<int>& paddings,
                                  const std::string& padding_algorithm,
                                  int groups,
                                  const std::vector<int>& dilations,
                                  const std::string& data_format,
                                  DenseTensor* input_grad,
                                  DenseTensor* filter_grad) {
  ConvCudnnGradKernel<T>(dev_ctx,
                         input,
                         filter,
                         out_grad,
                         strides,
                         paddings,
                         padding_algorithm,
                         dilations,
                         groups,
                         data_format,
                         input_grad,
                         filter_grad);
}

}  // namespace phi
