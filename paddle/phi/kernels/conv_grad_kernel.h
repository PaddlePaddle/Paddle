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

#pragma once

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void ConvGradKernel(const Context& dev_ctx,
                    const DenseTensor& input,
                    const DenseTensor& filter,
                    const DenseTensor& out_grad,
                    const std::vector<int>& strides,
                    const std::vector<int>& paddings,
                    const std::string& padding_algorithm,
                    const std::vector<int>& dilations,
                    int groups,
                    const std::string& data_format,
                    DenseTensor* input_grad,
                    DenseTensor* filter_grad);

template <typename T, typename Context>
void Conv3DGradKernel(const Context& dev_ctx,
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
                      DenseTensor* filter_grad);

template <typename T, typename Context>
void DepthwiseConvGradKernel(const Context& dev_ctx,
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
                             DenseTensor* filter_grad);

template <typename T, typename Context>
void ConvGradGradKernel(const Context& dev_ctx,
                        const DenseTensor& input,
                        const DenseTensor& filter,
                        const DenseTensor& out_grad,
                        const paddle::optional<DenseTensor>& input_grad_grad,
                        const paddle::optional<DenseTensor>& filter_grad_grad,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings,
                        const std::string& padding_algorithm,
                        const std::vector<int>& dilations,
                        int groups,
                        const std::string& data_format,
                        DenseTensor* input_grad,
                        DenseTensor* filter_grad,
                        DenseTensor* out_grad_grad);

template <typename T, typename Context>
void Conv3DDoubleGradKernel(
    const Context& dev_ctx,
    const DenseTensor& input,
    const DenseTensor& filter,
    const DenseTensor& out_grad,
    const paddle::optional<DenseTensor>& input_grad_grad,
    const paddle::optional<DenseTensor>& filter_grad_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format,
    DenseTensor* input_grad,
    DenseTensor* filter_grad,
    DenseTensor* out_grad_grad);

}  // namespace phi
