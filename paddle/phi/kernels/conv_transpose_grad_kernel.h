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

#include <string>
#include <vector>
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void Conv2dTransposeGradKernel(const Context& ctx,
                               const DenseTensor& x,
                               const DenseTensor& filter,
                               const DenseTensor& dout,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               const std::vector<int>& output_padding,
                               const std::vector<int>& output_size,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations,
                               const std::string& data_format,
                               DenseTensor* dx,
                               DenseTensor* dfilter);

template <typename T, typename Context>
void Conv2dTransposeDoubleGradKernel(const Context& ctx,
                                     const DenseTensor& x,
                                     const DenseTensor& filter,
                                     const DenseTensor& dout,
                                     const DenseTensor& ddx,
                                     const DenseTensor& ddfilter,
                                     const std::vector<int>& strides,
                                     const std::vector<int>& paddings,
                                     const std::vector<int>& output_padding,
                                     const std::vector<int>& output_size,
                                     const std::string& padding_algorithm,
                                     int groups,
                                     const std::vector<int>& dilations,
                                     const std::string& data_format,
                                     DenseTensor* dx,
                                     DenseTensor* dfilter,
                                     DenseTensor* ddout);

template <typename T, typename Context>
void Conv3dTransposeGradKernel(const Context& ctx,
                               const DenseTensor& x,
                               const DenseTensor& filter,
                               const DenseTensor& dout,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               const std::vector<int>& output_padding,
                               const std::vector<int>& output_size,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations,
                               const std::string& data_format,
                               DenseTensor* dx,
                               DenseTensor* dfilter);

template <typename T, typename Context>
void DepthwiseConv2dTransposeGradKernel(const Context& ctx,
                                        const DenseTensor& x,
                                        const DenseTensor& filter,
                                        const DenseTensor& dout,
                                        const std::vector<int>& strides,
                                        const std::vector<int>& paddings,
                                        const std::vector<int>& output_padding,
                                        const std::vector<int>& output_size,
                                        const std::string& padding_algorithm,
                                        int groups,
                                        const std::vector<int>& dilations,
                                        const std::string& data_format,
                                        DenseTensor* dx,
                                        DenseTensor* dfilter);

}  // namespace phi
