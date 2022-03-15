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
void Pool2dKernel(const Context& ctx,
                  const DenseTensor& x,
                  const std::vector<int>& kernel_size,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool ceil_mode,
                  bool exclusive,
                  const std::string& data_format,
                  const std::string& pooling_type,
                  bool global_pooling,
                  bool adaptive,
                  const std::string& padding_algorithm,
                  DenseTensor* out);

template <typename T, typename Context>
void Pool2dGPUDNNKernel(const Context& ctx,
                        const DenseTensor& x,
                        const std::vector<int>& kernel_size,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings,
                        bool ceil_mode,
                        bool exclusive,
                        const std::string& data_format,
                        const std::string& pooling_type,
                        bool global_pooling,
                        bool adaptive,
                        const std::string& padding_algorithm,
                        DenseTensor* out);

template <typename T, typename Context>
void MaxPool2dWithIndexKernel(const Context& ctx,
                              const DenseTensor& x,
                              const std::vector<int>& kernel_size,
                              const std::vector<int>& strides,
                              const std::vector<int>& paddings,
                              bool global_pooling,
                              bool adaptive,
                              DenseTensor* out,
                              DenseTensor* mask);

template <typename T, typename Context>
void Pool3dKernel(const Context& ctx,
                  const DenseTensor& x,
                  const std::vector<int>& kernel_size,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool ceil_mode,
                  bool exclusive,
                  const std::string& data_format,
                  const std::string& pooling_type,
                  bool global_pooling,
                  bool adaptive,
                  const std::string& padding_algorithm,
                  DenseTensor* out);

template <typename T, typename Context>
void Pool3dGPUDNNKernel(const Context& ctx,
                        const DenseTensor& x,
                        const std::vector<int>& kernel_size,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings,
                        bool ceil_mode,
                        bool exclusive,
                        const std::string& data_format,
                        const std::string& pooling_type,
                        bool global_pooling,
                        bool adaptive,
                        const std::string& padding_algorithm,
                        DenseTensor* out);

template <typename T, typename Context>
void MaxPool3dWithIndexKernel(const Context& ctx,
                              const DenseTensor& x,
                              const std::vector<int>& kernel_size,
                              const std::vector<int>& strides,
                              const std::vector<int>& paddings,
                              bool global_pooling,
                              bool adaptive,
                              DenseTensor* out,
                              DenseTensor* mask);

}  // namespace phi
