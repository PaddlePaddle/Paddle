// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/phi/core/kernel_registry.h"

namespace phi {

inline std::vector<int64_t> CorrelationOutputSize(int batch,
                                                  int input_height,
                                                  int input_width,
                                                  int stride1,
                                                  int stride2,
                                                  int kernel_size,
                                                  int pad_size,
                                                  int max_displacement) {
  std::vector<int64_t> output_shape({batch});
  int kernel_radius = (kernel_size - 1) / 2;
  int border_radius = kernel_radius + max_displacement;
  int padded_input_height = input_height + 2 * pad_size;
  int padded_input_width = input_width + 2 * pad_size;
  int output_channel = ((max_displacement / stride2) * 2 + 1) *
                       ((max_displacement / stride2) * 2 + 1);
  output_shape.push_back(output_channel);
  int output_height =
      std::ceil(static_cast<float>(padded_input_height - 2 * border_radius) /
                static_cast<float>(stride1));
  int output_width =
      std::ceil(static_cast<float>(padded_input_width - 2 * border_radius) /
                static_cast<float>(stride1));
  output_shape.push_back(output_height);
  output_shape.push_back(output_width);
  return output_shape;
}

template <typename T, typename Context>
void CorrelationKernel(const Context& dev_ctx,
                       const DenseTensor& input1,
                       const DenseTensor& input2,
                       int pad_size,
                       int kernel_size,
                       int max_displacement,
                       int stride1,
                       int stride2,
                       int corr_type_multiply,
                       DenseTensor* out) {
  bool is_gpu_place = dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU;
  PADDLE_ENFORCE_EQ(
      is_gpu_place,
      true,
      phi::errors::Unimplemented("Correlation only supports GPU now."));
}

}  // namespace phi

PD_REGISTER_KERNEL(
    correlation, CPU, ALL_LAYOUT, phi::CorrelationKernel, float, double) {}
