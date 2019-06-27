// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdint.h>
#include <vector>
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/core/target_wrapper.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <PrecisionType IN, PrecisionType OUT>
bool trans_tensor_dtype(Tensor* tin, Tensor* tout, float input_scale,
                        float output_scale, std::vector<float> weights_scale) {
  LOG(FATAL) << "trans_tensor_dtype has no impl";
  return false;
}

template <>
bool trans_tensor_dtype<PRECISION(kInt32), PRECISION(kInt8)>(
    Tensor* tin, Tensor* tout, float input_scale, float output_scale,
    std::vector<float> weights_scale);

template <>
bool trans_tensor_dtype<PRECISION(kInt32), PRECISION(kFloat)>(
    Tensor* tin, Tensor* tout, float input_scale, float output_scale,
    std::vector<float> weights_scale);

template <PrecisionType IN>
bool get_tensor_scale(const Tensor& tin, std::vector<float>* scale_out,
                      int axis, float scale_factor) {
  return false;
}

std::vector<float> get_tensor_scale_n(const float* in_data, int axis_size,
                                      int64_t inner_size, float scale_factor);

bool trans_fp32_bias_to_int32_basic(Tensor* tin, Tensor* tout, float in_scale,
                                    std::vector<float> vector_weight_scale);

bool trans_tensor_int32_to_int8(Tensor* tin, Tensor* tout, float input_scale,
                                float output_scale,
                                std::vector<float> weights_scale, int axis = 1);

bool trans_tensor_int32_to_fp32(Tensor* tin, Tensor* tout, float input_scale,
                                std::vector<float> weights_scale, int axis = 1);

bool trans_tensor_fp32_to_int8(Tensor* tin, Tensor* tout, float input_scale);

template <>
bool get_tensor_scale<PRECISION(kFloat)>(const Tensor& tin,
                                         std::vector<float>* scale_out,
                                         int axis, float scale_factor);

template <typename dtype>
void int32_to_dtype(const int* din, dtype* dout, const float* scale,
                    int axis_size, int64_t outer_size, int64_t inner_size);

void fp32_to_int8(const float* din, int8_t* dout, const float* scale,
                  int axis_size, int64_t outer_size, int64_t inner_size);

void int8_to_fp32(const int8_t* in, float* out, const float* scale,
                  int axis_size, int64_t outer_size, int64_t inner_size);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
