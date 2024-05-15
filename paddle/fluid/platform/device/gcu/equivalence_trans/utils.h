/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {

enum class BindingFuncType { IDENTITY, ADD, MAX, MIN, GE, OR };

// KERNEL TYPE DEFINE
const char* const kConv2D = "conv2d";
const char* const kConv2DGrad = "conv2d_grad";
const char* const kDepthWiseConv2D = "depthwise_conv2d";
const char* const kDepthWiseConv2DGrad = "depthwise_conv2d_grad";
const char* const kConv2DTranspose = "conv2d_transpose";
const char* const kConv2DTransposeGrad = "conv2d_transpose_grad";
const char* const kConv3D = "conv3d";
const char* const kConv3DGrad = "conv3d_grad";
const char* const kConv3DTranspose = "conv3d_transpose";
const char* const kConv3DTransposeGrad = "conv3d_transpose_grad";

// for conv
// ref: https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
static std::vector<int64_t> get_same_padding_value(int64_t dim,
                                                   int64_t ksize,
                                                   int64_t stride) {
  int64_t pad_along_dim = 0;
  if (dim % stride == 0) {
    pad_along_dim = std::max(ksize - stride, static_cast<int64_t>(0));
  } else {
    pad_along_dim = std::max(ksize - (dim % stride), static_cast<int64_t>(0));
  }
  int64_t pad_low = pad_along_dim / 2;
  int64_t pad_high = pad_along_dim - pad_low;
  std::vector<int64_t> padding{pad_low, pad_high};
  return std::move(padding);
}

static std::pair<int64_t, int64_t> get_backprop_filter_padding(
    int64_t input_dim,
    int64_t output_dim,
    int64_t kernel_size,
    int64_t stride,
    int64_t dilation,
    int64_t padding_before,
    const std::string& padding_algorithm) {
  std::pair<int64_t, int64_t> padding_dim;
  int64_t expanded_output_size = (output_dim - 1) * stride + 1;
  int64_t padded_in_size = (kernel_size - 1) * dilation;
  padded_in_size += expanded_output_size;
  int64_t pad_total = padded_in_size - input_dim;
  int64_t pad_before = padding_algorithm == "EXPLICIT" ? padding_before
                       : padding_algorithm == "SAME"
                           ? std::max<int64_t>(pad_total / 2, 0)
                           : 0;
  padding_dim = {pad_before, pad_total - pad_before};
  return padding_dim;
}

static std::pair<int64_t, int64_t> get_backprop_input_padding(
    int64_t input_dim,
    int64_t output_dim,
    int64_t kernel_size,
    int64_t stride,
    int64_t dilation,
    int64_t padding_before) {
  std::pair<int64_t, int64_t> padding_dim;
  int64_t effective_filter_size = (kernel_size - 1) * dilation + 1;
  int64_t expanded_output_size = (output_dim - 1) * stride + 1;
  int64_t padded_out_size = input_dim + effective_filter_size - 1;
  int64_t pad_before = effective_filter_size - 1 - padding_before;
  int64_t pad_after = padded_out_size - expanded_output_size - pad_before;
  padding_dim = {pad_before, pad_after};
  return padding_dim;
}

static int64_t GetConvTransposeDim(int64_t input_dim,
                                   int64_t ksize,
                                   int64_t stride,
                                   int64_t dilation,
                                   int64_t pad_low,
                                   int64_t pad_high,
                                   int64_t output_padding) {
  int64_t expanded_input_size = (input_dim - 1) * stride + 1;
  int64_t effective_filter_size = (ksize - 1) * dilation + 1;
  int64_t output_dim = expanded_input_size - 1 + output_padding +
                       effective_filter_size - pad_low - pad_high;
  return output_dim;
}

static std::vector<int64_t> get_conv2d_transpose_padding(
    const std::vector<int64_t>& input_spatial_dims,
    const std::vector<int64_t>& output_spatial_dims,
    const std::vector<int64_t>& ksize,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& dilation,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& output_padding,
    const std::string& auto_pad) {
  std::vector<int64_t> padding_value;
  for (size_t i = 0; i < input_spatial_dims.size(); ++i) {
    int64_t expanded_input_size = (input_spatial_dims[i] - 1) * stride[i] + 1;
    int64_t effective_filter_size = (ksize[i] - 1) * dilation[i] + 1;
    int64_t pad_before = effective_filter_size - 1 - padding[i * 2];
    int64_t padded_out_size =
        output_spatial_dims[i] + effective_filter_size - 1;
    int64_t pad_after = padded_out_size - expanded_input_size - pad_before;
    padding_value.emplace_back(pad_before);
    padding_value.emplace_back(pad_after);
  }
  return padding_value;
}

static std::vector<int64_t> GetBroadcastDimensions(int64_t dims,
                                                   int64_t reduce_dim) {
  std::vector<int64_t> result_dims;
  result_dims.reserve(dims);
  for (int64_t i = 0; i < dims; ++i) {
    if (reduce_dim != i) {
      result_dims.push_back(i);
    }
  }
  return result_dims;
}
static std::vector<float> RngUniform(float a,
                                     float b,
                                     int64_t seed,
                                     std::vector<int64_t> shape) {
  std::default_random_engine e(seed);
  std::uniform_real_distribution<float> u(a, b);
  std::vector<float> result;
  int64_t total_size = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    total_size *= shape[i];
  }
  for (int64_t i = 0; i < total_size; ++i) {
    result.push_back(u(e));
  }
  return result;
}

static void SetBuilderScopeName(GcuBuilderPtr builder,
                                const std::string& state,
                                const std::string& scope_name) {
  builder->SetState(state.c_str(), builder::Attribute(scope_name.c_str()));
}

static void CleanBuilderScopeName(GcuBuilderPtr builder,
                                  const std::string& state) {
  builder->SetState(state.c_str(), builder::Attribute("-"), "clean");
}

static size_t SizeOf(const builder::PrimitiveType& type) {
  if (type == builder::PrimitiveType::PRED() ||
      type == builder::PrimitiveType::S8() ||
      type == builder::PrimitiveType::U8()) {
    return 1;
  } else if (type == builder::PrimitiveType::S16() ||
             type == builder::PrimitiveType::U16() ||
             type == builder::PrimitiveType::F16()) {
    return 2;
  } else if (type == builder::PrimitiveType::S32() ||
             type == builder::PrimitiveType::U32() ||
             type == builder::PrimitiveType::F32()) {
    return 4;
  } else if (type == builder::PrimitiveType::S64() ||
             type == builder::PrimitiveType::U64() ||
             type == builder::PrimitiveType::F64()) {
    return 8;
  } else {
    return -1;
  }
}

static std::vector<std::string> CreateBindingFunc(
    std::shared_ptr<builder::Builder> builder,
    const std::vector<BindingFuncType>& func_types,
    const std::vector<builder::PrimitiveType>& ptypes,
    const std::string& base_name = "body_") {
  PADDLE_ENFORCE_EQ(ptypes.size(),
                    func_types.size(),
                    platform::errors::NotFound(
                        "func_types:%zu must have same size with ptypes:%zu",
                        func_types.size(),
                        ptypes.size()));
  std::vector<std::string> func_names;
  func_names.reserve(func_types.size());
  builder::Op arg0, arg1, out;
  for (size_t i = 0; i < func_types.size(); ++i) {
    std::string func_name = base_name + std::to_string(i);
    builder->AddFunc(func_name.c_str());
    builder::Type scalar_type({1}, ptypes[i]);
    arg0 = builder->CreateInput(scalar_type, func_name.c_str());
    arg1 = builder->CreateInput(scalar_type, func_name.c_str());
    switch (func_types[i]) {
      case BindingFuncType::IDENTITY:
        out = arg0;
        break;
      case BindingFuncType::ADD:
        out = arg0 + arg1;
        break;
      case BindingFuncType::MAX:
        out = builder::Max(arg0, arg1);
        break;
      case BindingFuncType::MIN:
        out = builder::Min(arg0, arg1);
        break;
      case BindingFuncType::GE:
        out = builder::GreaterEqual(arg0, arg1);
        break;
      case BindingFuncType::OR:
        out = builder::Or(arg0, arg1);
        break;
      default:
        PADDLE_THROW(platform::errors::NotFound("Unsupport BindingFuncType."));
        break;
    }
    builder->SetOutput({out}, func_name.c_str());
    func_names.emplace_back(func_name);
  }
  return std::move(func_names);
}
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
