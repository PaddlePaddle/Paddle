/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/plugin/group_norm_op_plugin.h"
#include "paddle/phi/kernels/group_norm_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {
using DataLayout = phi::DataLayout;

int GroupNormPlugin::initialize() TRT_NOEXCEPT {
  if (!with_fp16_) {
    // if use fp32
    cudaMalloc(&scale_gpu_, sizeof(float) * scale_.size());
    cudaMalloc(&bias_gpu_, sizeof(float) * bias_.size());
    cudaMemcpy(scale_gpu_,
               scale_.data(),
               scale_.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(bias_gpu_,
               bias_.data(),
               bias_.size() * sizeof(float),
               cudaMemcpyHostToDevice);
  } else {
    // if use fp16
    std::vector<half> scale_half(scale_.size());
    std::vector<half> bias_half(bias_.size());
    for (int i = 0; i < scale_.size(); ++i) {
      scale_half[i] = static_cast<half>(scale_[i]);
    }
    for (int i = 0; i < bias_.size(); ++i) {
      bias_half[i] = static_cast<half>(bias_[i]);
    }
    cudaMalloc(&scale_gpu_, sizeof(half) * scale_half.size());
    cudaMalloc(&bias_gpu_, sizeof(half) * bias_half.size());
    cudaMemcpy(scale_gpu_,
               scale_half.data(),
               scale_half.size() * sizeof(half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(bias_gpu_,
               bias_half.data(),
               bias_half.size() * sizeof(half),
               cudaMemcpyHostToDevice);
  }
  return 0;
}

bool GroupNormPlugin::supportsFormat(
    nvinfer1::DataType type, nvinfer1::PluginFormat format) const TRT_NOEXCEPT {
  if (with_fp16_) {
    return ((type == nvinfer1::DataType::kHALF) &&
            (format == nvinfer1::PluginFormat::kLINEAR));
  } else {
    return ((type == nvinfer1::DataType::kFLOAT) &&
            (format == nvinfer1::PluginFormat::kLINEAR));
  }
}

nvinfer1::Dims GroupNormPlugin::getOutputDimensions(
    int index, const nvinfer1::Dims *inputDims, int nbInputs) TRT_NOEXCEPT {
  return inputDims[0];
}

int GroupNormPlugin::enqueue(int batch_size,
                             const void *const *inputs,
#if IS_TRT_VERSION_LT(8000)
                             void **outputs,
                             void *workspace,
#else
                             void *const *outputs,
                             void *workspace,
#endif
                             cudaStream_t stream) TRT_NOEXCEPT {
  const auto &input_dims = this->getInputDims(0);
  int groups = groups_;
  float eps = eps_;
  std::vector<int> input_shape;
  input_shape.push_back(batch_size);
  for (int i = 0; i < input_dims.nbDims; i++) {
    input_shape.push_back(input_dims.d[i]);
  }
  const auto input_ddim = phi::make_ddim(input_shape);

  int C = input_shape[1];

  PADDLE_ENFORCE_EQ(
      C,
      scale_.size(),
      platform::errors::InvalidArgument(
          "scale's size should be equal to the channel number in groupnorm,"
          "but got channel number:%d, scale's size:%d.",
          C,
          scale_.size()));
  PADDLE_ENFORCE_EQ(
      C,
      bias_.size(),
      platform::errors::InvalidArgument(
          "bias's size should be equal to the channel number in groupnorm,"
          "but got channel number:%d, bias's size:%d.",
          C,
          bias_.size()));
  float *mean_d = static_cast<float *>(workspace);
  float *variance_d = mean_d + input_shape[0] * groups_;
  float *temp_variance_d = variance_d + input_shape[0] * groups_;
  // phi::DenseTensor mean_t;
  // phi::DenseTensor variance_t;
  // phi::DenseTensor temp_variance_t;
  auto input_type = getDataType();
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. GroupNorm-->fp32";
    const float *input = static_cast<const float *>(inputs[0]);
    float *output = static_cast<float *>(outputs[0]);
    phi::GroupNormDirectCUDAFunctor<float> group_norm;
    group_norm(stream,
               input,
               input_shape,
               reinterpret_cast<float *>(bias_gpu_),
               reinterpret_cast<float *>(scale_gpu_),
               temp_variance_d,
               groups_,
               eps_,
               output,
               mean_d,
               variance_d,
               DataLayout::kNCHW);
  } else if (input_type == nvinfer1::DataType::kHALF) {
    VLOG(1) << "TRT Plugin DataType selected. GroupNorm-->fp16";
    const half *input = static_cast<const half *>(inputs[0]);
    half *output = static_cast<half *>(outputs[0]);
    phi::GroupNormDirectCUDAFunctor<half, float> group_norm;
    group_norm(stream,
               input,
               input_shape,
               reinterpret_cast<const half *>(bias_gpu_),
               reinterpret_cast<const half *>(scale_gpu_),
               temp_variance_d,
               groups_,
               eps_,
               output,
               mean_d,
               variance_d,
               DataLayout::kNCHW);
  } else {
    PADDLE_THROW(platform::errors::Fatal(
        "The GroupNorm TRT Plugin's input type should be float or half."));
  }
  return cudaGetLastError() != cudaSuccess;
}
nvinfer1::DimsExprs GroupNormPluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs *inputDims,
    int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  return inputDims[0];
}

bool GroupNormPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      platform::errors::InvalidArgument(
          "The input of groupnorm plugin shoule not be nullptr."));
  PADDLE_ENFORCE_LT(
      pos,
      nb_inputs + nb_outputs,
      platform::errors::InvalidArgument("The pos(%d) should be less than the "
                                        "num(%d) of the input and the output.",
                                        pos,
                                        nb_inputs + nb_outputs));
  const nvinfer1::PluginTensorDesc &in = in_out[pos];
  if (pos == 0) {
    if (with_fp16_) {
      return ((in.type == nvinfer1::DataType::kHALF) &&
              (in.format == nvinfer1::PluginFormat::kLINEAR));
    } else {
      return (in.type == nvinfer1::DataType::kFLOAT) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    }
  }
  const nvinfer1::PluginTensorDesc &prev = in_out[pos - 1];
  // output
  return in.type == prev.type && in.format == prev.format;
}

nvinfer1::DataType GroupNormPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(index,
                    0,
                    platform::errors::InvalidArgument(
                        "The groupnorm Plugin only has one input, so the "
                        "index value should be 0, but get %d.",
                        index));
  PADDLE_ENFORCE_EQ((input_types[0] == nvinfer1::DataType::kFLOAT ||
                     input_types[0] == nvinfer1::DataType::kHALF),
                    true,
                    platform::errors::InvalidArgument(
                        "The input type should be half or float"));

  return input_types[0];
}
int GroupNormPluginDynamic::initialize() TRT_NOEXCEPT {
  if (with_fp16_ == false) {
    // if use fp32
    cudaMalloc(&scale_gpu_, sizeof(float) * scale_.size());
    cudaMalloc(&bias_gpu_, sizeof(float) * bias_.size());
    cudaMemcpy(scale_gpu_,
               scale_.data(),
               scale_.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(bias_gpu_,
               bias_.data(),
               bias_.size() * sizeof(float),
               cudaMemcpyHostToDevice);
  } else {
    // if use fp16
    std::vector<half> scale_half(scale_.size());
    std::vector<half> bias_half(bias_.size());
    for (int i = 0; i < scale_.size(); ++i) {
      scale_half[i] = static_cast<half>(scale_[i]);
    }
    for (int i = 0; i < bias_.size(); ++i) {
      bias_half[i] = static_cast<half>(bias_[i]);
    }
    cudaMalloc(&scale_gpu_, sizeof(half) * scale_.size());
    cudaMalloc(&bias_gpu_, sizeof(half) * bias_.size());
    cudaMemcpy(scale_gpu_,
               scale_half.data(),
               scale_half.size() * sizeof(half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(bias_gpu_,
               bias_half.data(),
               bias_half.size() * sizeof(half),
               cudaMemcpyHostToDevice);
  }
  return 0;
}

int GroupNormPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  const auto &input_dims = input_desc[0].dims;
  int groups = groups_;
  float eps = eps_;

  std::vector<int> input_shape;
  for (int i = 0; i < input_dims.nbDims; i++) {
    input_shape.push_back(input_dims.d[i]);
  }

  const auto input_ddim = phi::make_ddim(input_shape);

  int C = input_shape[1];
  int image_size = input_shape[2] * input_shape[3];
  int batchSize = input_shape[0];
  std::vector<int64_t> batched_mean_shape = {batchSize * mean_shape_[0]};
  std::vector<int64_t> batched_variance_shape = {batchSize *
                                                 variance_shape_[0]};
  PADDLE_ENFORCE_EQ(
      C,
      scale_.size(),
      platform::errors::InvalidArgument(
          "scale's size should be equal to the channel number in groupnorm,"
          "but got feature_size:%d, scale's size:%d.",
          C,
          scale_.size()));
  PADDLE_ENFORCE_EQ(
      C,
      bias_.size(),
      platform::errors::InvalidArgument(
          "bias's size should be equal to the channel number in groupnorm,"
          "but got feature_size:%d, bias's size:%d.",
          C,
          bias_.size()));

  // phi::DenseTensor mean_t;
  // phi::DenseTensor variance_t;
  // phi::DenseTensor temp_variance_t;
  float *mean_d = static_cast<float *>(workspace);
  float *variance_d = mean_d + input_shape[0] * groups_;
  float *temp_variance_d = variance_d + input_shape[0] * groups_;
  auto input_type = input_desc[0].type;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. GroupNorm-->fp32";
    const float *input = reinterpret_cast<const float *>(inputs[0]);
    float *output = static_cast<float *>(outputs[0]);
    phi::GroupNormDirectCUDAFunctor<float, float> group_norm;
    group_norm(stream,
               input,
               input_shape,
               reinterpret_cast<float *>(bias_gpu_),
               reinterpret_cast<float *>(scale_gpu_),
               temp_variance_d,
               groups,
               eps,
               output,
               mean_d,
               variance_d,
               DataLayout::kNCHW);
  } else if (input_type == nvinfer1::DataType::kHALF) {
    VLOG(1) << "TRT Plugin DataType selected. GroupNorm-->fp16";
    const half *input = reinterpret_cast<const half *>(inputs[0]);
    half *output = static_cast<half *>(outputs[0]);

    phi::GroupNormDirectCUDAFunctor<half, float> group_norm;
    group_norm(stream,
               input,
               input_shape,
               reinterpret_cast<half *>(bias_gpu_),
               reinterpret_cast<half *>(scale_gpu_),
               temp_variance_d,
               groups,
               eps,
               output,
               mean_d,
               variance_d,
               DataLayout::kNCHW);
  } else {
    // input not float
    PADDLE_THROW(platform::errors::Fatal(
        "The Groupnorm TRT Plugin's only support fp32 input"));
  }
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
