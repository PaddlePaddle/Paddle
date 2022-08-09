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
using DataLayout = framework::DataLayout;

int GroupNormPlugin::initialize() TRT_NOEXCEPT { return 0; }

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

  int device_id;
  cudaGetDevice(&device_id);
  const float *input = static_cast<const float *>(inputs[0]);
  float *output = static_cast<float *>(outputs[0]);

  scale_t.Resize(phi::make_ddim({C}));
  bias_t.Resize(phi::make_ddim({C}));

  mean_t.Resize(phi::make_ddim(mean_shape_));
  variance_t.Resize(phi::make_ddim(variance_shape_));
  float *scale_d = scale_t.mutable_data<float>(platform::CUDAPlace(device_id));
  float *bias_d = bias_t.mutable_data<float>(platform::CUDAPlace(device_id));
  float *mean_d = mean_t.mutable_data<float>(platform::CUDAPlace(device_id));
  float *variance_d =
      variance_t.mutable_data<float>(platform::CUDAPlace(device_id));

  framework::Tensor temp_variance_t;
  temp_variance_t.Resize(phi::make_ddim(variance_shape_));
  float *temp_variance_d =
      temp_variance_t.mutable_data<float>(platform::CUDAPlace(device_id));
  cudaMemcpyAsync(scale_d,
                  scale_.data(),
                  sizeof(float) * C,
                  cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(
      bias_d, bias_.data(), sizeof(float) * C, cudaMemcpyHostToDevice, stream);
  phi::GroupNormDirectCUDAFunctor<float> group_norm;
  group_norm(stream,
             input,
             input_shape,
             bias_d,
             scale_d,
             mean_d,
             temp_variance_d,
             groups_,
             eps_,
             output,
             mean_d,
             variance_d,
             DataLayout::kNCHW);
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
    return (in.type == nvinfer1::DataType::kFLOAT) &&
           (in.format == nvinfer1::TensorFormat::kLINEAR);
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
  return input_types[0];
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

  int device_id;
  cudaGetDevice(&device_id);
  auto input_type = input_desc[0].type;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    const float *input = static_cast<const float *>(inputs[0]);
    float *output = static_cast<float *>(outputs[0]);
    scale_t.Resize(phi::make_ddim({C}));
    bias_t.Resize(phi::make_ddim({C}));

    mean_t.Resize(phi::make_ddim(batched_mean_shape));
    variance_t.Resize(phi::make_ddim(batched_variance_shape));
    float *scale_d =
        scale_t.mutable_data<float>(platform::CUDAPlace(device_id));
    float *bias_d = bias_t.mutable_data<float>(platform::CUDAPlace(device_id));
    float *mean_d = mean_t.mutable_data<float>(platform::CUDAPlace(device_id));
    float *variance_d =
        variance_t.mutable_data<float>(platform::CUDAPlace(device_id));

    framework::Tensor temp_variance_t;
    temp_variance_t.Resize(phi::make_ddim(batched_variance_shape));
    float *temp_variance_d =
        temp_variance_t.mutable_data<float>(platform::CUDAPlace(device_id));
    cudaMemcpyAsync(scale_d,
                    scale_.data(),
                    sizeof(float) * C,
                    cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(bias_d,
                    bias_.data(),
                    sizeof(float) * C,
                    cudaMemcpyHostToDevice,
                    stream);

    phi::GroupNormDirectCUDAFunctor<float> group_norm;
    group_norm(stream,
               input,
               input_shape,
               bias_d,
               scale_d,
               mean_d,
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
