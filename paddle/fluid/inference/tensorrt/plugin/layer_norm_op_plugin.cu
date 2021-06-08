// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdio.h>
#include <cassert>
#include <vector>
#include "glog/logging.h"
#include "paddle/fluid/inference/tensorrt/plugin/layer_norm_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"
#include "paddle/fluid/operators/layer_norm_op.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

LayerNormPlugin *CreateLayerNormPluginDeserialize(const void *buffer,
                                                  size_t length) {
  return new LayerNormPlugin(buffer, length);
}
REGISTER_TRT_PLUGIN("layer_norm_plugin", CreateLayerNormPluginDeserialize);

int LayerNormPlugin::initialize() { return 0; }

nvinfer1::Dims LayerNormPlugin::getOutputDimensions(
    int index, const nvinfer1::Dims *inputDims, int nbInputs) {
  assert(nbInputs == 1);
  assert(index < this->getNbOutputs());
  nvinfer1::Dims const &input_dims = inputDims[0];
  nvinfer1::Dims output_dims = input_dims;
  return output_dims;
}

int LayerNormPlugin::enqueue(int batch_size, const void *const *inputs,
                             void **outputs, void *workspace,
                             cudaStream_t stream) {
  const auto &input_dims = this->getInputDims(0);
  const float *input = reinterpret_cast<const float *>(inputs[0]);
  float *output = reinterpret_cast<float **>(outputs)[0];
  int begin_norm_axis = begin_norm_axis_;
  float eps = eps_;

  std::vector<int> input_shape;
  input_shape.push_back(batch_size);
  for (int i = 0; i < input_dims.nbDims; i++) {
    input_shape.push_back(input_dims.d[i]);
  }
  const auto input_ddim = framework::make_ddim(input_shape);
  auto matrix_dim = framework::flatten_to_2d(input_ddim, begin_norm_axis);
  int feature_size = static_cast<int>(matrix_dim[1]);
  PADDLE_ENFORCE_EQ(feature_size, scale_.size(),
                    platform::errors::InvalidArgument(
                        "scale's size should be equal to the feature_size,"
                        "but got feature_size:%d, scale's size:%d.",
                        feature_size, scale_.size()));
  PADDLE_ENFORCE_EQ(feature_size, bias_.size(),
                    platform::errors::InvalidArgument(
                        "bias's size should be equal to the feature_size,"
                        "but got feature_size:%d, bias's size:%d.",
                        feature_size, bias_.size()));

  scale_t.Resize(framework::make_ddim({feature_size}));
  bias_t.Resize(framework::make_ddim({feature_size}));
  mean_t.Resize(framework::make_ddim(mean_shape_));
  variance_t.Resize(framework::make_ddim(variance_shape_));
  int device_id;
  cudaGetDevice(&device_id);
  float *scale_d = scale_t.mutable_data<float>(platform::CUDAPlace(device_id));
  float *bias_d = bias_t.mutable_data<float>(platform::CUDAPlace(device_id));
  float *mean_d = mean_t.mutable_data<float>(platform::CUDAPlace(device_id));
  float *variance_d =
      variance_t.mutable_data<float>(platform::CUDAPlace(device_id));
  cudaMemcpyAsync(scale_d, scale_.data(), sizeof(float) * feature_size,
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(bias_d, bias_.data(), sizeof(float) * feature_size,
                  cudaMemcpyHostToDevice, stream);

  paddle::operators::LayerNormDirectCUDAFunctor<float> layer_norm;
  layer_norm(stream, input, input_shape, bias_d, scale_d, output, mean_d,
             variance_d, begin_norm_axis, eps);
  return cudaGetLastError() != cudaSuccess;
}

nvinfer1::DimsExprs LayerNormPluginDynamic::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs *inputDims, int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) {
  return inputDims[0];
}

bool LayerNormPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *in_out, int nb_inputs,
    int nb_outputs) {
  PADDLE_ENFORCE_NOT_NULL(
      in_out, platform::errors::InvalidArgument(
                  "The input of layernorm plugin shoule not be nullptr."));
  PADDLE_ENFORCE_LT(
      pos, nb_inputs + nb_outputs,
      platform::errors::InvalidArgument("The pos(%d) should be less than the "
                                        "num(%d) of the input and the output.",
                                        pos, nb_inputs + nb_outputs));
  const nvinfer1::PluginTensorDesc &in = in_out[pos];
  if (pos == 0) {
    // TODO(Shangzhizhou) FP16 support
    return (in.type == nvinfer1::DataType::kFLOAT) &&
           (in.format == nvinfer1::TensorFormat::kLINEAR);
  }
  const nvinfer1::PluginTensorDesc &prev = in_out[pos - 1];
  // output
  return in.type == prev.type && in.format == prev.format;
}

nvinfer1::DataType LayerNormPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *input_types, int nb_inputs) const {
  PADDLE_ENFORCE_EQ(index, 0,
                    platform::errors::InvalidArgument(
                        "The LayerNormPlugin only has one input, so the "
                        "index value should be 0, but get %d.",
                        index));
  return input_types[0];
}

int LayerNormPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc, const void *const *inputs,
    void *const *outputs, void *workspace, cudaStream_t stream) {
  const auto &input_dims = input_desc[0].dims;
  int begin_norm_axis = begin_norm_axis_;
  float eps = eps_;

  std::vector<int> input_shape;
  for (int i = 0; i < input_dims.nbDims; i++) {
    input_shape.push_back(input_dims.d[i]);
  }
  const auto input_ddim = framework::make_ddim(input_shape);
  auto matrix_dim = framework::flatten_to_2d(input_ddim, begin_norm_axis);
  int feature_size = static_cast<int>(matrix_dim[1]);
  PADDLE_ENFORCE_EQ(feature_size, scale_.size(),
                    platform::errors::InvalidArgument(
                        "scale's size should be equal to the feature_size,"
                        "but got feature_size:%d, scale's size:%d.",
                        feature_size, scale_.size()));
  PADDLE_ENFORCE_EQ(feature_size, bias_.size(),
                    platform::errors::InvalidArgument(
                        "bias's size should be equal to the feature_size,"
                        "but got feature_size:%d, bias's size:%d.",
                        feature_size, bias_.size()));
  int device_id;
  cudaGetDevice(&device_id);
  auto input_type = input_desc[0].type;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. LayerNorm-->fp32";
    const float *input = reinterpret_cast<const float *>(inputs[0]);
    float *output = static_cast<float *>(outputs[0]);
    scale_t.Resize(framework::make_ddim({feature_size}));
    bias_t.Resize(framework::make_ddim({feature_size}));
    mean_t.Resize(framework::make_ddim(mean_shape_));
    variance_t.Resize(framework::make_ddim(variance_shape_));

    float *scale_d =
        scale_t.mutable_data<float>(platform::CUDAPlace(device_id));
    float *bias_d = bias_t.mutable_data<float>(platform::CUDAPlace(device_id));
    float *mean_d = mean_t.mutable_data<float>(platform::CUDAPlace(device_id));
    float *variance_d =
        variance_t.mutable_data<float>(platform::CUDAPlace(device_id));

    cudaMemcpyAsync(scale_d, scale_.data(), sizeof(float) * feature_size,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(bias_d, bias_.data(), sizeof(float) * feature_size,
                    cudaMemcpyHostToDevice, stream);

    paddle::operators::LayerNormDirectCUDAFunctor<float> layer_norm;
    layer_norm(stream, input, input_shape, bias_d, scale_d, output, mean_d,
               variance_d, begin_norm_axis, eps);
  } else if (input_type == nvinfer1::DataType::kHALF) {
#ifdef TRT_PLUGIN_FP16_AVALIABLE
    VLOG(1) << "TRT Plugin DataType selected. LayerNorm-->fp16";
    const half *input = reinterpret_cast<const half *>(inputs[0]);
    half *output = static_cast<half *>(outputs[0]);
    size_t mean_shape_product = 1;
    for (auto s : mean_shape_) {
      mean_shape_product *= s;
    }
    size_t variance_shape_product = 1;
    for (auto s : variance_shape_) {
      variance_shape_product *= s;
    }
    if (!scale_gpu_half_d_) {
      cudaMalloc(&scale_gpu_half_d_, feature_size * sizeof(half));
    }
    if (!bias_gpu_half_d_) {
      cudaMalloc(&bias_gpu_half_d_, feature_size * sizeof(half));
    }
    if (!mean_gpu_half_d_) {
      cudaMalloc(&mean_gpu_half_d_, mean_shape_product * sizeof(half));
    }
    if (!variance_gpu_half_d_) {
      cudaMalloc(&variance_gpu_half_d_, variance_shape_product * sizeof(half));
    }

    half *scale_cpu_half =
        static_cast<half *>(malloc(feature_size * sizeof(half)));
    half *bias_cpu_half =
        static_cast<half *>(malloc(feature_size * sizeof(half)));
    PADDLE_ENFORCE_EQ(
        scale_cpu_half && bias_cpu_half, true,
        platform::errors::Unavailable("Out of memory, malloc size %d.",
                                      feature_size * sizeof(half)));

    for (int i = 0; i < feature_size; i++) {
      scale_cpu_half[i] = static_cast<half>(scale_[i]);
      bias_cpu_half[i] = static_cast<half>(bias_[i]);
    }
    cudaMemcpyAsync(scale_gpu_half_d_, scale_cpu_half,
                    sizeof(half) * feature_size, cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(bias_gpu_half_d_, bias_cpu_half,
                    sizeof(half) * feature_size, cudaMemcpyHostToDevice,
                    stream);
    free(scale_cpu_half);
    free(bias_cpu_half);

    paddle::operators::LayerNormDirectCUDAFunctor<half> layer_norm;
    layer_norm(stream, input, input_shape, bias_gpu_half_d_, scale_gpu_half_d_,
               output, mean_gpu_half_d_, variance_gpu_half_d_, begin_norm_axis,
               eps);
#else
    PADDLE_THROW(platform::errors::Fatal(
        "The layer_norm tensorRT plugin should be "
        "complied with CUDA version >= 10.0 when running with fp16. "
        "Please recomplie it or try to use fp32 by set "
        "config.SetTRTDynamicShapeInfo(min_input_shape, "
        "max_input_shape, opt_input_shape, true"));
#endif
  } else {
    PADDLE_THROW(platform::errors::Fatal(
        "The LayerNorm TRT Plugin's input type should be float or half."));
  }
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
