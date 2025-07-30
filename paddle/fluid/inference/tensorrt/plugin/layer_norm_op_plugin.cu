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
#include "paddle/phi/kernels/layer_norm_kernel.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

int LayerNormPlugin::initialize() TRT_NOEXCEPT {
  cudaMalloc(&bias_gpu_, sizeof(float) * bias_.size());
  cudaMemcpy(bias_gpu_,
             bias_.data(),
             bias_.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMalloc(&scale_gpu_, sizeof(float) * scale_.size());
  cudaMemcpy(scale_gpu_,
             scale_.data(),
             scale_.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  return 0;
}

void LayerNormPlugin::terminate() TRT_NOEXCEPT {
  if (bias_gpu_) {
    cudaFree(bias_gpu_);
    bias_gpu_ = nullptr;
  }
  if (scale_gpu_) {
    cudaFree(scale_gpu_);
    scale_gpu_ = nullptr;
  }
}

nvinfer1::Dims LayerNormPlugin::getOutputDimensions(
    int index, const nvinfer1::Dims *inputDims, int nbInputs) TRT_NOEXCEPT {
  assert(nbInputs == 1);
  assert(index < this->getNbOutputs());
  nvinfer1::Dims const &input_dims = inputDims[0];
  nvinfer1::Dims output_dims = input_dims;
  return output_dims;
}

bool LayerNormPlugin::supportsFormat(
    nvinfer1::DataType type, nvinfer1::PluginFormat format) const TRT_NOEXCEPT {
  if (with_fp16_) {
    return ((type == nvinfer1::DataType::kFLOAT ||
             type == nvinfer1::DataType::kHALF) &&
            (format == nvinfer1::PluginFormat::kLINEAR));
  } else {
    return ((type == nvinfer1::DataType::kFLOAT) &&
            (format == nvinfer1::PluginFormat::kLINEAR));
  }
}

int LayerNormPlugin::enqueue(int batch_size,
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
  int begin_norm_axis = begin_norm_axis_;
  float eps = eps_;

  PADDLE_ENFORCE_EQ(1,
                    mean_shape_.size(),
                    common::errors::InvalidArgument(
                        "Size of mean_shape vector should be equal to 1,"
                        "but got Size of mean_shape vector:%d",
                        mean_shape_.size()));
  PADDLE_ENFORCE_EQ(1,
                    variance_shape_.size(),
                    common::errors::InvalidArgument(
                        "Size of variance_shape vector should be equal to 1,"
                        "but got Size of mean_shape vector:%d",
                        mean_shape_.size()));

  int64_t batched_mean_shape = mean_shape_[0] * input_dims.d[0];
  int64_t batched_variance_shape = variance_shape_[0] * input_dims.d[0];

  std::vector<int64_t> input_shape;
  input_shape.push_back(batch_size);
  for (int i = 0; i < input_dims.nbDims; i++) {
    input_shape.push_back(input_dims.d[i]);
  }
  const auto input_ddim = common::make_ddim(input_shape);
  auto matrix_dim = common::flatten_to_2d(input_ddim, begin_norm_axis);
  int feature_size = static_cast<int>(matrix_dim[1]);
  PADDLE_ENFORCE_EQ(feature_size,
                    scale_.size(),
                    common::errors::InvalidArgument(
                        "scale's size should be equal to the feature_size,"
                        "but got feature_size:%d, scale's size:%d.",
                        feature_size,
                        scale_.size()));
  PADDLE_ENFORCE_EQ(feature_size,
                    bias_.size(),
                    common::errors::InvalidArgument(
                        "bias's size should be equal to the feature_size,"
                        "but got feature_size:%d, bias's size:%d.",
                        feature_size,
                        bias_.size()));

  int device_id;
  cudaGetDevice(&device_id);
  mean_t.Resize(common::make_ddim({batched_mean_shape}));
  variance_t.Resize(common::make_ddim({batched_variance_shape}));
  float *mean_d = mean_t.mutable_data<float>(phi::GPUPlace(device_id));
  float *variance_d = variance_t.mutable_data<float>(phi::GPUPlace(device_id));
  auto input_type = getDataType();
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. LayerNorm-->fp32";
    const float *input = reinterpret_cast<const float *>(inputs[0]);
    float *output = static_cast<float *>(outputs[0]);
    phi::LayerNormDirectCUDAFunctor<float, float> layer_norm;
    layer_norm(stream,
               input,
               input_shape,
               bias_gpu_,
               scale_gpu_,
               output,
               mean_d,
               variance_d,
               begin_norm_axis,
               eps);
  } else if (input_type == nvinfer1::DataType::kHALF) {
    VLOG(1) << "TRT Plugin DataType selected. LayerNorm-->fp16";
    const half *input = reinterpret_cast<const half *>(inputs[0]);
    half *output = static_cast<half *>(outputs[0]);
    phi::LayerNormDirectCUDAFunctor<half, float> layer_norm;
    layer_norm(stream,
               input,
               input_shape,
               bias_gpu_,
               scale_gpu_,
               output,
               mean_d,
               variance_d,
               begin_norm_axis,
               eps);
  } else {
    PADDLE_THROW(common::errors::Fatal(
        "The LayerNorm TRT Plugin's input type should be float or half."));
  }
  return cudaGetLastError() != cudaSuccess;
}

int LayerNormPluginDynamic::initialize() TRT_NOEXCEPT {
  cudaMalloc(&bias_gpu_, sizeof(float) * bias_.size());
  cudaMemcpy(bias_gpu_,
             bias_.data(),
             bias_.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMalloc(&scale_gpu_, sizeof(float) * scale_.size());
  cudaMemcpy(scale_gpu_,
             scale_.data(),
             scale_.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  return 0;
}

void LayerNormPluginDynamic::terminate() TRT_NOEXCEPT {
  if (bias_gpu_) {
    cudaFree(bias_gpu_);
    bias_gpu_ = nullptr;
  }
  if (scale_gpu_) {
    cudaFree(scale_gpu_);
    scale_gpu_ = nullptr;
  }
}

nvinfer1::DimsExprs LayerNormPluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs *inputDims,
    int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  return inputDims[0];
}

bool LayerNormPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      common::errors::InvalidArgument(
          "The input of layernorm plugin should not be nullptr."));
  PADDLE_ENFORCE_LT(
      pos,
      nb_inputs + nb_outputs,
      common::errors::InvalidArgument("The pos(%d) should be less than the "
                                      "num(%d) of the input and the output.",
                                      pos,
                                      nb_inputs + nb_outputs));
  const nvinfer1::PluginTensorDesc &in = in_out[pos];
  if (pos == 0) {
    if (with_fp16_) {
      return ((in.type == nvinfer1::DataType::kFLOAT ||
               in.type == nvinfer1::DataType::kHALF) &&
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

void LayerNormPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *out,
    int nbOutputs) TRT_NOEXCEPT {
  const auto &input_dims = in[0].desc.dims;
  int statis_num = 1;
  for (int i = 0; i < begin_norm_axis_; i++) {
    statis_num *= input_dims.d[i];
  }
  mean_shape_[0] = statis_num;
  variance_shape_[0] = statis_num;
}

nvinfer1::DataType LayerNormPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(index,
                    0,
                    common::errors::InvalidArgument(
                        "The LayerNormPlugin only has one input, so the "
                        "index value should be 0, but get %d.",
                        index));
  PADDLE_ENFORCE_EQ((input_types[0] == nvinfer1::DataType::kFLOAT ||
                     input_types[0] == nvinfer1::DataType::kHALF),
                    true,
                    common::errors::InvalidArgument(
                        "The input type should be half or float"));
  return input_types[0];
}

int LayerNormPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  const auto &input_dims = input_desc[0].dims;
  int begin_norm_axis = begin_norm_axis_;
  float eps = eps_;

  std::vector<int64_t> input_shape;
  for (int i = 0; i < input_dims.nbDims; i++) {
    input_shape.push_back(input_dims.d[i]);
  }
  // in dynamic shape
  // the batch num should be involved in mean/variance shape
  PADDLE_ENFORCE_EQ(1,
                    mean_shape_.size(),
                    common::errors::InvalidArgument(
                        "Size of mean_shape vector should be equal to 1,"
                        "but got Size of mean_shape vector:%d",
                        mean_shape_.size()));
  PADDLE_ENFORCE_EQ(1,
                    variance_shape_.size(),
                    common::errors::InvalidArgument(
                        "Size of variance_shape vector should be equal to 1,"
                        "but got Size of mean_shape vector:%d",
                        mean_shape_.size()));
  PADDLE_ENFORCE_GE(mean_shape_[0],
                    0,
                    common::errors::InvalidArgument(
                        "The size of mean vector should be positive,"
                        "but got:%d",
                        mean_shape_[0]));
  PADDLE_ENFORCE_GE(variance_shape_[0],
                    0,
                    common::errors::InvalidArgument(
                        "The size of mean vector should be positive,"
                        "but got:%d",
                        variance_shape_[0]));

  const auto input_ddim = common::make_ddim(input_shape);
  auto matrix_dim = common::flatten_to_2d(input_ddim, begin_norm_axis);
  int feature_size = static_cast<int>(matrix_dim[1]);
  PADDLE_ENFORCE_EQ(feature_size,
                    scale_.size(),
                    common::errors::InvalidArgument(
                        "scale's size should be equal to the feature_size,"
                        "but got feature_size:%d, scale's size:%d.",
                        feature_size,
                        scale_.size()));
  PADDLE_ENFORCE_EQ(feature_size,
                    bias_.size(),
                    common::errors::InvalidArgument(
                        "bias's size should be equal to the feature_size,"
                        "but got feature_size:%d, bias's size:%d.",
                        feature_size,
                        bias_.size()));

  int device_id;
  cudaGetDevice(&device_id);
  mean_t.Resize(common::make_ddim(mean_shape_));
  variance_t.Resize(common::make_ddim(variance_shape_));
  float *mean_d = mean_t.mutable_data<float>(phi::GPUPlace(device_id));
  float *variance_d = variance_t.mutable_data<float>(phi::GPUPlace(device_id));
  auto input_type = input_desc[0].type;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. LayerNorm-->fp32";
    const float *input = reinterpret_cast<const float *>(inputs[0]);
    float *output = static_cast<float *>(outputs[0]);
    phi::LayerNormDirectCUDAFunctor<float, float> layer_norm;
    layer_norm(stream,
               input,
               input_shape,
               bias_gpu_,
               scale_gpu_,
               output,
               mean_d,
               variance_d,
               begin_norm_axis,
               eps);
  } else if (input_type == nvinfer1::DataType::kHALF) {
    VLOG(1) << "TRT Plugin DataType selected. LayerNorm-->fp16";
    const half *input = reinterpret_cast<const half *>(inputs[0]);
    half *output = static_cast<half *>(outputs[0]);
    phi::LayerNormDirectCUDAFunctor<half, float> layer_norm;
    layer_norm(stream,
               input,
               input_shape,
               bias_gpu_,
               scale_gpu_,
               output,
               mean_d,
               variance_d,
               begin_norm_axis,
               eps);
  } else {
    PADDLE_THROW(common::errors::Fatal(
        "The LayerNorm TRT Plugin's input type should be float or half."));
  }
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
