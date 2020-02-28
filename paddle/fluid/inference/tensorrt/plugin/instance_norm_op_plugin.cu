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
#include "paddle/fluid/inference/tensorrt/plugin/instance_norm_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

cudnnStatus_t convert_trt2cudnn_dtype(nvinfer1::DataType trt_dtype,
                                      cudnnDataType_t *cudnn_dtype) {
  switch (trt_dtype) {
    case nvinfer1::DataType::kFLOAT:
      *cudnn_dtype = CUDNN_DATA_FLOAT;
      break;
    case nvinfer1::DataType::kHALF:
      *cudnn_dtype = CUDNN_DATA_HALF;
      break;
    default:
      return CUDNN_STATUS_BAD_PARAM;
  }
  return CUDNN_STATUS_SUCCESS;
}

template <typename T>
T *CreateInstanceNormPluginDeserialize(const void *buffer, size_t length) {
  return new T(buffer, length);
}
REGISTER_TRT_PLUGIN("instance_norm_plugin", PluginTensorRT,
                    CreateInstanceNormPluginDeserialize<InstanceNormPlugin>);

int InstanceNormPlugin::initialize() {
  platform::dynload::cudnnCreate(&handle_);
  platform::dynload::cudnnCreateTensorDescriptor(&x_desc_);
  platform::dynload::cudnnCreateTensorDescriptor(&y_desc_);
  platform::dynload::cudnnCreateTensorDescriptor(&b_desc_);
  return 0;
}

nvinfer1::Dims InstanceNormPlugin::getOutputDimensions(
    int index, const nvinfer1::Dims *inputDims, int nbInputs) {
  assert(nbInputs == 1);
  assert(index < this->getNbOutputs());
  nvinfer1::Dims const &input_dims = inputDims[0];
  nvinfer1::Dims output_dims = input_dims;
  return output_dims;
}

int InstanceNormPlugin::enqueue(int batch_size, const void *const *inputs,
                                void **outputs, void *workspace,
                                cudaStream_t stream) {
  const auto &input_dims = this->getInputDims(0);

  PADDLE_ENFORCE_EQ(input_dims.nbDims, 3,
                    platform::errors::InvalidArgument(
                        "Input Dims should be 3 (except the batch), got %d",
                        input_dims.nbDims));
  int n = batch_size;
  int c = input_dims.d[0];
  int h = input_dims.d[1];
  int w = input_dims.d[2];

  scale_t.Resize(framework::make_ddim({n, c}));
  bias_t.Resize(framework::make_ddim({n, c}));
  int device_id;
  cudaGetDevice(&device_id);
  float *scale_d = scale_t.mutable_data<float>(platform::CUDAPlace(device_id));
  float *bias_d = bias_t.mutable_data<float>(platform::CUDAPlace(device_id));

  for (int i = 0; i < n; i++) {
    cudaMemcpyAsync(scale_d + i * c, scale_.data(), sizeof(float) * c,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(bias_d + i * c, bias_.data(), sizeof(float) * c,
                    cudaMemcpyHostToDevice, stream);
  }
  platform::dynload::cudnnSetTensor4dDescriptor(
      b_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, n * c, 1, 1);

  cudnnDataType_t cudnn_dtype;
  nvinfer1::DataType data_type = getDataType();
  convert_trt2cudnn_dtype(data_type, &cudnn_dtype);
  platform::dynload::cudnnSetTensor4dDescriptor(x_desc_, CUDNN_TENSOR_NCHW,
                                                cudnn_dtype, 1, n * c, h, w);
  platform::dynload::cudnnSetTensor4dDescriptor(y_desc_, CUDNN_TENSOR_NCHW,
                                                cudnn_dtype, 1, n * c, h, w);
  float alpha = 1;
  float beta = 0;
  platform::dynload::cudnnSetStream(handle_, stream);

  void const *x_ptr = inputs[0];
  void *y_ptr = outputs[0];
  platform::dynload::cudnnBatchNormalizationForwardTraining(
      handle_, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &alpha, &beta, x_desc_,
      x_ptr, y_desc_, y_ptr, b_desc_, scale_d, bias_d, 1., nullptr, nullptr,
      eps_, nullptr, nullptr);
  return cudaGetLastError() != cudaSuccess;
}

// Dynamic Plugin below.
#if IS_TRT_VERSION_GE(6000)

int InstanceNormPluginDynamic::initialize() {
  platform::dynload::cudnnCreate(&handle_);
  platform::dynload::cudnnCreateTensorDescriptor(&x_desc_);
  platform::dynload::cudnnCreateTensorDescriptor(&y_desc_);
  platform::dynload::cudnnCreateTensorDescriptor(&b_desc_);
  return 0;
}

size_t InstanceNormPluginDynamic::getSerializationSize() const {
  return getBaseSerializationSize() + SerializedSize(eps_) +
         SerializedSize(scale_) + SerializedSize(bias_) +
         SerializedSize(getPluginType());
}

void InstanceNormPluginDynamic::serialize(void *buffer) const {
  SerializeValue(&buffer, getPluginType());
  serializeBase(buffer);
  SerializeValue(&buffer, eps_);
  SerializeValue(&buffer, scale_);
  SerializeValue(&buffer, bias_);
}

nvinfer1::DimsExprs InstanceNormPluginDynamic::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs *inputs, int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) {
  return inputs[0];
}

bool InstanceNormPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *in_out, int nb_inputs,
    int nb_outputs) {
  PADDLE_ENFORCE_NOT_NULL(
      in_out, platform::errors::InvalidArgument(
                  "The input of swish plugin shoule not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos, nb_inputs + nb_outputs,
      platform::errors::InvalidArgument("The pos(%d) should be less than the "
                                        "num(%d) of the input and the output.",
                                        pos, nb_inputs + nb_outputs));
  (in_out && pos < (nb_inputs + nb_outputs));

  return ((in_out[pos].type == nvinfer1::DataType::kFLOAT ||
           in_out[pos].type == nvinfer1::DataType::kHALF) &&
          in_out[pos].format == nvinfer1::PluginFormat::kNCHW);
}

nvinfer1::DataType InstanceNormPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *input_types, int nb_inputs) const {
  PADDLE_ENFORCE_EQ(index, 0,
                    platform::errors::InvalidArgument(
                        "The instance norm Plugin only has one input, so the "
                        "index value should be 0, but get %d.",
                        index));
  PADDLE_ENFORCE_EQ((input_types[0] == nvinfer1::DataType::kFLOAT ||
                     input_types[0] == nvinfer1::DataType::kHALF),
                    true, platform::errors::InvalidArgument(
                              "The input type should be half or float"));
  return input_types[0];
}

int InstanceNormPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc, const void *const *inputs,
    void *const *outputs, void *workspace, cudaStream_t stream) {
  const auto &input_dims = input_desc[0].dims;

  PADDLE_ENFORCE_EQ(input_dims.nbDims, 4,
                    platform::errors::InvalidArgument(
                        "Input Dims should be 4, got %d", input_dims.nbDims));
  int n = input_dims.d[0];
  int c = input_dims.d[1];
  int h = input_dims.d[2];
  int w = input_dims.d[3];

  scale_t.Resize(framework::make_ddim({n, c}));
  bias_t.Resize(framework::make_ddim({n, c}));
  int device_id;
  cudaGetDevice(&device_id);
  float *scale_d = scale_t.mutable_data<float>(platform::CUDAPlace(device_id));
  float *bias_d = bias_t.mutable_data<float>(platform::CUDAPlace(device_id));

  for (int i = 0; i < n; i++) {
    cudaMemcpyAsync(scale_d + i * c, scale_.data(), sizeof(float) * c,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(bias_d + i * c, bias_.data(), sizeof(float) * c,
                    cudaMemcpyHostToDevice, stream);
  }
  platform::dynload::cudnnSetTensor4dDescriptor(
      b_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, n * c, 1, 1);

  cudnnDataType_t cudnn_dtype;
  nvinfer1::DataType data_type = input_desc[0].type;
  convert_trt2cudnn_dtype(data_type, &cudnn_dtype);
  platform::dynload::cudnnSetTensor4dDescriptor(x_desc_, CUDNN_TENSOR_NCHW,
                                                cudnn_dtype, 1, n * c, h, w);
  platform::dynload::cudnnSetTensor4dDescriptor(y_desc_, CUDNN_TENSOR_NCHW,
                                                cudnn_dtype, 1, n * c, h, w);
  float alpha = 1;
  float beta = 0;
  platform::dynload::cudnnSetStream(handle_, stream);

  void const *x_ptr = inputs[0];
  void *y_ptr = outputs[0];
  platform::dynload::cudnnBatchNormalizationForwardTraining(
      handle_, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &alpha, &beta, x_desc_,
      x_ptr, y_desc_, y_ptr, b_desc_, scale_d, bias_d, 1., nullptr, nullptr,
      eps_, nullptr, nullptr);
  return cudaGetLastError() != cudaSuccess;
}
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
