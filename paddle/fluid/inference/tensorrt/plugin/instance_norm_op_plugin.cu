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
#include "paddle/phi/core/platform/device/gpu/gpu_dnn.h"

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

int InstanceNormPlugin::initialize() TRT_NOEXCEPT { return 0; }

nvinfer1::Dims InstanceNormPlugin::getOutputDimensions(
    int index, const nvinfer1::Dims *inputDims, int nbInputs) TRT_NOEXCEPT {
  assert(nbInputs == 1);
  assert(index < this->getNbOutputs());
  nvinfer1::Dims const &input_dims = inputDims[0];
  nvinfer1::Dims output_dims = input_dims;
  return output_dims;
}

bool InstanceNormPlugin::supportsFormat(
    nvinfer1::DataType type, nvinfer1::PluginFormat format) const TRT_NOEXCEPT {
  return ((type == nvinfer1::DataType::kFLOAT ||
           type == nvinfer1::DataType::kHALF) &&
          (format == nvinfer1::PluginFormat::kLINEAR));
}

int InstanceNormPlugin::enqueue(int batch_size,
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
  int n = batch_size;
  int c = input_dims.d[0];
  int h = input_dims.d[1];
  int w = input_dims.d[2];

  scale_t.Resize(common::make_ddim({batch_size, c}));
  bias_t.Resize(common::make_ddim({batch_size, c}));
  int device_id;
  cudaGetDevice(&device_id);
  float *scale_d = scale_t.mutable_data<float>(phi::GPUPlace(device_id));
  float *bias_d = bias_t.mutable_data<float>(phi::GPUPlace(device_id));

  for (int i = 0; i < batch_size; i++) {
    cudaMemcpyAsync(scale_d + i * c,
                    scale_.data(),
                    sizeof(float) * c,
                    cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(bias_d + i * c,
                    bias_.data(),
                    sizeof(float) * c,
                    cudaMemcpyHostToDevice,
                    stream);
  }
  phi::dynload::cudnnSetTensor4dDescriptor(
      b_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, n * c, 1, 1);

  cudnnDataType_t cudnn_dtype;
  nvinfer1::DataType data_type = getDataType();
  convert_trt2cudnn_dtype(data_type, &cudnn_dtype);
  phi::dynload::cudnnSetTensor4dDescriptor(
      x_desc_, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n * c, h, w);
  phi::dynload::cudnnSetTensor4dDescriptor(
      y_desc_, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n * c, h, w);
  float alpha = 1;
  float beta = 0;
  phi::dynload::cudnnSetStream(handle_, stream);

  void const *x_ptr = inputs[0];
  void *y_ptr = outputs[0];
  phi::dynload::cudnnBatchNormalizationForwardTraining(
      handle_,
      CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
      &alpha,
      &beta,
      x_desc_,
      x_ptr,
      y_desc_,
      y_ptr,
      b_desc_,
      scale_d,
      bias_d,
      1.,
      nullptr,
      nullptr,
      eps_,
      nullptr,
      nullptr);
  return cudaGetLastError() != cudaSuccess;
}

int InstanceNormPluginEnqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                              const nvinfer1::PluginTensorDesc *outputDesc,
                              const void *const *inputs,
                              void *const *outputs,
                              void *workspace,
                              const float *scale,
                              const float *bias,
                              float eps,
                              cudnnTensorDescriptor_t x_desc_,
                              cudnnTensorDescriptor_t y_desc_,
                              cudnnTensorDescriptor_t b_desc_,
                              cudnnHandle_t handle_,
                              cudaStream_t stream) {
  nvinfer1::Dims input_dims = inputDesc[0].dims;
  int n = input_dims.d[0];
  int c = input_dims.d[1];
  int h = input_dims.d[2];
  int w = input_dims.d[3];
  phi::DenseTensor scale_t;
  phi::DenseTensor bias_t;
  scale_t.Resize(common::make_ddim({n, c}));
  bias_t.Resize(common::make_ddim({n, c}));
  int device_id;
  cudaGetDevice(&device_id);
  float *scale_d = scale_t.mutable_data<float>(phi::GPUPlace(device_id));
  float *bias_d = bias_t.mutable_data<float>(phi::GPUPlace(device_id));

  for (int i = 0; i < n; i++) {
    cudaMemcpyAsync(scale_d + i * c,
                    scale,
                    sizeof(float) * c,
                    cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(bias_d + i * c,
                    bias,
                    sizeof(float) * c,
                    cudaMemcpyHostToDevice,
                    stream);
  }
  phi::dynload::cudnnSetTensor4dDescriptor(
      b_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, n * c, 1, 1);

  cudnnDataType_t cudnn_dtype;
  auto data_type = inputDesc[0].type;
  convert_trt2cudnn_dtype(data_type, &cudnn_dtype);
  phi::dynload::cudnnSetTensor4dDescriptor(
      x_desc_, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n * c, h, w);
  phi::dynload::cudnnSetTensor4dDescriptor(
      y_desc_, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n * c, h, w);
  float alpha = 1;
  float beta = 0;
  phi::dynload::cudnnSetStream(handle_, stream);

  void const *x_ptr = inputs[0];
  void *y_ptr = outputs[0];
  phi::dynload::cudnnBatchNormalizationForwardTraining(
      handle_,
      CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
      &alpha,
      &beta,
      x_desc_,
      x_ptr,
      y_desc_,
      y_ptr,
      b_desc_,
      scale_d,
      bias_d,
      1.,
      nullptr,
      nullptr,
      eps,
      nullptr,
      nullptr);
  return cudaGetLastError() != cudaSuccess;
}

int InstanceNormPluginDynamic::initialize() TRT_NOEXCEPT { return 0; }

nvinfer1::DimsExprs InstanceNormPluginDynamic::getOutputDimensions(
    int index,
    const nvinfer1::DimsExprs *inputs,
    int nbInputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  assert(nbInputs == 1);
  assert(index < this->getNbOutputs());
  nvinfer1::DimsExprs output(inputs[0]);
  return output;
}

bool InstanceNormPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *inOut,
    int nbInputs,
    int nbOutputs) TRT_NOEXCEPT {
  assert(inOut && pos < (nbInputs + nbOutputs));
  assert(pos == 0 || pos == 1);
  return ((inOut[pos].type == nvinfer1::DataType::kFLOAT ||
           inOut[pos].type == nvinfer1::DataType::kHALF) &&
          (inOut[pos].format == nvinfer1::PluginFormat::kLINEAR) &&
          inOut[pos].type == inOut[0].type);
}

int InstanceNormPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *inputDesc,
    const nvinfer1::PluginTensorDesc *outputDesc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  return InstanceNormPluginEnqueue(inputDesc,
                                   outputDesc,
                                   inputs,
                                   outputs,
                                   workspace,
                                   scale_.data(),
                                   bias_.data(),
                                   eps_,
                                   x_desc_,
                                   y_desc_,
                                   b_desc_,
                                   handle_,
                                   stream);
}

nvinfer1::DataType InstanceNormPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *inputTypes,
    int nbInputs) const TRT_NOEXCEPT {
  assert(inputTypes && nbInputs > 0 && index == 0);
  return inputTypes[0];
}

void InstanceNormPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *out,
    int nbOutputs) TRT_NOEXCEPT {}

int PIRInstanceNormPlugin::initialize() TRT_NOEXCEPT { return 0; }

nvinfer1::DimsExprs PIRInstanceNormPlugin::getOutputDimensions(
    int index,
    const nvinfer1::DimsExprs *inputs,
    int nbInputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  assert(nbInputs == 1);
  assert(index < this->getNbOutputs());
  nvinfer1::DimsExprs output(inputs[0]);
  return output;
}

bool PIRInstanceNormPlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *inOut,
    int nbInputs,
    int nbOutputs) TRT_NOEXCEPT {
  assert(inOut && pos < (nbInputs + nbOutputs));
  assert(pos == 0 || pos == 1);
  return ((inOut[pos].type == nvinfer1::DataType::kFLOAT ||
           inOut[pos].type == nvinfer1::DataType::kHALF) &&
          (inOut[pos].format == nvinfer1::PluginFormat::kLINEAR) &&
          inOut[pos].type == inOut[0].type);
}

int PIRInstanceNormPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                   const nvinfer1::PluginTensorDesc *outputDesc,
                                   const void *const *inputs,
                                   void *const *outputs,
                                   void *workspace,
                                   cudaStream_t stream) TRT_NOEXCEPT {
  const float *scale_ = reinterpret_cast<const float *>(inputs[1]);
  const float *bias_ = reinterpret_cast<const float *>(inputs[2]);
  return InstanceNormPluginEnqueue(inputDesc,
                                   outputDesc,
                                   inputs,
                                   outputs,
                                   workspace,
                                   scale_,
                                   bias_,
                                   eps_,
                                   x_desc_,
                                   y_desc_,
                                   b_desc_,
                                   handle_,
                                   stream);
}

nvinfer1::DataType PIRInstanceNormPlugin::getOutputDataType(
    int index,
    const nvinfer1::DataType *inputTypes,
    int nbInputs) const TRT_NOEXCEPT {
  assert(inputTypes && nbInputs > 0 && index == 0);
  return inputTypes[0];
}

void PIRInstanceNormPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *out,
    int nbOutputs) TRT_NOEXCEPT {}

nvinfer1::IPluginV2 *PIRInstanceNormPluginCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
  float epsilon = 1e-8;
  for (int i = 0; i < fc->nbFields; ++i) {
    const std::string field_name(fc->fields[i].name);
    if (field_name.compare("epsilon") == 0) {
      epsilon = *static_cast<const float *>(fc->fields[i].data);
    } else {
      assert(false && "unknown plugin field name.");
    }
  }
  return new PIRInstanceNormPlugin(epsilon);
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
